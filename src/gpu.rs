use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    },
    nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts},
};
use std::{
    cell::{RefCell, RefMut},
    sync::Arc,
    time::Instant,
};
use zeroize::Zeroizing;

pub struct Gpu {
    pub name: String,
    pub kernel_cache_hit: bool,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    init: CudaFunction,
    step: CudaFunction,
    finish: CudaFunction,
    validate: CudaFunction,
    init_indexed: CudaFunction,
    phrase_entropy: CudaFunction,
    bip39_init: CudaFunction,
    bip39_slip10: CudaFunction,
    wallet_addresses: CudaFunction,
    filter_addresses: CudaFunction,
    base_table: CudaSlice<u16>,
    // Dictionary is public and immutable. Secret workspaces are reusable but wiped
    // after each batch and again on drop, including an early return from an error.
    dictionary: CudaSlice<u8>,
    workspace: RefCell<Option<Workspace>>,
}

pub struct ResidentBatch {
    pub tested_wallets: usize,
    pub address_seconds: f64,
    pub address_checksum: u32,
    pub indexes: Vec<usize>,
    pub seeds: Zeroizing<Vec<[u8; 64]>>,
    pub generation_seconds: f64,
    pub validation_seconds: f64,
    pub derivation_seconds: f64,
    pub cleanup_seconds: f64,
}

/// A bounded VRAM arena. Capacity is separate from the logical length of a
/// batch; kernels always receive the logical length, never uninitialized tails.
struct Workspace {
    stream: Arc<CudaStream>,
    raw_capacity: usize,
    valid_capacity: usize,
    words: CudaSlice<u16>,
    entropy: CudaSlice<u8>,
    indexes: CudaSlice<u32>,
    count: CudaSlice<u32>,
    state: CudaSlice<u64>,
    output: CudaSlice<u8>,
    addresses: CudaSlice<u8>,
    winner: CudaSlice<u32>,
    checksum: CudaSlice<u32>,
}

impl Workspace {
    fn new(stream: Arc<CudaStream>, n: usize) -> Result<Self> {
        // 1/256 pass on average. Reserve twice that, but explicitly grow if an
        // unusually large passing set occurs; probabilities never set array bounds.
        let valid_capacity = n.div_ceil(128) + 128;
        Ok(Self {
            raw_capacity: n,
            valid_capacity,
            words: stream.alloc_zeros(n * 24)?,
            entropy: stream.alloc_zeros(n * 64)?,
            indexes: stream.alloc_zeros(n)?,
            count: stream.alloc_zeros(1)?,
            state: stream.alloc_zeros(valid_capacity * 32)?,
            output: stream.alloc_zeros(valid_capacity * 64)?,
            addresses: stream.alloc_zeros(valid_capacity * 36)?,
            winner: stream.alloc_zeros(1)?,
            checksum: stream.alloc_zeros(1)?,
            stream,
        })
    }
    fn reserve_valid(&mut self, n: usize) -> Result<()> {
        if n > self.valid_capacity {
            self.stream.memset_zeros(&mut self.state)?;
            self.stream.memset_zeros(&mut self.output)?;
            self.stream.memset_zeros(&mut self.addresses)?;
            let capacity = n.max(self.valid_capacity * 2);
            self.state = self.stream.alloc_zeros(capacity * 32)?;
            self.output = self.stream.alloc_zeros(capacity * 64)?;
            self.addresses = self.stream.alloc_zeros(capacity * 36)?;
            self.valid_capacity = capacity;
        }
        Ok(())
    }
    fn clear(&mut self) -> Result<()> {
        self.stream.memset_zeros(&mut self.words)?;
        self.stream.memset_zeros(&mut self.entropy)?;
        self.stream.memset_zeros(&mut self.indexes)?;
        self.stream.memset_zeros(&mut self.count)?;
        self.stream.memset_zeros(&mut self.state)?;
        self.stream.memset_zeros(&mut self.output)?;
        self.stream.memset_zeros(&mut self.addresses)?;
        self.stream.memset_zeros(&mut self.winner)?;
        self.stream.memset_zeros(&mut self.checksum)?;
        self.stream.synchronize()?;
        Ok(())
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = self.clear();
    }
}

impl Gpu {
    /// Bytes held by the reusable secret arena, excluding driver/JIT allocations
    /// and the immutable public dictionary. This is capacity, not live secret data.
    pub fn workspace_bytes(&self) -> Option<usize> {
        self.workspace.borrow().as_ref().map(|w| {
            w.words.len() * 2
                + w.entropy.len()
                + w.indexes.len() * 4
                + w.count.len() * 4
                + w.state.len() * 8
                + w.output.len()
                + w.addresses.len()
                + 8
        })
    }
    pub fn new(device: usize) -> Result<Self> {
        ensure!(
            unsafe { cudarc::driver::sys::is_culib_present() },
            "NVIDIA CUDA driver unavailable"
        );
        // Check first to turn a missing NVRTC into a useful error instead of a loader panic.
        ensure!(
            unsafe { cudarc::nvrtc::sys::is_culib_present() },
            "CUDA 13 NVRTC is unavailable. Install the CUDA runtime compiler, place its libraries beside blocknames, or pass --cuda-path <directory>. Use --backend cpu to run without CUDA."
        );
        let ctx = CudaContext::new(device).context("opening CUDA device")?;
        let name = ctx.name()?;
        let (major, minor) = ctx.compute_capability()?;
        let constants = format!(
            "\n__device__ __constant__ unsigned char TON_DATA[43]={{{}}};\n__device__ __constant__ unsigned char TON_STATE[71]={{{}}};\n",
            crate::ton::DATA_TEMPLATE
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(","),
            crate::ton::STATE_TEMPLATE
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let source = [
            include_str!("pbkdf2.cu"),
            &constants,
            include_str!("addresses.cu"),
        ]
        .concat();
        // Cache PUBLIC compiler output, never recovery material. Invalidate on
        // source, target architecture, flags or NVRTC version changes. Cache I/O
        // failures fall back to normal compilation, not a failed wallet search.
        use sha2::{Digest, Sha256};
        let (mut nv_major, mut nv_minor) = (0, 0);
        unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nv_major, &mut nv_minor) }.result()?;
        let flags = vec![
            format!("--gpu-architecture=compute_{major}{minor}"),
            "--std=c++14".into(),
        ];
        let fingerprint = hex::encode(Sha256::digest(format!(
            "{nv_major}.{nv_minor}|{flags:?}|{source}"
        )));
        let cache_path =
            crate::runtime::kernel_cache_dir().map(|d| d.join(format!("{fingerprint}.json")));
        let cached = cache_path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| {
                let code = v["ptx"].as_str()?;
                (v["sha256"].as_str()? == hex::encode(Sha256::digest(code.as_bytes())))
                    .then(|| code.to_owned())
            });
        let kernel_cache_hit = cached.is_some();
        let ptx = if let Some(code) = cached {
            Ptx::from_src(code)
        } else {
            let ptx = compile_ptx_with_opts(
                source,
                CompileOptions {
                    options: flags,
                    ..Default::default()
                },
            )
            .context("compiling CUDA kernels")?;
            if let Some(path) = &cache_path {
                let code = ptx.to_src();
                let data = serde_json::json!({"sha256":hex::encode(Sha256::digest(code.as_bytes())),"ptx":code});
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                let _ = std::fs::write(path, serde_json::to_vec(&data)?);
            }
            ptx
        };
        let module = ctx.load_module(ptx)?;
        let stream = ctx.default_stream();
        let base_table: Vec<u16> = include_bytes!("ed25519-base.bin")
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut dictionary = Vec::with_capacity(2048 * 9);
        for word in bip39::Language::English.word_list() {
            ensure!(word.len() <= 8, "unexpected BIP39 word length");
            dictionary.push(word.len() as u8);
            dictionary.extend(word.bytes());
            dictionary.resize(dictionary.len() + 8 - word.len(), 0);
        }
        Ok(Self {
            name,
            kernel_cache_hit,
            stream: ctx.default_stream(),
            init: module.load_function("init")?,
            step: module.load_function("step")?,
            finish: module.load_function("finish")?,
            validate: module.load_function("validate")?,
            init_indexed: module.load_function("init_indexed")?,
            phrase_entropy: module.load_function("phrase_entropy")?,
            bip39_init: module.load_function("bip39_init")?,
            bip39_slip10: module.load_function("bip39_slip10")?,
            wallet_addresses: module.load_function("wallet_addresses")?,
            filter_addresses: module.load_function("filter_addresses")?,
            base_table: stream.clone_htod(&base_table)?,
            dictionary: stream.clone_htod(&dictionary)?,
            workspace: RefCell::new(None),
            _module: module,
        })
    }

    fn workspace(&self, n: usize) -> Result<RefMut<'_, Workspace>> {
        ensure!(
            (1..=8_388_608).contains(&n),
            "invalid resident batch length"
        );
        let mut slot = self.workspace.borrow_mut();
        if slot.as_ref().is_none_or(|w| w.raw_capacity < n) {
            if let Some(old) = slot.as_mut() {
                old.clear()?;
            }
            *slot = Some(Workspace::new(self.stream.clone(), n)?);
        }
        Ok(RefMut::map(slot, |s| s.as_mut().unwrap()))
    }

    /// Retain entropy + compact passing indexes in VRAM through both KDF stages.
    /// The CPU receives only the passing indexes and corresponding final seeds.
    /// `keys=None` constructs legacy TON phrase entropy on GPU from word indexes.
    pub fn resident(
        &self,
        words: &[[u16; 24]],
        keys: Option<&[[u8; 64]]>,
    ) -> Result<ResidentBatch> {
        self.resident_impl(words, keys, false, false, None)
    }

    pub fn bip39(&self, words: &[[u16; 24]]) -> Result<ResidentBatch> {
        ensure!(
            words.len() <= 262144,
            "12-word batches are limited to 262144 candidates to bound KDF state and address latency"
        );
        self.resident_impl(words, None, true, false, None)
    }

    pub fn process(
        &self,
        words: &[[u16; 24]],
        keys: Option<&[[u8; 64]]>,
        bip39: bool,
        gpu_addresses: bool,
        suffix: Option<&crate::ton::Suffix>,
    ) -> Result<ResidentBatch> {
        ensure!(
            !bip39 || words.len() <= 262144,
            "12-word batch limit exceeded"
        );
        self.resident_impl(words, keys, bip39, gpu_addresses, suffix)
    }

    fn resident_impl(
        &self,
        words: &[[u16; 24]],
        keys: Option<&[[u8; 64]]>,
        bip39: bool,
        gpu_addresses: bool,
        suffix: Option<&crate::ton::Suffix>,
    ) -> Result<ResidentBatch> {
        ensure!(!words.is_empty(), "empty resident batch");
        ensure!(
            words.iter().flatten().all(|&i| i < 2048),
            "word index out of range"
        );
        if let Some(keys) = keys {
            ensure!(keys.len() == words.len(), "word/key batch mismatch");
        }
        let mut arena = self.workspace(words.len())?;
        let work = &mut *arena;
        let result = (|| {
            let start = Instant::now();
            let n = words.len() as u32;
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            if let Some(keys) = keys {
                self.stream
                    .memcpy_htod(keys.as_flattened(), &mut work.entropy)?;
            } else {
                self.stream
                    .memcpy_htod(words.as_flattened(), &mut work.words)?;
                if !bip39 {
                    // SAFETY: Every input index is checked above. words has 24*n
                    // elements, dictionary has 2048*9 bytes, entropy has 64*n bytes.
                    unsafe {
                        self.stream
                            .launch_builder(&self.phrase_entropy)
                            .arg(&work.words)
                            .arg(&self.dictionary)
                            .arg(&mut work.entropy)
                            .arg(&n)
                            .launch(cfg)?;
                    }
                }
            }
            self.stream.synchronize()?;
            let generation_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            // Count is explicitly reset; arena reuse must never append into a
            // previous batch's index list. Validation writes at most one per input.
            let passing = if bip39 {
                words.len()
            } else {
                self.stream.memset_zeros(&mut work.count)?;
                // The validation salt's padded message block is compiled into CUDA.
                unsafe {
                    self.stream
                        .launch_builder(&self.validate)
                        .arg(&work.entropy)
                        .arg(&mut work.indexes)
                        .arg(&mut work.count)
                        .arg(&n)
                        .launch(cfg)?;
                }
                self.stream.clone_dtoh(&work.count)?[0] as usize
            };
            ensure!(passing <= words.len(), "invalid resident validation count");
            work.reserve_valid(passing)?;
            let validation_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let mut seeds = Zeroizing::new(Vec::new());
            let mut indexes = Vec::new();
            let mut address_seconds = 0.0;
            let mut address_checksum = 0;
            let mut kdf_seconds = None;
            if passing > 0 {
                let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                let kdf_start = self.stream.record_event(timing)?;
                let valid = passing as u32;
                let cfg = LaunchConfig {
                    grid_dim: (valid.div_ceil(128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                // No host gather: init_indexed reads entropy[indexes[i]] on GPU.
                // The SoA PBKDF2 state stride is the current passing count.
                unsafe {
                    if bip39 {
                        self.stream
                            .launch_builder(&self.bip39_init)
                            .arg(&work.words)
                            .arg(&self.dictionary)
                            .arg(&mut work.state)
                            .arg(&valid)
                            .launch(cfg)?;
                    } else {
                        self.stream
                            .launch_builder(&self.init_indexed)
                            .arg(&work.entropy)
                            .arg(&work.indexes)
                            .arg(&mut work.state)
                            .arg(&valid)
                            .launch(cfg)?;
                    }
                }
                let mut remaining = if bip39 { 2047u32 } else { 99_999u32 };
                while remaining > 0 {
                    let rounds = remaining.min(512);
                    unsafe {
                        self.stream
                            .launch_builder(&self.step)
                            .arg(&mut work.state)
                            .arg(&valid)
                            .arg(&rounds)
                            .launch(cfg)?;
                    }
                    remaining -= rounds;
                }
                unsafe {
                    self.stream
                        .launch_builder(if bip39 {
                            &self.bip39_slip10
                        } else {
                            &self.finish
                        })
                        .arg(&work.state)
                        .arg(&mut work.output)
                        .arg(&valid)
                        .launch(cfg)?;
                }
                if gpu_addresses {
                    // Kernel launches are asynchronous: timing host submission
                    // would incorrectly charge the KDF to the address stage.
                    let kdf_end = self.stream.record_event(timing)?;
                    self.stream.memcpy_htod(&[u32::MAX], &mut work.winner)?;
                    self.stream.memset_zeros(&mut work.checksum)?;
                    // Fixed-base Ed25519 -> W5 cell hashes -> CRC -> binary suffix
                    // all remain on GPU. No private seeds leave VRAM for non-winners.
                    unsafe {
                        self.stream
                            .launch_builder(&self.wallet_addresses)
                            .arg(&work.output)
                            .arg(&self.base_table)
                            .arg(&mut work.addresses)
                            .arg(&valid)
                            .launch(cfg)?;
                        let mask = suffix.map_or(0, |s| s.mask);
                        let value = suffix.map_or(0, |s| s.value);
                        let bits = suffix.map_or(0, |s| s.bits);
                        self.stream
                            .launch_builder(&self.filter_addresses)
                            .arg(&work.addresses)
                            .arg(&mut work.winner)
                            .arg(&mut work.checksum)
                            .arg(&mask)
                            .arg(&value)
                            .arg(&bits)
                            .arg(&valid)
                            .launch(cfg)?;
                    }
                    let address_end = self.stream.record_event(timing)?;
                    address_checksum = self.stream.clone_dtoh(&work.checksum)?[0];
                    let selected = self.stream.clone_dtoh(&work.winner)?[0] as usize;
                    if selected < passing {
                        let key =
                            Zeroizing::new(self.stream.clone_dtoh(
                                &work.output.slice(selected * 64..selected * 64 + 64),
                            )?);
                        seeds.push(key.as_slice().try_into().unwrap());
                        indexes.push(if bip39 {
                            selected
                        } else {
                            self.stream
                                .clone_dtoh(&work.indexes.slice(selected..selected + 1))?[0]
                                as usize
                        });
                    }
                    address_end.synchronize()?;
                    kdf_seconds = Some(kdf_start.elapsed_ms(&kdf_end)? as f64 / 1000.0);
                    address_seconds = kdf_end.elapsed_ms(&address_end)? as f64 / 1000.0;
                } else {
                    let bytes =
                        Zeroizing::new(self.stream.clone_dtoh(&work.output.slice(..passing * 64))?);
                    *seeds = bytes
                        .chunks_exact(64)
                        .map(|c| c.try_into().unwrap())
                        .collect();
                    indexes = if bip39 {
                        (0..passing).collect()
                    } else {
                        self.stream
                            .clone_dtoh(&work.indexes.slice(..passing))?
                            .into_iter()
                            .map(|i| i as usize)
                            .collect()
                    };
                    // Atomic compaction order is intentionally preserved for BOTH
                    // arrays. Sorting indexes alone would associate words with wrong keys.
                    ensure!(
                        indexes.iter().all(|&i| i < words.len()),
                        "invalid resident index"
                    );
                }
            }
            Ok(ResidentBatch {
                tested_wallets: passing,
                address_seconds,
                address_checksum,
                indexes,
                seeds,
                generation_seconds,
                validation_seconds,
                derivation_seconds: kdf_seconds.unwrap_or_else(|| start.elapsed().as_secs_f64()),
                cleanup_seconds: 0.0,
            })
        })();
        // Cleanup executes on success AND error. Keep allocations for the next
        // batch, but never keep an old batch's secret contents as a performance cache.
        let start = Instant::now();
        let cleanup = work.clear();
        let mut result = result?;
        cleanup?;
        result.cleanup_seconds = start.elapsed().as_secs_f64();
        Ok(result)
    }

    pub fn derive(&self, keys: &[[u8; 64]], salt: &[u8], rounds: u32) -> Result<Vec<[u8; 64]>> {
        ensure!(rounds > 0 && salt.len() <= 100, "invalid PBKDF2 parameters");
        ensure!(keys.len() <= u32::MAX as usize, "batch too large");
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let n = keys.len() as u32;
        let mut input = self.stream.clone_htod(keys.as_flattened())?;
        let salt_dev = self.stream.clone_htod(salt)?;
        let mut state = self.stream.alloc_zeros::<u64>(keys.len() * 32)?;
        let mut output = self.stream.alloc_zeros::<u8>(keys.len() * 64)?;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: All slices have lengths matching the kernel's n and saltlen parameters;
        // cudarc tracks allocations and stream dependencies through the launch arguments.
        unsafe {
            self.stream
                .launch_builder(&self.init)
                .arg(&input)
                .arg(&salt_dev)
                .arg(&(salt.len() as i32))
                .arg(&mut state)
                .arg(&n)
                .launch(cfg)?;
        }
        let mut remaining = rounds - 1;
        while remaining > 0 {
            // Bounded launches avoid Windows display-driver watchdog resets.
            let count = remaining.min(512);
            unsafe {
                self.stream
                    .launch_builder(&self.step)
                    .arg(&mut state)
                    .arg(&n)
                    .arg(&count)
                    .launch(cfg)?;
            }
            remaining -= count;
        }
        unsafe {
            self.stream
                .launch_builder(&self.finish)
                .arg(&state)
                .arg(&mut output)
                .arg(&n)
                .launch(cfg)?;
        }
        let bytes = Zeroizing::new(self.stream.clone_dtoh(&output)?);
        self.stream.memset_zeros(&mut input)?;
        self.stream.memset_zeros(&mut state)?;
        self.stream.memset_zeros(&mut output)?;
        self.stream.synchronize()?;
        Ok(bytes
            .chunks_exact(64)
            .map(|c| c.try_into().unwrap())
            .collect())
    }

    pub fn validate(&self, keys: &[[u8; 64]]) -> Result<Vec<usize>> {
        ensure!(
            keys.len() <= 8_388_608,
            "validation batch exceeds tested watchdog-safe limit"
        );
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let n = keys.len() as u32;
        let mut input = self.stream.clone_htod(keys.as_flattened())?;
        let mut indexes = self.stream.alloc_zeros::<u32>(keys.len())?;
        let mut count = self.stream.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: At most n threads can append one index each; count starts at zero.
        // Input contains n*64 bytes; the validation salt is compiled into CUDA.
        unsafe {
            self.stream
                .launch_builder(&self.validate)
                .arg(&input)
                .arg(&mut indexes)
                .arg(&mut count)
                .arg(&n)
                .launch(cfg)?;
        }
        let count = self.stream.clone_dtoh(&count)?[0] as usize;
        ensure!(count <= keys.len(), "invalid GPU validation count");
        let mut result: Vec<usize> = if count == 0 {
            Vec::new()
        } else {
            self.stream
                .clone_dtoh(&indexes.slice(..count))?
                .into_iter()
                .map(|i| i as usize)
                .collect()
        };
        self.stream.memset_zeros(&mut input)?;
        self.stream.memset_zeros(&mut indexes)?;
        self.stream.synchronize()?;
        // Atomic append order is unspecified. Restore source order for deterministic
        // candidate/seed correspondence and reproducible CPU cross-checks.
        result.sort_unstable();
        ensure!(
            result.iter().all(|&i| i < keys.len()),
            "GPU validation index out of bounds"
        );
        Ok(result)
    }

    pub fn self_test(&self) -> Result<()> {
        use crate::ton;
        // Compare both salts and boundary iteration counts; non-warp-aligned batches
        // also exercise bounds checks and the structure-of-arrays stride.
        let keys: Vec<[u8; 64]> = (0..33)
            .map(|i| ton::entropy(&format!("public CUDA test vector {i}")))
            .collect();
        for (salt, rounds) in [
            (ton::VALIDATION_SALT, 1),
            (ton::VALIDATION_SALT, 390),
            (ton::WALLET_SALT, 513),
            (ton::WALLET_SALT, 100_000),
        ] {
            let actual = self.derive(&keys, salt, rounds)?;
            use rayon::prelude::*;
            let expected: Vec<_> = keys
                .par_iter()
                .map(|k| ton::derive(k, salt, rounds))
                .collect();
            ensure!(
                actual == expected,
                "GPU/CPU PBKDF2 mismatch at {rounds} rounds"
            );
        }
        let mut keys: Vec<[u8; 64]> = (0..1025)
            .map(|i| ton::entropy(&format!("public validation test {i}")))
            .collect();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/ton-fixture.json"))?;
        keys.push(ton::entropy(fixture["mnemonic"].as_str().unwrap()));
        use rayon::prelude::*;
        let expected: Vec<usize> = keys
            .par_iter()
            .enumerate()
            .filter_map(|(i, k)| ton::valid(k).then_some(i))
            .collect();
        ensure!(
            !expected.is_empty(),
            "validation self-test must exercise passing inputs"
        );
        ensure!(
            self.validate(&keys)? == expected,
            "fused GPU mnemonic validation disagrees with CPU"
        );
        self.resident_self_test()?;
        self.bip39_self_test()?;
        self.address_self_test()?;
        Ok(())
    }

    fn address_self_test(&self) -> Result<()> {
        use crate::ton;
        let seeds: Vec<[u8; 64]> = (0..513)
            .map(|i| ton::entropy(&format!("public CUDA address vector {i}")))
            .collect();
        let mut arena = self.workspace(seeds.len())?;
        let work = &mut *arena;
        work.reserve_valid(seeds.len())?;
        self.stream
            .memcpy_htod(seeds.as_flattened(), &mut work.output)?;
        let n = seeds.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.wallet_addresses)
                .arg(&work.output)
                .arg(&self.base_table)
                .arg(&mut work.addresses)
                .arg(&n)
                .launch(cfg)
                .context("CUDA address self-test launch")?;
        }
        let raw = self
            .stream
            .clone_dtoh(&work.addresses.slice(..seeds.len() * 36))?;
        let expected: Vec<_> = seeds.iter().map(ton::friendly).collect();
        for (i, (a, b)) in raw.chunks_exact(36).zip(expected.iter()).enumerate() {
            ensure!(a == b, "CUDA Ed25519/W5/CRC mismatch at public vector {i}");
        }
        let encoded = ton::encode_address(&expected[0]);
        let checksum = expected.iter().fold(0u32, |c, a| {
            c ^ u32::from_be_bytes(a[32..].try_into().unwrap())
        });
        for len in 1..=8 {
            let suffix = ton::Suffix::new(&encoded[48 - len..]);
            self.stream.memcpy_htod(&[u32::MAX], &mut work.winner)?;
            self.stream.memset_zeros(&mut work.checksum)?;
            unsafe {
                self.stream
                    .launch_builder(&self.filter_addresses)
                    .arg(&work.addresses)
                    .arg(&mut work.winner)
                    .arg(&mut work.checksum)
                    .arg(&suffix.mask)
                    .arg(&suffix.value)
                    .arg(&suffix.bits)
                    .arg(&n)
                    .launch(cfg)?;
            }
            ensure!(
                self.stream.clone_dtoh(&work.winner)?[0] == 0
                    && self.stream.clone_dtoh(&work.checksum)?[0] == checksum,
                "GPU binary suffix/checksum reduction mismatch"
            );
        }
        work.clear()?;
        drop(arena);
        // Test the complete resident KDF -> GPU address -> CPU winner handoff,
        // including matching private-key index and both recovery schemes.
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/ton-fixture.json"))?;
        let list = bip39::Language::English.word_list();
        let mut words = [0u16; 24];
        for (i, w) in words
            .iter_mut()
            .zip(fixture["mnemonic"].as_str().unwrap().split_whitespace())
        {
            *i = list.iter().position(|&s| s == w).unwrap() as u16;
        }
        let suffix = ton::Suffix::new(&fixture["address"].as_str().unwrap()[45..]);
        let actual = self.process(&[words], None, false, true, Some(&suffix))?;
        ensure!(
            actual.tested_wallets == 1
                && actual.seeds.len() == 1
                && ton::address(&actual.seeds[0]) == fixture["address"],
            "legacy GPU winner handoff mismatch"
        );
        let bfixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/bip39-fixture.json"))?;
        let case = &bfixture["cases"][0];
        let entropy: [u8; 16] = hex::decode(case["entropy"].as_str().unwrap())?
            .try_into()
            .unwrap();
        let suffix = ton::Suffix::new(&case["address"].as_str().unwrap()[45..]);
        let actual = self.process(
            &[crate::bip39_ton::indexes(&entropy)],
            None,
            true,
            true,
            Some(&suffix),
        )?;
        ensure!(
            actual.tested_wallets == 1
                && actual.seeds.len() == 1
                && ton::address(&actual.seeds[0]) == case["address"],
            "BIP39 GPU winner handoff mismatch"
        );
        Ok(())
    }

    fn bip39_self_test(&self) -> Result<()> {
        use crate::{bip39_ton, ton};
        use rayon::prelude::*;
        let words: Vec<_> = (0..33u8).map(|i| bip39_ton::indexes(&[i; 16])).collect();
        let expected: Vec<_> = words
            .par_iter()
            .map(|w| bip39_ton::private_seed(&bip39_ton::phrase(w)).unwrap())
            .collect();
        let actual = self.bip39(&words)?;
        ensure!(
            actual.indexes == (0..words.len()).collect::<Vec<_>>() && *actual.seeds == expected,
            "BIP39 CUDA KDF/SLIP-0010 mismatch"
        );
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/bip39-fixture.json"))?;
        for case in fixture["cases"].as_array().unwrap() {
            let entropy: [u8; 16] = hex::decode(case["entropy"].as_str().unwrap())?
                .try_into()
                .unwrap();
            let actual = self.bip39(&[bip39_ton::indexes(&entropy)])?;
            ensure!(
                hex::encode(&actual.seeds[0][..32]) == case["private_seed"]
                    && ton::address(&actual.seeds[0]) == case["address"],
                "BIP39 CUDA differs from official SDK fixture"
            );
        }
        Ok(())
    }

    fn resident_self_test(&self) -> Result<()> {
        use crate::ton;
        use rayon::prelude::*;
        // Every dictionary entry is exercised. Short and long phrases cross both
        // HMAC's 128-byte key threshold and SHA-512's 112-byte padding threshold.
        let mut words: Vec<[u16; 24]> = (0..2048u16).map(|i| [i; 24]).collect();
        let list = bip39::Language::English.word_list();
        let by_length: Vec<u16> = (3..=8)
            .map(|len| list.iter().position(|w| w.len() == len).unwrap() as u16)
            .collect();
        for len in [95, 111, 112, 127, 128, 129, 143, 215] {
            let mut row = [by_length[0]; 24];
            let mut extra = len - 95;
            for i in &mut row {
                let add = extra.min(5);
                *i = by_length[add];
                extra -= add;
            }
            ensure!(
                ton::phrase(&row).len() == len,
                "invalid SHA boundary fixture"
            );
            words.push(row);
        }
        {
            let mut arena = self.workspace(words.len())?;
            let work = &mut *arena;
            self.stream
                .memcpy_htod(words.as_flattened(), &mut work.words)?;
            let n = words.len() as u32;
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.phrase_entropy)
                    .arg(&work.words)
                    .arg(&self.dictionary)
                    .arg(&mut work.entropy)
                    .arg(&n)
                    .launch(cfg)?;
            }
            let entropy = Zeroizing::new(
                self.stream
                    .clone_dtoh(&work.entropy.slice(..words.len() * 64))?,
            );
            let expected: Vec<[u8; 64]> = words.par_iter().map(ton::entropy_from_indexes).collect();
            ensure!(
                entropy.as_slice() == expected.as_flattened(),
                "GPU phrase entropy disagrees with CPU"
            );
            work.clear()?;
        }
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/ton-fixture.json"))?;
        let mut row = [0u16; 24];
        for (i, w) in row
            .iter_mut()
            .zip(fixture["mnemonic"].as_str().unwrap().split_whitespace())
        {
            *i = list.iter().position(|&word| word == w).unwrap() as u16;
        }
        let key = ton::entropy_from_indexes(&row);
        let expected_seed = ton::derive(&key, ton::WALLET_SALT, 100_000);
        // All-pass 257 exceeds the initial expected-pass reservation. Subsequent
        // tiny batches reuse the larger arena with a different SoA stride.
        for n in [257, 1, 33] {
            let words = vec![row; n];
            let keys = vec![key; n];
            for host_entropy in [true, false] {
                let actual = self.resident(&words, host_entropy.then_some(keys.as_slice()))?;
                ensure!(
                    actual.indexes.len() == n && actual.seeds.len() == n,
                    "resident lost valid inputs"
                );
                let mut seen = actual.indexes.clone();
                seen.sort_unstable();
                ensure!(
                    seen == (0..n).collect::<Vec<_>>(),
                    "resident duplicated or lost indexes"
                );
                ensure!(
                    actual.seeds.iter().all(|s| *s == expected_seed),
                    "resident derivation mismatch"
                );
            }
        }
        // Mixed validity and distinct keys catch index/key association errors that
        // identical-key fixtures cannot detect after nondeterministic compaction.
        let words: Vec<_> = (0..513u16).map(|i| [i; 24]).collect();
        let keys: Vec<_> = words.par_iter().map(ton::entropy_from_indexes).collect();
        let expected: Vec<_> = keys
            .par_iter()
            .enumerate()
            .filter_map(|(i, k)| ton::valid(k).then_some(i))
            .collect();
        let actual = self.resident(&words, None)?;
        let mut indexes = actual.indexes.clone();
        indexes.sort_unstable();
        ensure!(
            indexes == expected,
            "resident validation disagrees with CPU"
        );
        for (&index, seed) in actual.indexes.iter().zip(actual.seeds.iter()) {
            ensure!(
                *seed == ton::derive(&keys[index], ton::WALLET_SALT, 100_000),
                "resident index/seed association mismatch"
            );
        }
        let no_pass: Vec<_> = (0..2048u16)
            .map(|i| [i; 24])
            .filter(|w| !ton::valid(&ton::entropy_from_indexes(w)))
            .take(33)
            .collect();
        let empty = self.resident(&no_pass, None)?;
        ensure!(
            empty.indexes.is_empty() && empty.seeds.is_empty(),
            "zero-pass batch retained stale results"
        );
        Ok(())
    }
}
