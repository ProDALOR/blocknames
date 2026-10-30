mod bip39_ton;
mod console;
mod gpu;
mod prefetch;
mod runtime;
mod ton;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rand::{RngCore, rngs::OsRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use zeroize::{Zeroize, Zeroizing};

#[derive(Parser)]
#[command(
    version,
    about = "Local TON W5 vanity wallets with legacy TON or 12-word BIP39 recovery"
)]
struct Cli {
    /// Final output format. JSON writes one machine-readable object to stdout.
    #[arg(long,global=true,value_enum,default_value_t=OutputFormat::Table)]
    format: OutputFormat,
    /// Suppress progress and startup messages (errors and final results remain).
    #[arg(long, global = true)]
    quiet: bool,
    /// Seconds between progress-table rows.
    #[arg(long,global=true,default_value_t=1,value_parser=clap::value_parser!(u64).range(1..=3600))]
    progress_interval: u64,
    /// CPU worker count. Defaults to the number of available logical processors.
    #[arg(long,global=true,value_parser=clap::value_parser!(u16).range(1..=1024))]
    threads: Option<u16>,
    /// CUDA toolkit root or directory containing NVRTC and its matching builtins library.
    #[arg(long, global = true)]
    cuda_path: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Table,
    Json,
}

struct DisplayOptions {
    started: Instant,
    format: OutputFormat,
    quiet: bool,
    progress_interval: u64,
}

#[derive(Subcommand)]
enum Command {
    /// Search fresh, cryptographically random TON phrases for a UQ address suffix.
    Search {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long, default_value = "_ai", value_parser = parse_suffix)]
        suffix: String,
        /// Stop at a batch boundary after this duration; zero means unlimited.
        #[arg(long, default_value_t = 0)]
        max_seconds: u64,
        /// Plaintext secret JSON; default is under LOCALAPPDATA, outside this repo.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Public performance data only; never includes recovery words.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Measure the complete pipeline (randomness, mnemonic validation, KDF, addresses).
    Bench {
        #[command(flatten)]
        engine: EngineArgs,
        /// Minimum timed duration; finishes the current batch.
        #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
        seconds: u64,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Compare GPU results with RustCrypto, including full 100,000-round derivation.
    SelfTest {
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Re-derive a saved wallet from its words, without displaying those words.
    Verify {
        #[arg(long)]
        wallet: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Pipeline {
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum GpuMemory {
    Resident,
    Transient,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PhraseEngine {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
enum MnemonicMode {
    #[serde(rename = "ton24")]
    Ton24,
    #[serde(rename = "bip39-12")]
    #[value(name = "bip39-12")]
    Bip3912,
}
impl MnemonicMode {
    fn name(self) -> &'static str {
        match self {
            Self::Ton24 => "ton24",
            Self::Bip3912 => "bip39-12",
        }
    }
    fn phrase(self, words: &[u16; 24]) -> String {
        match self {
            Self::Ton24 => ton::phrase(words),
            Self::Bip3912 => bip39_ton::phrase(words),
        }
    }
}

#[derive(Args)]
struct EngineArgs {
    /// CPU or CUDA Ed25519/W5/suffix stage. Auto uses CUDA with resident GPU memory.
    #[arg(long,value_enum,default_value_t=PhraseEngine::Auto)]
    address_engine: PhraseEngine,
    /// 12-word BIP39 uses Keeper's standard m/44'/607'/0' path and empty passphrase.
    #[arg(long,value_enum,default_value_t=MnemonicMode::Ton24)]
    mode: MnemonicMode,
    /// Reuse VRAM and retain validation output through derivation.
    #[arg(long,value_enum,default_value_t=GpuMemory::Resident)]
    gpu_memory: GpuMemory,
    /// Construct TON phrase entropy on GPU or CPU. Auto selects CUDA with resident VRAM.
    #[arg(long,value_enum,default_value_t=PhraseEngine::Auto)]
    phrase_engine: PhraseEngine,
    /// Prepare the next phrase batch during GPU work. Auto enables this for CUDA.
    #[arg(long, value_enum, default_value_t = Pipeline::Auto)]
    pipeline: Pipeline,
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Raw phrases per batch. CUDA: TON24 4194304, BIP39-12 131072. CPU: 4096/512.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=8_388_608))]
    batch_size: Option<u32>,
}

struct PreparedBatch {
    // OS randomness stays on the CPU. Exact indexes are retained for recovery of
    // a winner; random generation is never replaced with counters or weak RNGs.
    words: Zeroizing<Vec<[u16; 24]>>,
    keys: Option<Zeroizing<Vec<[u8; 64]>>>,
    generation_seconds: f64,
}

#[derive(Serialize, Deserialize)]
struct Wallet {
    #[serde(default = "legacy_scheme")]
    mnemonic_scheme: String,
    #[serde(default)]
    derivation_path: Option<String>,
    address: String,
    mnemonic: String,
    wallet_version: String,
    wallet_id: u32,
    network: String,
    format: String,
}
fn legacy_scheme() -> String {
    "ton24".into()
}
impl Drop for Wallet {
    fn drop(&mut self) {
        self.mnemonic.zeroize();
    }
}

#[derive(Default, Serialize)]
struct Stats {
    address_checksum: u32,
    ambiguous_winners_rejected: u64,
    gpu_generation_seconds: f64,
    gpu_cleanup_seconds: f64,
    raw_phrases: u64,
    valid_wallets: u64,
    generation_seconds: f64,
    generation_wait_seconds: f64,
    producer_shutdown_seconds: f64,
    discarded_prefetch_batches: usize,
    validation_seconds: f64,
    derivation_seconds: f64,
    address_seconds: f64,
}

#[derive(Serialize)]
struct Report<'a> {
    cpu_threads: usize,
    requested_suffix: Option<&'a str>,
    expected_match_seconds: Option<f64>,
    result: Option<SearchResult>,
    setup_seconds: f64,
    total_wall_seconds: f64,
    kernel_cache_hit: Option<bool>,
    address_engine: &'a str,
    mnemonic_scheme: &'a str,
    derivation_path: Option<&'a str>,
    backend: Backend,
    device: &'a str,
    batch_size: usize,
    pipeline: bool,
    gpu_memory: GpuMemory,
    phrase_engine: &'a str,
    vram_workspace_bytes: Option<usize>,
    elapsed_seconds: f64,
    wallets_per_second: f64,
    expected_ai_seconds: f64,
    expected_underscore_ai_seconds: f64,
    gpu_cpu_self_test: bool,
    stats: &'a Stats,
}

#[derive(Serialize)]
struct SearchResult {
    address: String,
    wallet_file: PathBuf,
}

fn parse_suffix(s: &str) -> Result<String, String> {
    if s.is_empty()
        || s.len() > 8
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err("use 1-8 case-sensitive Base64url characters (A-Z, a-z, 0-9, _, -)".into());
    }
    Ok(s.to_owned())
}

fn generate(n: usize) -> Zeroizing<Vec<[u16; 24]>> {
    // Masking is unbiased because 2048 divides 2^16. Each word gets fresh OS
    // randomness; no phrase prefix is fixed to make the vanity search cheaper.
    let mut random = Zeroizing::new(vec![0u8; n * 48]);
    OsRng.fill_bytes(&mut random);
    Zeroizing::new(
        random
            .par_chunks_exact(48)
            .map(|bytes| {
                let mut indexes = [0u16; 24];
                for (i, b) in bytes.as_chunks::<2>().0.iter().enumerate() {
                    indexes[i] = u16::from_le_bytes(*b) & 2047;
                }
                indexes
            })
            .collect(),
    )
}

fn derive_batch(
    gpu: Option<&gpu::Gpu>,
    keys: &[[u8; 64]],
    salt: &[u8],
    rounds: u32,
) -> Result<Vec<[u8; 64]>> {
    match gpu {
        Some(g) => g.derive(keys, salt, rounds),
        None => Ok(keys
            .par_iter()
            .map(|k| ton::derive(k, salt, rounds))
            .collect()),
    }
}

fn prepare(n: usize, gpu_phrases: bool, mode: MnemonicMode) -> PreparedBatch {
    let start = Instant::now();
    let words = if mode == MnemonicMode::Ton24 {
        generate(n)
    } else {
        let mut entropy = Zeroizing::new(vec![0u8; n * 16]);
        OsRng.fill_bytes(&mut entropy);
        Zeroizing::new(
            entropy
                .par_chunks_exact(16)
                .map(|c| bip39_ton::indexes(c.try_into().unwrap()))
                .collect(),
        )
    };
    // GPU phrase mode needs only the small word-index representation on the host.
    // CPU mode is retained as a fully independent entropy/reference implementation.
    let keys = if gpu_phrases || mode == MnemonicMode::Bip3912 {
        None
    } else {
        Some(Zeroizing::new(
            words
                .par_iter()
                .map(|w| {
                    if mode == MnemonicMode::Ton24 {
                        ton::entropy_from_indexes(w)
                    } else {
                        bip39_ton::private_seed(&Zeroizing::new(bip39_ton::phrase(w))).unwrap()
                    }
                })
                .collect::<Vec<_>>(),
        ))
    };
    PreparedBatch {
        words,
        keys,
        generation_seconds: start.elapsed().as_secs_f64(),
    }
}

fn batch(
    gpu: Option<&gpu::Gpu>,
    prepared: PreparedBatch,
    stats: &mut Stats,
    suffix: Option<&str>,
    resident: bool,
    mode: MnemonicMode,
    gpu_addresses: bool,
) -> Result<Option<Wallet>> {
    let PreparedBatch {
        words,
        keys,
        generation_seconds,
    } = prepared;
    let n = words.len();
    stats.generation_seconds += generation_seconds;
    let pattern = suffix.map(ton::Suffix::new);
    let (indexes, seeds, tested) = if let Some(g) = gpu.filter(|_| resident) {
        let result = g.process(
            &words,
            keys.as_ref().map(|k| k.as_slice()),
            mode == MnemonicMode::Bip3912,
            gpu_addresses,
            pattern.as_ref(),
        )?;
        stats.gpu_generation_seconds += result.generation_seconds;
        stats.validation_seconds += result.validation_seconds;
        stats.derivation_seconds += result.derivation_seconds;
        stats.gpu_cleanup_seconds += result.cleanup_seconds;
        stats.address_seconds += result.address_seconds;
        stats.address_checksum ^= result.address_checksum;
        (result.indexes, result.seeds, result.tested_wallets)
    } else if mode == MnemonicMode::Bip3912 {
        let start = Instant::now();
        let keys = Zeroizing::new(
            words
                .par_iter()
                .map(|w| bip39_ton::private_seed(&Zeroizing::new(bip39_ton::phrase(w))).unwrap())
                .collect(),
        );
        stats.derivation_seconds += start.elapsed().as_secs_f64();
        ((0..n).collect(), keys, n)
    } else {
        let keys = keys
            .as_ref()
            .context("CPU phrase entropy is required in transient mode")?;
        let start = Instant::now();
        let indexes: Vec<usize> = match gpu {
            Some(g) => g.validate(keys)?,
            None => keys
                .par_iter()
                .enumerate()
                .filter_map(|(i, k)| ton::valid(k).then_some(i))
                .collect(),
        };
        let valid_keys = Zeroizing::new(indexes.iter().map(|&i| keys[i]).collect::<Vec<_>>());
        stats.validation_seconds += start.elapsed().as_secs_f64();
        let start = Instant::now();
        let seeds = Zeroizing::new(derive_batch(gpu, &valid_keys, ton::WALLET_SALT, 100_000)?);
        stats.derivation_seconds += start.elapsed().as_secs_f64();
        let tested = seeds.len();
        (indexes, seeds, tested)
    };
    let start = Instant::now();
    // No per-address String allocation or Base64 encoding in the hot loop.
    // A public XOR accumulator forces complete address work in benchmark mode.
    let (checksum, winner) = seeds
        .par_iter()
        .enumerate()
        .map(|(j, seed)| {
            let raw = ton::friendly(seed);
            let checksum = u32::from_be_bytes(raw[32..36].try_into().unwrap());
            (
                checksum,
                pattern
                    .as_ref()
                    .filter(|p| p.matches(&raw))
                    .map(|_| (j, raw)),
            )
        })
        .reduce(
            || (0, None),
            |(a, x), (b, y)| {
                (
                    a ^ b,
                    match (x, y) {
                        (Some(x), Some(y)) => Some(if x.0 < y.0 { x } else { y }),
                        (x, y) => x.or(y),
                    },
                )
            },
        );
    if !gpu_addresses {
        stats.address_checksum ^= checksum;
    }
    stats.address_seconds += start.elapsed().as_secs_f64();
    stats.raw_phrases += n as u64;
    stats.valid_wallets += tested as u64;
    if let Some((j, raw)) = winner {
        let address = ton::encode_address(&raw);
        let candidate = &words[indexes[j]];
        // Recompute from the saved WORD indexes, rather than trusting a cached
        // GPU entropy buffer. This checks the complete phrase-to-address mapping.
        let phrase = Zeroizing::new(mode.phrase(candidate));
        let entropy = Zeroizing::new(ton::entropy(&phrase));
        let expected = Zeroizing::new(if mode == MnemonicMode::Ton24 {
            ensure!(
                ton::valid(&entropy),
                "winner failed CPU mnemonic validation"
            );
            ton::derive(&entropy, ton::WALLET_SALT, 100_000)
        } else {
            // Keeper's fallback resolver checks legacy TON before BIP39. Reject
            // the rare dual-valid phrase, rather than relying on UI scheme selection.
            if ton::valid(&entropy) {
                stats.ambiguous_winners_rejected += 1;
                return Ok(None);
            }
            bip39_ton::private_seed(&phrase)?
        });
        ensure!(*expected == seeds[j], "winning GPU seed disagrees with CPU");
        ensure!(
            ton::address(&expected) == address,
            "winning address disagrees with CPU"
        );
        return Ok(Some(Wallet {
            mnemonic_scheme: mode.name().into(),
            derivation_path: (mode == MnemonicMode::Bip3912).then(|| bip39_ton::PATH.into()),
            address,
            mnemonic: mode.phrase(candidate),
            wallet_version: "v5r1".into(),
            wallet_id: ton::WALLET_ID,
            network: "mainnet".into(),
            format: "non-bounceable base64url".into(),
        }));
    }
    Ok(None)
}

fn save_wallet(wallet: &Wallet, output: Option<PathBuf>) -> Result<PathBuf> {
    let path = match output {
        Some(p) => p,
        None => {
            let dir = runtime::data_dir()?.join("wallets");
            fs::create_dir_all(&dir)?;
            dir.join(format!(
                "{}.secret.json",
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
            ))
        }
    };
    let json = Zeroizing::new(serde_json::to_vec_pretty(wallet)?);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .with_context(|| format!("creating secret file {}", path.display()))?;
    file.write_all(&json)?;
    file.sync_all()?;
    Ok(path)
}

fn verify(path: &PathBuf, format: OutputFormat) -> Result<()> {
    let bytes = Zeroizing::new(fs::read(path)?);
    let w: Wallet = serde_json::from_slice(&bytes)?;
    ensure!(
        w.network == "mainnet"
            && w.wallet_version == "v5r1"
            && w.wallet_id == ton::WALLET_ID
            && w.format == "non-bounceable base64url",
        "unsupported wallet settings"
    );
    let words: Vec<_> = w.mnemonic.split_whitespace().collect();
    ensure!(
        words.len()
            == if w.mnemonic_scheme == "bip39-12" {
                12
            } else {
                24
            }
            && words
                .iter()
                .all(|w| bip39::Language::English.word_list().contains(w)),
        "invalid word list"
    );
    let e = Zeroizing::new(ton::entropy(&w.mnemonic));
    let seed = Zeroizing::new(if w.mnemonic_scheme == "ton24" {
        ensure!(
            w.derivation_path.is_none(),
            "legacy wallet must not specify BIP39 path"
        );
        ensure!(ton::valid(&e), "invalid TON phrase");
        ton::derive(&e, ton::WALLET_SALT, 100_000)
    } else {
        ensure!(
            w.mnemonic_scheme == "bip39-12"
                && w.derivation_path.as_deref() == Some(bip39_ton::PATH),
            "unsupported mnemonic scheme/path"
        );
        ensure!(!ton::valid(&e), "ambiguous legacy TON/BIP39 phrase");
        bip39_ton::private_seed(&w.mnemonic)?
    });
    ensure!(
        ton::address(&seed) == w.address,
        "words do not restore saved address"
    );
    match format {
        OutputFormat::Table => println!("Verified W5R1/mainnet: {}", w.address),
        OutputFormat::Json => println!(
            "{}",
            serde_json::json!({"verified":true,"address":w.address,"mnemonic_scheme":w.mnemonic_scheme})
        ),
    }
    Ok(())
}

fn run(
    engine: EngineArgs,
    seconds: u64,
    suffix: Option<String>,
    output: Option<PathBuf>,
    report_path: Option<PathBuf>,
    display: DisplayOptions,
) -> Result<()> {
    if let Some(path) = &report_path {
        ensure!(
            !path.exists(),
            "report already exists: {}; choose a new report filename",
            path.display()
        );
    }
    if let Some(path) = &output {
        ensure!(
            !path.exists(),
            "output already exists: {}; refusing to replace it",
            path.display()
        );
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            ensure!(
                parent.is_dir(),
                "output directory does not exist: {}",
                parent.display()
            );
        }
    }
    // Include runtime discovery and worker-pool setup performed by main, too.
    let setup_start = display.started;
    let gpu = match engine.backend {
        Backend::Cuda => Some(gpu::Gpu::new(engine.device)?),
        Backend::Cpu => None,
    };
    let resident = matches!(engine.gpu_memory, GpuMemory::Resident);
    let gpu_addresses = match engine.address_engine {
        PhraseEngine::Auto => gpu.is_some() && resident,
        PhraseEngine::Cpu => false,
        PhraseEngine::Cuda => {
            ensure!(
                gpu.is_some() && resident,
                "CUDA address stage requires resident CUDA memory"
            );
            true
        }
    };
    let gpu_phrases = match engine.phrase_engine {
        PhraseEngine::Auto => gpu.is_some() && resident,
        PhraseEngine::Cpu => false,
        PhraseEngine::Cuda => {
            ensure!(
                gpu.is_some() && resident,
                "CUDA phrase construction requires --backend cuda --gpu-memory resident"
            );
            true
        }
    };
    if gpu.is_some() && engine.mode == MnemonicMode::Bip3912 {
        ensure!(
            resident && gpu_phrases,
            "12-word CUDA mode requires resident memory and CUDA phrase engine; CPU backend is also supported"
        );
    }
    let device = gpu.as_ref().map_or("CPU / Rayon", |g| g.name.as_str());
    let n = engine.batch_size.unwrap_or(match engine.backend {
        Backend::Cpu => {
            if engine.mode == MnemonicMode::Bip3912 {
                512
            } else {
                4096
            }
        }
        Backend::Cuda => {
            if engine.mode == MnemonicMode::Bip3912 {
                131072
            } else {
                4194304
            }
        }
    }) as usize;
    ensure!(
        engine.mode != MnemonicMode::Bip3912 || n <= 262144,
        "12-word mode supports batches up to 262144"
    );
    let pipeline = match engine.pipeline {
        Pipeline::Auto => matches!(engine.backend, Backend::Cuda),
        Pipeline::On => true,
        Pipeline::Off => false,
    };
    if !display.quiet {
        eprintln!(
            "Backend: {:?}; device: {device}; raw batch: {n}; pipeline: {pipeline}",
            engine.backend
        );
    }
    if let Some(g) = &gpu {
        if !display.quiet {
            eprintln!("Checking CUDA against CPU...");
        }
        g.self_test()?;
    }
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = stop.clone();
    ctrlc::set_handler(move || stop_handler.store(true, Ordering::Relaxed))?;
    let start = Instant::now();
    let mut producer = if pipeline {
        Some(prefetch::Prefetch::new(move || {
            prepare(n, gpu_phrases, engine.mode)
        })?)
    } else {
        None
    };
    let mut stats = Stats::default();
    let mut found = false;
    let mut result = None;
    if !display.quiet {
        console::progress_header();
    }
    let mut last_progress = Instant::now();
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let waiting = Instant::now();
        let prepared = match &mut producer {
            Some(p) => p.take()?,
            None => prepare(n, gpu_phrases, engine.mode),
        };
        stats.generation_wait_seconds += waiting.elapsed().as_secs_f64();
        if let Some(w) = batch(
            gpu.as_ref(),
            prepared,
            &mut stats,
            suffix.as_deref(),
            resident,
            engine.mode,
            gpu_addresses,
        )? {
            let path = save_wallet(&w, output.clone())?;
            result = Some(SearchResult {
                address: w.address.clone(),
                wallet_file: path,
            });
            found = true;
        }
        let elapsed = start.elapsed().as_secs_f64();
        if !display.quiet
            && (last_progress.elapsed().as_secs_f64() >= display.progress_interval as f64
                || found
                || (seconds > 0 && elapsed >= seconds as f64))
        {
            console::progress(
                elapsed,
                &stats,
                gpu.as_ref().and_then(|g| g.workspace_bytes()),
            );
            last_progress = Instant::now();
        }
        if found || (seconds > 0 && elapsed >= seconds as f64) {
            break;
        }
    }
    if !display.quiet {
        console::progress_footer();
    }
    if let Some(p) = &mut producer {
        let shutdown = Instant::now();
        stats.discarded_prefetch_batches = p.finish()?;
        stats.producer_shutdown_seconds = shutdown.elapsed().as_secs_f64();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let rate = stats.valid_wallets as f64 / elapsed;
    let report = Report {
        cpu_threads: rayon::current_num_threads(),
        requested_suffix: suffix.as_deref(),
        expected_match_seconds: suffix
            .as_ref()
            .filter(|_| rate > 0.0)
            .map(|s| 64f64.powi(s.len() as i32) / rate),
        result,
        setup_seconds,
        total_wall_seconds: setup_seconds + elapsed,
        kernel_cache_hit: gpu.as_ref().map(|g| g.kernel_cache_hit),
        address_engine: if gpu_addresses { "cuda" } else { "cpu" },
        mnemonic_scheme: engine.mode.name(),
        derivation_path: (engine.mode == MnemonicMode::Bip3912).then_some(bip39_ton::PATH),
        backend: engine.backend,
        device,
        batch_size: n,
        pipeline,
        gpu_memory: engine.gpu_memory,
        phrase_engine: if gpu_phrases { "cuda" } else { "cpu" },
        vram_workspace_bytes: gpu
            .as_ref()
            .filter(|_| resident)
            .and_then(|g| g.workspace_bytes()),
        elapsed_seconds: elapsed,
        wallets_per_second: rate,
        expected_ai_seconds: 4096.0 / rate,
        expected_underscore_ai_seconds: 262144.0 / rate,
        gpu_cpu_self_test: gpu.is_some(),
        stats: &stats,
    };
    let json = serde_json::to_string_pretty(&report)?;
    match display.format {
        OutputFormat::Json => println!("{json}"),
        OutputFormat::Table => console::summary(&report),
    }
    if let Some(path) = report_path {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(json.as_bytes())?;
    }
    if suffix.is_some() && !found {
        bail!("search stopped without a match; run again to search fresh random phrases");
    }
    Ok(())
}

fn main() -> Result<()> {
    let started = Instant::now();
    let cli = Cli::parse();
    // Configure optional DLL search BEFORE Rayon or the producer starts threads.
    let needs_cuda = match &cli.command {
        Command::SelfTest { .. } => true,
        Command::Bench { engine, .. } | Command::Search { engine, .. } => {
            matches!(engine.backend, Backend::Cuda)
        }
        Command::Verify { .. } => false,
    };
    let _cuda = if needs_cuda {
        Some(runtime::configure(cli.cuda_path.as_deref())?)
    } else {
        None
    };
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build_global()
            .context("configuring CPU worker pool")?;
    }
    let display = DisplayOptions {
        started,
        format: cli.format,
        quiet: cli.quiet,
        progress_interval: cli.progress_interval,
    };
    match cli.command {
        Command::SelfTest { device } => {
            let g = gpu::Gpu::new(device)?;
            g.self_test()?;
            match cli.format {
                OutputFormat::Table => println!("CUDA/CPU self-test passed on {}", g.name),
                OutputFormat::Json => {
                    println!("{}", serde_json::json!({"passed":true,"device":g.name}))
                }
            }
            Ok(())
        }
        Command::Verify { wallet } => verify(&wallet, cli.format),
        Command::Bench {
            engine,
            seconds,
            report,
        } => run(engine, seconds, None, None, report, display),
        Command::Search {
            engine,
            suffix,
            max_seconds,
            output,
            report,
        } => run(engine, max_seconds, Some(suffix), output, report, display),
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    #[test]
    fn rejects_dual_valid_bip39_ton_winner_before_export() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/bip39-fixture.json")).unwrap();
        let case = &fixture["ambiguous_case"];
        let entropy: [u8; 16] = hex::decode(case["entropy"].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        let row = bip39_ton::indexes(&entropy);
        let phrase = bip39_ton::phrase(&row);
        assert!(ton::valid(&ton::entropy(&phrase)));
        let prepared = PreparedBatch {
            words: Zeroizing::new(vec![row]),
            keys: None,
            generation_seconds: 0.0,
        };
        let mut stats = Stats::default();
        let wallet = batch(
            None,
            prepared,
            &mut stats,
            Some(&case["address"].as_str().unwrap()[40..]),
            false,
            MnemonicMode::Bip3912,
            false,
        )
        .unwrap();
        assert!(wallet.is_none());
        assert_eq!(stats.ambiguous_winners_rejected, 1);
    }

    #[test]
    fn prefetched_words_and_keys_restore_the_official_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/ton-fixture.json")).unwrap();
        let words = fixture["mnemonic"].as_str().unwrap();
        let expected = fixture["address"].as_str().unwrap();
        let mut indexes = [0u16; 24];
        for (index, word) in indexes.iter_mut().zip(words.split_whitespace()) {
            *index = bip39::Language::English
                .word_list()
                .iter()
                .position(|w| *w == word)
                .unwrap() as u16;
        }
        let mut producer = prefetch::Prefetch::new(move || {
            let entropy = ton::entropy_from_indexes(&indexes);
            PreparedBatch {
                words: Zeroizing::new(vec![indexes]),
                keys: Some(Zeroizing::new(vec![entropy])),
                generation_seconds: 0.0,
            }
        })
        .unwrap();
        let mut stats = Stats::default();
        for _ in 0..2 {
            let wallet = batch(
                None,
                producer.take().unwrap(),
                &mut stats,
                Some(&expected[45..]),
                false,
                MnemonicMode::Ton24,
                false,
            )
            .unwrap()
            .unwrap();
            assert_eq!(wallet.address, expected);
            assert_eq!(wallet.mnemonic, words);
        }
        assert_eq!(stats.valid_wallets, 2);
        producer.finish().unwrap();
    }
}
