# blocknames

A local Rust CLI for generating TON W5R1 wallet addresses with a chosen ending.
CPU and NVIDIA CUDA backends support 12-word BIP39 and legacy 24-word TON recovery.
Keys are generated locally; the app never sends recovery material to a service.

## Quick start

Download the Windows executable or ZIP from [Releases](https://github.com/ProDALOR/blocknames/releases).
No launcher scripts, Node.js, Python, or `nvcc` are required.

```powershell
.\blocknames.exe search --mode bip39-12 --suffix _ai
.\blocknames.exe bench --mode bip39-12 --seconds 30
```

CUDA requires an NVIDIA driver and CUDA 13 NVRTC plus its matching builtins library.
Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), place those
libraries beside the executable, or point to their directory:

```powershell
.\blocknames.exe search --mode bip39-12 --suffix _ai --cuda-path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
```

The CLI also checks `CUDA_PATH` and its user-local `blocknames/cuda` directory.
NVIDIA libraries are not bundled in releases. CPU mode needs no CUDA installation:

```powershell
.\blocknames.exe search --backend cpu --mode bip39-12 --suffix _ai
```

## Build from source

Requires Rust 1.88+ and a native Rust linker (MSVC Build Tools on Windows).
CUDA headers and SDK libraries are not needed at build time: runtime loading is dynamic.
Cargo automatically generates the public 96 KiB Ed25519 lookup table in `OUT_DIR`,
checks its SHA-256 against the verified reference, and embeds it in the executable.
The generated binary stays under `target/` and is never written into the source tree.

```text
cargo build --locked --release
cargo run --locked --release -- search --mode bip39-12 --suffix _ai
```

The Windows executable is `target/release/blocknames.exe`; Unix builds produce
`target/release/blocknames`. Run it directly. Existing commands default to `ton24`;
select `--mode bip39-12` for the faster 12-word scheme.

## Commands and options

```text
blocknames search --mode bip39-12 --suffix _ai
blocknames search --mode bip39-12 --suffix pretty --max-seconds 3600
blocknames bench --mode bip39-12 --seconds 30 --report benchmark.json
blocknames self-test
blocknames verify --wallet path/to/wallet.secret.json
blocknames --help
blocknames search --help
```

| Option | Purpose |
| --- | --- |
| `--mode ton24\|bip39-12` | Recovery scheme; default `ton24` |
| `--backend cuda\|cpu` | Compute backend; default `cuda` |
| `--device N` | CUDA device index; default `0` |
| `--cuda-path DIR` | CUDA toolkit root or NVRTC library directory |
| `--threads N` | CPU worker count; default available logical processors |
| `--batch-size N` | Candidate batch size; tuned defaults vary by mode/backend |
| `--pipeline auto\|on\|off` | Prepare one CPU batch ahead of GPU work |
| `--address-engine auto\|cpu\|cuda` | Address engine; resident CUDA uses CUDA automatically |
| `--phrase-engine auto\|cpu\|cuda` | Phrase construction backend |
| `--gpu-memory resident\|transient` | Reuse VRAM or select the comparison implementation |
| `--format table\|json` | Final output; default table |
| `--progress-interval SECONDS` | Progress row frequency; default `1` |
| `--quiet` | Suppress progress/startup messages |
| `--suffix TEXT` | Search only; 1–8 case-sensitive Base64url characters |
| `--max-seconds N` | Search limit; `0` means unlimited |
| `--seconds N` | Benchmark duration; default `30` |
| `--output FILE` | Search recovery file destination; never overwrites existing files |
| `--report FILE` | Public JSON metrics destination; never contains recovery words |

Duration limits and Ctrl+C take effect at batch boundaries. An in-flight prefetched
batch may finish before the worker is joined and cleared. BIP39 CUDA mode requires
resident memory and CUDA phrase construction; CPU mode is also supported.

## Console metrics

Progress rows show elapsed search time, valid wallets, throughput, raw phrases and
VRAM. The final table also shows configuration, CPU threads, startup time, kernel
cache status, target suffix, and the result path. Recovery words are never printed.

For automation, stdout can be a single JSON object; progress stays on stderr:

```text
blocknames bench --mode bip39-12 --seconds 30 --quiet --format json
blocknames search --mode bip39-12 --suffix _ai --quiet --format json
```

JSON includes scheme/path, backend, device, batch, setup/search times, throughput,
workspace bytes, phase counters, and a public address/file path when found. Average
search estimates exclude startup and batch rounding and are not guarantees.

## Recovery and secret files

- `bip39-12`: standard BIP39, empty passphrase, SLIP-0010 Ed25519 at
  `m/44'/607'/0'`. Rare phrases also valid as legacy TON are rejected before export.
- `ton24`: legacy TON mnemonic validation and its full 100,000-round KDF.
- Both use W5R1/mainnet, wallet ID `0x7fffff11`, and the non-bounceable `UQ...` form.
  Switching to the equivalent `EQ...` display changes its checksum and ending.

A winner is independently rederived on CPU before saving a plaintext JSON containing
`mnemonic`, `address`, recovery scheme, path and wallet settings. Import its words
as an existing wallet in Keeper, select W5, and compare the receiving address.
Existing legacy recovery files without scheme metadata remain readable.

Default secret locations:

- Windows: `%LOCALAPPDATA%\blocknames\wallets`
- Linux: `$XDG_DATA_HOME/blocknames/wallets` or `~/.local/share/blocknames/wallets`

Use `--output` to choose another location. Protect the file: anyone with the words
can spend the funds. This is unaudited software; check recovery in the intended
wallet before funding it. No transaction or wallet deployment is performed.

## Validation and releases

```text
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --release
cargo run --locked --release --example generate_ed25519_table -- --check
blocknames self-test
```

CPU tests and CI need no GPU. `self-test` adds live CUDA comparisons against
RustCrypto/dalek and independent SDK-derived fixtures. All fixture wallets are public
and must never be funded. See [test provenance](tests/README.md).

CI checks Windows and Linux. Pushing a version tag matching `Cargo.toml`, such as
`v0.1.0`, builds and publishes a Windows `.exe`, documentation ZIP and `SHA256SUMS`.
The release workflow can also be dispatched for an existing tag. No application
launcher or build scripts are needed; GitHub Actions invokes Cargo directly.

See [architecture](docs/ARCHITECTURE.md) and [reference benchmarks](docs/BENCHMARKS.md).
Local downloads, scripts, JS verification tools, raw logs and generated binaries are
ignored; only Rust sources (including the table generator), embedded CUDA, public
fixtures and documentation are published.

## License

MIT. Public-domain TweetNaCl-derived field operations are documented in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). NVIDIA runtime libraries are separate
components subject to NVIDIA's terms.
