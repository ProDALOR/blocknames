# Architecture

The application is a Rust/clap binary. It embeds its CUDA C++ kernels and public
Ed25519 lookup table. NVRTC compiles kernels for the selected GPU at runtime; no
launcher, Node.js, Python, or `nvcc` is needed to run it.

| File | Responsibility |
| --- | --- |
| `src/main.rs` | CLI options, candidate batches, recovery checks, output files |
| `src/console.rs` | Human-readable progress/summary tables |
| `src/runtime.rs` | Optional runtime library discovery and platform data paths |
| `src/bip39_ton.rs` | Independent CPU BIP39 / SLIP-0010 reference |
| `src/ton.rs` | Legacy TON mnemonic KDF, W5 hashing, CRC and binary suffix checks |
| `src/gpu.rs` | CUDA allocation, execution, kernel cache and cross-checks |
| `src/pbkdf2.cu` | Mnemonic HMAC, PBKDF2, validation and SLIP-0010 kernels |
| `src/addresses.cu` | Fixed-base Ed25519, W5 hashing, CRC and suffix kernels |
| `src/prefetch.rs` | Bounded one-batch CPU lookahead and worker cleanup |

## Recovery schemes

- `ton24`: independent random words, legacy TON mnemonic validation and full
  100,000-round PBKDF2-HMAC-SHA512.
- `bip39-12`: 128 OS-random bits with BIP39 checksum, full 2,048-round
  PBKDF2-HMAC-SHA512, empty passphrase, SLIP-0010 Ed25519 at `m/44'/607'/0'`.

Both produce W5R1/mainnet wallets with default wallet ID `0x7fffff11`. Friendly
addresses are non-bounceable Base64url (`UQ...`). Changing the display flags changes
the checksum and suffix even when the underlying account ID is unchanged.

Every winning phrase is independently rederived on CPU before export. BIP39
winners that also validate as legacy TON are rejected to avoid ambiguous Keeper
scheme detection. Recovery files record the scheme and path. Legacy files with
no scheme metadata are still read as `ton24`.

## Device memory and ownership

The resident workspace retains allocation capacity, not previous secret contents.
It is cleared after each batch, on error where possible, and on drop. Validation
compacts passing indexes directly on GPU; indexed derivation consumes them without
a CPU gather. Index/key ordering must remain consistent because atomic append
order is unspecified. Reservations grow using actual counts, never probability
as an array bound.

GPU address mode keeps intermediate private seeds on-device. It returns only a
matching index/seed and a public benchmark checksum. CPU address mode remains an
independent comparison path. The CPU producer holds at most one future batch and
is joined on match, duration limit, Ctrl+C, or error.

The public 96 KiB fixed-base table is reproducible independently:

```text
cargo run --locked --release --example generate_ed25519_table -- --check
```

Compiled public PTX is cached using source, architecture, flags, and NVRTC version.
The cache contains no wallet secrets. GPU phase timings use CUDA events rather
than asynchronous host submission timestamps.

## Validation and limits

`cargo test --locked --release` needs no GPU. `blocknames self-test` compares CUDA
with RustCrypto/dalek and SDK-derived public fixtures: dictionary and padding
boundaries, both derivation schemes, compaction, buffer growth/reuse, Ed25519/W5
addresses, suffix lengths 1–8, and complete winner handoffs.

This project has not received an independent security audit. Buffer clearing does
not guarantee protection against process inspection, paging, crash dumps, GPU
register/local-memory remnants, or copies inside runtime/cryptography libraries.
Verify recovery and the receiving address in the intended wallet before funding.
