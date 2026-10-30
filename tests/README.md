# Public test vectors

These fixtures are public and disposable. Never fund their addresses or use their
recovery words as a real wallet backup.

- `ton-fixture.json`: legacy TON mnemonic, seed, W5 address, code-cell metadata,
  and synthetic public-key address cases. Produced independently using
  `@ton/crypto 3.3.0` and `@ton/ton 16.0.0`.
- `bip39-fixture.json`: BIP39 entropy/words/seed, hardened SLIP-0010 derivation at
  `m/44'/607'/0'`, W5 address, and a dual-valid BIP39/TON phrase. Produced using
  `bip39 3.1.0` plus the same TON SDK versions, with an empty BIP39 passphrase.
- `cli.rs`: process-level restoration, corruption/path rejection, overwrite
  protection, JSON/table output, and pipeline shutdown checks.

`cargo test --locked` runs without CUDA or Node.js. `blocknames self-test` adds
live GPU comparisons against RustCrypto/dalek, including both complete recovery
schemes, suffix matching, compaction, workspace reuse, and public-key generation.

The public Ed25519 lookup table is independently reproducible in Rust:

```text
cargo run --locked --release --example generate_ed25519_table -- --check
```
