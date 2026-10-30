#[path = "build/ed25519_table.rs"]
mod ed25519_table;

use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

fn main() {
    // Cargo caches the generated file and reruns this only when its sources change.
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=build/ed25519_table.rs");

    let bytes = ed25519_table::generate();
    assert_eq!(bytes.len(), ed25519_table::TABLE_LEN);
    // Pin the digest of the original, GPU-verified table so generator changes
    // cannot silently change the embedded CUDA data format or curve constants.
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        ed25519_table::SHA256,
        "generated Ed25519 table differs from the verified reference"
    );

    // Build artifacts belong in OUT_DIR, never in the source tree. Generation is
    // host-side Rust and needs no GPU, CUDA toolkit, or external programs.
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    fs::write(output.join("ed25519-base.bin"), bytes).expect("write generated Ed25519 table");
}
