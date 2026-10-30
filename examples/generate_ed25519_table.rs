//! Reproduce the public Ed25519 table and check its pinned reference digest.
//! Run: cargo run --release --example generate_ed25519_table -- --check
use anyhow::{Result, ensure};
use clap::Parser;
use sha2::{Digest, Sha256};
#[path = "../build/ed25519_table.rs"]
mod ed25519_table;
use std::{fs::OpenOptions, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    /// Verify the embedded table byte-for-byte (also the default with no output).
    #[arg(long)]
    check: bool,
    /// Write a public table to a new file. Existing files are not replaced.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = ed25519_table::generate();
    if args.check || args.output.is_none() {
        ensure!(
            bytes == include_bytes!(concat!(env!("OUT_DIR"), "/ed25519-base.bin")),
            "embedded Ed25519 table differs from the generator"
        );
        ensure!(
            bytes.len() == ed25519_table::TABLE_LEN,
            "incorrect table length"
        );
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == ed25519_table::SHA256,
            "table differs from pinned reference digest"
        );
        println!(
            "Verified all {} public table bytes against the reference digest",
            bytes.len()
        );
    }
    if let Some(path) = args.output {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?
            .write_all(&bytes)?;
        println!("Wrote {} public bytes to {}", bytes.len(), path.display());
    }
    Ok(())
}
