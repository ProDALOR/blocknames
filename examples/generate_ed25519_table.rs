//! Reproduce the PUBLIC Ed25519 fixed-base table using independent BigUint math.
//! Run: cargo run --release --example generate_ed25519_table -- --check
use anyhow::{Result, ensure};
use clap::Parser;
use num_bigint::BigUint;
use num_traits::{One, Zero};
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

#[derive(Clone)]
struct Point {
    x: BigUint,
    y: BigUint,
    z: BigUint,
    t: BigUint,
}
fn identity() -> Point {
    Point {
        x: BigUint::zero(),
        y: BigUint::one(),
        z: BigUint::one(),
        t: BigUint::zero(),
    }
}
fn sub(a: &BigUint, b: &BigUint, p: &BigUint) -> BigUint {
    (a + p - b) % p
}
fn add(a: &Point, b: &Point, p: &BigUint, d: &BigUint) -> Point {
    // Complete extended-coordinate Edwards addition, a=-1. This reference uses
    // arbitrary precision, independently of the CUDA radix-2^16 implementation.
    let aa = (sub(&a.y, &a.x, p) * sub(&b.y, &b.x, p)) % p;
    let bb = ((&a.y + &a.x) * (&b.y + &b.x)) % p;
    let cc = (&a.t * &b.t * d * 2u8) % p;
    let dd = (&a.z * &b.z * 2u8) % p;
    let e = sub(&bb, &aa, p);
    let f = sub(&dd, &cc, p);
    let g = (&dd + &cc) % p;
    let h = (&bb + &aa) % p;
    Point {
        x: (&e * &f) % p,
        y: (&g * &h) % p,
        z: (&f * &g) % p,
        t: (&e * &h) % p,
    }
}

fn table() -> Vec<u8> {
    let p = (BigUint::one() << 255usize) - 19u8;
    let exponent = &p - 2u8;
    let inverse = |n: &BigUint| n.modpow(&exponent, &p);
    let d = (&p - (BigUint::from(121665u32) * inverse(&BigUint::from(121666u32))) % &p) % &p;
    let x = BigUint::parse_bytes(
        b"15112221349535400772501151409588531511454012693041857206046113283949847762202",
        10,
    )
    .unwrap();
    let y = (BigUint::from(4u8) * inverse(&BigUint::from(5u8))) % &p;
    let mut base = Point {
        t: (&x * &y) % &p,
        x,
        y,
        z: BigUint::one(),
    };
    let mut bytes = Vec::with_capacity(98304);
    for _window in 0..64 {
        let mut point = identity();
        for _digit in 0..16 {
            let zi = inverse(&point.z);
            let x = (&point.x * &zi) % &p;
            let y = (&point.y * &zi) % &p;
            for value in [(&y + &x) % &p, sub(&y, &x, &p), (&d * &x * &y * 2u8) % &p] {
                let mut encoded = value.to_bytes_le();
                encoded.resize(32, 0);
                bytes.extend(encoded);
            }
            point = add(&point, &base, &p, &d);
        }
        for _ in 0..4 {
            base = add(&base, &base, &p, &d);
        }
    }
    bytes
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = table();
    if args.check || args.output.is_none() {
        ensure!(
            bytes == include_bytes!("../src/ed25519-base.bin"),
            "embedded Ed25519 table differs from the independent generator"
        );
        println!("Verified all {} public table bytes", bytes.len());
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
