//! Public curve constants, generated with arbitrary-precision arithmetic that is
//! independent of CUDA's radix-2^16 field implementation. No wallet data is used.
use num_bigint::BigUint;
use num_traits::{One, Zero};

pub const TABLE_LEN: usize = 64 * 16 * 3 * 32;
pub const SHA256: &str = "824f58704e88896f617f1942dd33f3a168e5103ae71771e543d4faa56a2a2393";

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
    // Complete extended-coordinate Edwards addition, a=-1.
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

pub fn generate() -> Vec<u8> {
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
    let mut bytes = Vec::with_capacity(TABLE_LEN);
    // Window w stores digit * 16^w * B for all 16 possible 4-bit digits.
    for _window in 0..64 {
        let mut point = identity();
        for _digit in 0..16 {
            let zi = inverse(&point.z);
            let x = (&point.x * &zi) % &p;
            let y = (&point.y * &zi) % &p;
            // CUDA consumes (y+x, y-x, 2*d*x*y), each as 32 little-endian bytes.
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
