//! Keeper-compatible BIP39 (12 words, empty passphrase), SLIP-0010 Ed25519,
//! standard first TON account m/44'/607'/0'. This is NOT legacy TON mnemonic KDF.
use anyhow::{Result, ensure};
use bip39::{Language, Mnemonic};
use hmac::{Hmac, Mac};
use sha2::Sha512;
use zeroize::Zeroizing;

pub const PATH: &str = "m/44'/607'/0'";

pub fn indexes(entropy: &[u8; 16]) -> [u16; 24] {
    // The library supplies BIP39's SHA-256 checksum. All 128 entropy bits are
    // random; there is no fixed phrase prefix, rejected-checksum search, or custom path.
    let mnemonic = Mnemonic::from_entropy(entropy).unwrap();
    let mut out = [0u16; 24];
    for (dst, index) in out.iter_mut().zip(mnemonic.word_indices()) {
        *dst = index as u16;
    }
    out
}

pub fn phrase(words: &[u16; 24]) -> String {
    let list = Language::English.word_list();
    words[..12]
        .iter()
        .map(|&i| list[i as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn private_seed(words: &str) -> Result<[u8; 64]> {
    let mnemonic = Mnemonic::parse_in_normalized(Language::English, words)?;
    ensure!(mnemonic.word_count() == 12, "BIP39 mode requires 12 words");
    // Empty passphrase is deliberate: Keeper recovery requires no extra secret.
    // English generated words are already NFKD normalized ASCII.
    let seed = Zeroizing::new(mnemonic.to_seed_normalized(""));
    Ok(slip10(&seed))
}

pub fn slip10(seed: &[u8; 64]) -> [u8; 64] {
    fn hmac(key: &[u8], data: &[u8]) -> [u8; 64] {
        let mut h = Hmac::<Sha512>::new_from_slice(key).unwrap();
        h.update(data);
        h.finalize().into_bytes().into()
    }
    let mut extended = Zeroizing::new(hmac(b"ed25519 seed", seed));
    for index in [44u32, 607, 0] {
        let mut data = Zeroizing::new([0u8; 37]);
        data[1..33].copy_from_slice(&extended[..32]);
        data[33..].copy_from_slice(&(index | 0x80000000).to_be_bytes());
        *extended = hmac(&extended[32..], &*data);
    }
    // Shared address routines take 64 bytes, but only the first 32 are an
    // Ed25519 private seed. The remainder is zero, not a retained chain code.
    let mut result = [0u8; 64];
    result[..32].copy_from_slice(&extended[..32]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keeper_path_matches_independent_javascript_fixtures() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/bip39-fixture.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let entropy: [u8; 16] = hex::decode(case["entropy"].as_str().unwrap())
                .unwrap()
                .try_into()
                .unwrap();
            let words = phrase(&indexes(&entropy));
            assert_eq!(words, case["mnemonic"]);
            let key = private_seed(&words).unwrap();
            assert_eq!(hex::encode(&key[..32]), case["private_seed"]);
            assert_eq!(crate::ton::address(&key), case["address"]);
        }
    }
}
