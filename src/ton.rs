//! Standard passwordless TON mnemonics and W5R1/mainnet wallet addresses.
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::SigningKey;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256, Sha512};

pub const VALIDATION_SALT: &[u8] = b"TON seed version";
pub const WALLET_SALT: &[u8] = b"TON default seed";
pub const WALLET_ID: u32 = 0x7fffff11;

pub fn phrase(indexes: &[u16; 24]) -> String {
    let words = bip39::Language::English.word_list();
    indexes
        .iter()
        .map(|&i| words[i as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn entropy(words: &str) -> [u8; 64] {
    entropy_bytes(words.as_bytes())
}

fn entropy_bytes(words: &[u8]) -> [u8; 64] {
    let mut h = Hmac::<Sha512>::new_from_slice(words).unwrap();
    h.update(b"");
    h.finalize().into_bytes().into()
}

pub fn entropy_from_indexes(indexes: &[u16; 24]) -> [u8; 64] {
    // Longest English BIP39 word is 8 bytes: 24*8 + 23 spaces fits here.
    let mut words = zeroize::Zeroizing::new([0u8; 256]);
    let mut used = 0;
    let list = bip39::Language::English.word_list();
    for (i, &index) in indexes.iter().enumerate() {
        if i > 0 {
            words[used] = b' ';
            used += 1;
        }
        let word = list[index as usize].as_bytes();
        words[used..used + word.len()].copy_from_slice(word);
        used += word.len();
    }
    entropy_bytes(&words[..used])
}

pub fn derive(entropy: &[u8; 64], salt: &[u8], rounds: u32) -> [u8; 64] {
    let mut out = [0; 64];
    pbkdf2::pbkdf2_hmac::<Sha512>(entropy, salt, rounds, &mut out);
    out
}

pub fn valid(entropy: &[u8; 64]) -> bool {
    derive(entropy, VALIDATION_SALT, 390)[0] == 0
}

// Pinned W5R1 code cell metadata, verified against @ton/ton SDK fixtures in tests/.
pub const CODE_HASH: [u8; 32] = [
    0x20, 0x83, 0x4b, 0x7b, 0x72, 0xb1, 0x12, 0x14, 0x7e, 0x1b, 0x2f, 0xb4, 0x57, 0xb8, 0x4e, 0x74,
    0xd1, 0xa3, 0x0f, 0x04, 0xf7, 0x37, 0xd4, 0xf6, 0x2a, 0x66, 0x8e, 0x95, 0x52, 0xd2, 0xb7, 0x2f,
];
pub const CODE_DEPTH: u16 = 6;

// Cache fixed cell descriptors, flags, seqno, wallet ID and top-up bits at compile
// time. Public-key bits begin at offset 65, crossing a byte boundary; only those
// 256 bits vary. This is the cell representation to hash, NOT a serialized BOC.
const fn data_template() -> [u8; 43] {
    let mut repr = [0u8; 43];
    // Ordinary level-zero cell, no references. d2=40+41=81 for 322 data bits.
    repr[1] = 81;
    repr[2] = 0x80;
    let mut j = 0;
    while j < 32 {
        let bit = 33 + j;
        if (WALLET_ID >> (31 - j)) & 1 != 0 {
            repr[2 + bit / 8] |= 1 << (7 - bit % 8);
        }
        j += 1;
    }
    repr[42] = 0x20; // Top-up marker follows 322 actual data bits.
    repr
}
pub const DATA_TEMPLATE: [u8; 43] = data_template();
const fn state_template() -> [u8; 71] {
    let mut repr = [0u8; 71];
    repr[0] = 2;
    repr[1] = 1;
    // StateInit bits 00110 plus a top-up marker: optional depth/special absent,
    // code and data references present, empty library. Reference depths precede hashes.
    repr[2] = 0x34;
    repr[3] = (CODE_DEPTH >> 8) as u8;
    repr[4] = CODE_DEPTH as u8;
    let mut j = 0;
    while j < 32 {
        repr[7 + j] = CODE_HASH[j];
        j += 1;
    }
    repr
}
pub const STATE_TEMPLATE: [u8; 71] = state_template();

pub fn address(seed: &[u8; 64]) -> String {
    let key = SigningKey::from_bytes(seed[..32].try_into().unwrap());
    address_from_public_key(key.verifying_key().as_bytes())
}

pub fn address_from_public_key(public_key: &[u8; 32]) -> String {
    encode_address(&friendly_from_public_key(public_key))
}

pub fn friendly(seed: &[u8; 64]) -> [u8; 36] {
    let key = SigningKey::from_bytes(seed[..32].try_into().unwrap());
    friendly_from_public_key(key.verifying_key().as_bytes())
}

pub fn encode_address(bytes: &[u8; 36]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn friendly_from_public_key(public_key: &[u8; 32]) -> [u8; 36] {
    let mut data = DATA_TEMPLATE;
    for (j, &b) in public_key.iter().enumerate() {
        data[10 + j] |= b >> 1;
        data[11 + j] |= b << 7;
    }
    let data_hash = Sha256::digest(data);
    let mut state = STATE_TEMPLATE;
    state[39..].copy_from_slice(&data_hash);
    let hash = Sha256::digest(state);
    let mut friendly = [0u8; 36];
    // Friendly mainnet/non-bounceable form. Changing this flag to 0x11 produces
    // EQ instead of UQ AND changes the CRC/suffix, even for the same account ID.
    friendly[0] = 0x51;
    friendly[2..34].copy_from_slice(&hash);
    let crc = crc16(&friendly[..34]);
    friendly[34..].copy_from_slice(&crc.to_be_bytes());
    friendly
}

const fn crc_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut j = 0;
        while j < 8 {
            c = (c << 1) ^ if c & 0x8000 != 0 { 0x1021 } else { 0 };
            j += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}
const CRC: [u16; 256] = crc_table();
fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in bytes {
        crc = (crc << 8) ^ CRC[((crc >> 8) as u8 ^ b) as usize];
    }
    crc
}

/// A friendly address is exactly 36 bytes / 48 Base64url characters, with no
/// padding. Its last <=8 characters are exactly the last <=48 bits. Compare those
/// bits directly and encode only a winning address; the CRC is still mandatory.
pub struct Suffix {
    pub bits: i32,
    pub value: u64,
    pub mask: u64,
}
impl Suffix {
    pub fn new(text: &str) -> Self {
        let mut value = 0;
        for b in text.bytes() {
            let digit = match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => b - b'a' + 26,
                b'0'..=b'9' => b - b'0' + 52,
                b'-' => 62,
                b'_' => 63,
                _ => unreachable!(),
            };
            value = (value << 6) | digit as u64;
        }
        let bits = (text.len() * 6) as i32;
        Self {
            bits,
            value,
            mask: (1u64 << bits) - 1,
        }
    }
    pub fn matches(&self, address: &[u8; 36]) -> bool {
        let tail = address[30..].iter().fold(0u64, |n, &b| (n << 8) | b as u64);
        tail & self.mask == self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_sdk_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/ton-fixture.json")).unwrap();
        let words = fixture["mnemonic"].as_str().unwrap();
        let e = entropy(words);
        assert_eq!(hex::encode(e), fixture["entropy"]);
        assert!(valid(&e));
        let seed = derive(&e, WALLET_SALT, 100_000);
        assert_eq!(hex::encode(seed), fixture["seed"]);
        assert_eq!(address(&seed), fixture["address"]);
        for case in fixture["public_key_cases"].as_array().unwrap() {
            let key: [u8; 32] = hex::decode(case["public_key"].as_str().unwrap())
                .unwrap()
                .try_into()
                .unwrap();
            assert_eq!(address_from_public_key(&key), case["address"]);
        }
    }

    #[test]
    fn allocation_free_entropy_matches_text_across_word_list() {
        // Covers short/long words, spaces, and HMAC keys either side of 128 bytes.
        for word in 0..2048u16 {
            let indexes = [word; 24];
            assert_eq!(entropy_from_indexes(&indexes), entropy(&phrase(&indexes)));
        }
    }

    #[test]
    fn binary_suffix_matches_base64url_including_checksum_bits() {
        for byte in 0..=255u8 {
            let raw = friendly_from_public_key(&[byte; 32]);
            let encoded = encode_address(&raw);
            for len in 1..=8 {
                let suffix = &encoded[48 - len..];
                assert!(Suffix::new(suffix).matches(&raw));
                for other in ["_ai", "pretty", "-", "A", "________"] {
                    assert_eq!(Suffix::new(other).matches(&raw), encoded.ends_with(other));
                }
            }
        }
    }
}
