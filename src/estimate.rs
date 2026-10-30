//! Vanity search uses independent random trials. Expected wait is not a deadline
//! or a countdown: previous misses do not reduce the work expected from now on.

pub fn mean_wait(suffix: Option<&str>, wallets_per_second: f64) -> Option<f64> {
    let suffix = suffix?;
    (wallets_per_second.is_finite() && wallets_per_second > 0.0)
        .then(|| 64f64.powi(suffix.len() as i32) / wallets_per_second)
        .filter(|seconds| seconds.is_finite())
}

pub fn duration(seconds: f64) -> String {
    let Ok(duration) = std::time::Duration::try_from_secs_f64(seconds) else {
        return "-".into();
    };
    // The two largest components are enough for a statistical estimate and keep
    // the live table readable. humantime handles units, precision and plurals.
    humantime::format_duration(duration)
        .to_string()
        .split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_suffix_length_and_valid_wallet_rate_without_a_countdown() {
        assert_eq!(mean_wait(Some("_ai"), 262_144.0), Some(1.0));
        assert_eq!(mean_wait(Some("_AHHET"), 1_048_576.0), Some(65_536.0));
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(mean_wait(Some("_ai"), rate), None);
        }
        assert_eq!(mean_wait(None, 1.0), None);
    }

    #[test]
    fn formats_human_readable_waits() {
        assert_eq!(duration(0.16), "160ms");
        assert_eq!(duration(59.6), "59s 600ms");
        assert_eq!(duration(3600.0), "1h");
        assert_eq!(duration(41_400.0), "11h 30m");
        assert_eq!(duration(90_000.0), "1day 1h");
        assert_eq!(duration(f64::INFINITY), "-");
    }
}
