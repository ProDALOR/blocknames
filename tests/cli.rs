use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn search_displays_live_mean_wait_and_reports_numeric_seconds() {
    let dir = std::env::temp_dir().join(format!(
        "blocknames-estimate-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let report_path = dir.join("metrics.json");
    let output_path = dir.join("test.secret.json");
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args([
            "search",
            "--backend",
            "cpu",
            "--mode",
            "bip39-12",
            "--suffix",
            "_AHHET",
            "--max-seconds",
            "1",
            "--batch-size",
            "16",
            "--threads",
            "2",
            "--report",
        ])
        .arg(&report_path)
        .arg("--output")
        .arg(&output_path)
        .output()
        .unwrap();
    // Both an ordinary duration limit and a lucky early match are valid outcomes.
    assert!(result.status.success() || result.status.code() == Some(1));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Mean wait") && stderr.contains("not a countdown"),
        "{stderr}"
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("Mean search estimate"));
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    let rate = report["wallets_per_second"].as_f64().unwrap();
    let wait = report["expected_match_seconds"].as_f64().unwrap();
    assert!(rate > 0.0 && wait > 0.0);
    assert!((wait * rate / 64f64.powi(6) - 1.0).abs() < 1e-12);
    fs::remove_file(report_path).unwrap();
    if output_path.exists() {
        fs::remove_file(output_path).unwrap();
    }
    fs::remove_dir(dir).unwrap();
}

#[test]
fn auto_backend_uses_cpu_defaults_when_cuda_device_is_unavailable() {
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args([
            "bench",
            "--device",
            "2147483647",
            "--mode",
            "bip39-12",
            "--seconds",
            "1",
            "--threads",
            "2",
            "--quiet",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["requested_backend"], "auto");
    assert_eq!(report["backend"], "cpu");
    assert_eq!(report["batch_size"], 512);
    assert_eq!(report["pipeline"], false);
    assert_eq!(report["gpu_cpu_self_test"], false);
    assert!(
        report["cuda_fallback_reason"]
            .as_str()
            .is_some_and(|reason| !reason.is_empty())
    );
    assert!(report["stats"]["valid_wallets"].as_u64().unwrap() > 0);

    let forced = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args([
            "bench",
            "--backend",
            "cuda",
            "--device",
            "2147483647",
            "--seconds",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!forced.status.success());
    assert!(forced.stdout.is_empty());
}

#[test]
fn twelve_word_wallet_restores_and_wrong_path_is_rejected() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("bip39-fixture.json")).unwrap();
    let case = &fixtures["cases"][0];
    let dir = std::env::temp_dir().join(format!(
        "blocknames-bip39-public-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let path = dir.join("public.json");
    let mut wallet = serde_json::json!({"mnemonic_scheme":"bip39-12","derivation_path":"m/44'/607'/0'","mnemonic":case["mnemonic"],"address":case["address"],"wallet_version":"v5r1","wallet_id":0x7fffff11u32,"network":"mainnet","format":"non-bounceable base64url"});
    fs::write(&path, serde_json::to_vec(&wallet).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args(["verify", "--wallet"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains(case["mnemonic"].as_str().unwrap()));
    wallet["derivation_path"] = "m/44'/607'/1'".into();
    fs::write(&path, serde_json::to_vec(&wallet).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args(["verify", "--wallet"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!result.status.success());
    wallet["derivation_path"] = "m/44'/607'/0'".into();
    wallet["mnemonic"] = fixtures["ambiguous_case"]["mnemonic"].clone();
    wallet["address"] = fixtures["ambiguous_case"]["address"].clone();
    fs::write(&path, serde_json::to_vec(&wallet).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args(["verify", "--wallet"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("ambiguous"));
    fs::remove_file(path).unwrap();
    fs::remove_dir(dir).unwrap();
}

#[test]
fn pipeline_benchmark_exits_at_limit_and_reports_prefetch_cleanup() {
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args([
            "bench",
            "--format",
            "json",
            "--quiet",
            "--threads",
            "2",
            "--backend",
            "cpu",
            "--pipeline",
            "on",
            "--batch-size",
            "512",
            "--seconds",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["pipeline"], true);
    assert_eq!(report["cpu_threads"], 2);
    assert!(report["stats"]["raw_phrases"].as_u64().unwrap() > 0);
    assert!(
        report["stats"]["discarded_prefetch_batches"]
            .as_u64()
            .unwrap()
            <= 1
    );
    assert!(report["stats"]["producer_shutdown_seconds"].is_number());
    assert!(report["stats"]["generation_wait_seconds"].is_number());
    assert!(report.get("mnemonic").is_none());
}

#[test]
fn default_cli_output_is_a_metrics_table() {
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args([
            "bench",
            "--backend",
            "cpu",
            "--mode",
            "bip39-12",
            "--seconds",
            "1",
            "--batch-size",
            "16",
            "--threads",
            "2",
            "--quiet",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.starts_with("+---"));
    assert!(
        stdout.contains("Wallets / second")
            && stdout.contains("Valid wallets")
            && stdout.contains("CPU threads")
    );
    assert!(!stdout.contains("mnemonic\""));
}

#[test]
fn restore_verification_rejects_corruption_and_never_prints_words() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("ton-fixture.json")).unwrap();
    let dir = std::env::temp_dir().join(format!(
        "blocknames-public-fixture-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    let path = dir.join("public-fixture.json");
    let mut wallet = serde_json::json!({
        "address": fixture["address"], "mnemonic": fixture["mnemonic"],
        "wallet_version": "v5r1", "wallet_id": 0x7fffff11u32,
        "network": "mainnet", "format": "non-bounceable base64url"
    });
    fs::write(&path, serde_json::to_vec(&wallet).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args(["verify", "--wallet"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(
        !String::from_utf8_lossy(&result.stdout).contains(fixture["mnemonic"].as_str().unwrap())
    );
    wallet["address"] = "UQ_invalid".into();
    fs::write(&path, serde_json::to_vec(&wallet).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args(["verify", "--wallet"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("words do not restore"));
    // Reject a pre-existing output before any GPU work or secret replacement.
    let before = fs::read(&path).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args(["search", "--output"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("refusing to replace"));
    assert_eq!(before, fs::read(&path).unwrap());
    // A report path must also never overwrite a pre-existing wallet file.
    let result = Command::new(env!("CARGO_BIN_EXE_blocknames"))
        .args(["bench", "--report"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(before, fs::read(&path).unwrap());
    fs::remove_file(&path).unwrap();
    fs::remove_dir(&dir).unwrap();
}
