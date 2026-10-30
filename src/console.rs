//! Plain ASCII tables: readable in Windows terminals and safe in redirected logs.
use crate::{Report, Stats};

fn count(n: u64) -> String {
    let text = n.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (i, c) in text.chars().enumerate() {
        if i > 0 && (text.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
fn bytes(n: Option<usize>) -> String {
    n.map_or_else(
        || "-".into(),
        |n| format!("{:.1} MiB", n as f64 / 1048576.0),
    )
}

pub fn progress_header() {
    eprintln!(
        "+------------+------------------+----------------+------------------+-------------+"
    );
    eprintln!(
        "|    Elapsed |    Valid wallets |      Wallets/s |      Raw phrases |        VRAM |"
    );
    eprintln!(
        "+------------+------------------+----------------+------------------+-------------+"
    );
}
pub fn progress(elapsed: f64, stats: &Stats, vram: Option<usize>) {
    let rate = stats.valid_wallets as f64 / elapsed.max(f64::EPSILON);
    eprintln!(
        "| {:>9.2}s | {:>16} | {:>14} | {:>16} | {:>11} |",
        elapsed,
        count(stats.valid_wallets),
        count(rate.round() as u64),
        count(stats.raw_phrases),
        bytes(vram)
    );
}
pub fn progress_footer() {
    eprintln!(
        "+------------+------------------+----------------+------------------+-------------+"
    );
}

pub fn summary(report: &Report<'_>) {
    let mut rows = vec![
        ("Device", report.device.to_owned()),
        ("Recovery mode", report.mnemonic_scheme.to_owned()),
        ("Backend", format!("{:?}", report.backend)),
        ("Address engine", report.address_engine.to_owned()),
        ("Batch size", count(report.batch_size as u64)),
        ("CPU threads", report.cpu_threads.to_string()),
        ("CPU/GPU pipeline", report.pipeline.to_string()),
        ("Valid wallets", count(report.stats.valid_wallets)),
        ("Raw phrases", count(report.stats.raw_phrases)),
        (
            "Wallets / second",
            count(report.wallets_per_second.round() as u64),
        ),
        ("Search time", format!("{:.3} s", report.elapsed_seconds)),
        ("Startup + checks", format!("{:.3} s", report.setup_seconds)),
        ("Total time", format!("{:.3} s", report.total_wall_seconds)),
        ("VRAM workspace", bytes(report.vram_workspace_bytes)),
        (
            "Kernel cache",
            report
                .kernel_cache_hit
                .map_or("not used", |hit| if hit { "hit" } else { "miss" })
                .into(),
        ),
    ];
    if let Some(suffix) = report.requested_suffix {
        rows.push(("Requested suffix", suffix.into()));
    }
    if let Some(seconds) = report.expected_match_seconds {
        rows.push((
            "Mean search estimate",
            format!("{seconds:.3} s (random; excludes startup)"),
        ));
    }
    if let Some(result) = &report.result {
        rows.push(("Found address", result.address.clone()));
        rows.push((
            "Recovery words file",
            result.wallet_file.display().to_string(),
        ));
    }
    let width = rows
        .iter()
        .map(|(_, value)| value.len())
        .max()
        .unwrap_or(16)
        .max(16);
    let line = format!("+----------------------+{}+", "-".repeat(width + 2));
    println!("{line}");
    for (name, value) in rows {
        println!("| {name:<20} | {value:<width$} |");
    }
    println!("{line}");
}
