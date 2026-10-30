//! Plain ASCII tables: readable in Windows terminals and safe in redirected logs.
use crate::{Report, Stats, estimate};
use tabled::{
    Table,
    builder::Builder,
    settings::{Alignment, Style, Width, object::Columns},
};

const PROGRESS_LABELS: [&str; 6] = [
    "Elapsed",
    "Valid wallets",
    "Wallets/s",
    "Raw phrases",
    "VRAM",
    "Mean wait",
];

fn progress_table(record: [String; 6]) -> Table {
    let mut table = Builder::from_iter([record]).build();
    table.with(Style::ascii()).with(Alignment::right());
    // Fixed column sizes let progress append one row at a time in redirected
    // logs as well as interactive terminals. tabled owns padding and borders.
    for (column, width) in [14, 16, 14, 16, 11, 24].into_iter().enumerate() {
        table.modify(
            Columns::new(column..=column),
            Width::truncate(width).suffix("..."),
        );
        table.modify(Columns::new(column..=column), Width::increase(width));
    }
    table
}

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

pub fn progress_header(suffix: Option<&str>) {
    if let Some(suffix) = suffix {
        eprintln!(
            "Target: {suffix} | Mean wait is a statistical estimate, not a countdown or deadline."
        );
    }
    eprintln!("{}", progress_table(PROGRESS_LABELS.map(str::to_owned)));
}
pub fn progress(elapsed: f64, stats: &Stats, vram: Option<usize>, suffix: Option<&str>) {
    let rate = stats.valid_wallets as f64 / elapsed.max(f64::EPSILON);
    let wait = estimate::mean_wait(suffix, rate).map_or_else(
        || {
            if suffix.is_some() {
                "warming up".into()
            } else {
                "-".into()
            }
        },
        estimate::duration,
    );
    let mut table = progress_table([
        estimate::duration(elapsed),
        count(stats.valid_wallets),
        count(rate.round() as u64),
        count(stats.raw_phrases),
        bytes(vram),
        wait,
    ]);
    table.with(
        Style::ascii()
            .remove_top()
            .remove_bottom()
            .remove_horizontals(),
    );
    eprintln!("{table}");
}
pub fn progress_footer() {
    let table = progress_table(PROGRESS_LABELS.map(str::to_owned)).to_string();
    if let Some(border) = table.lines().last() {
        eprintln!("{border}");
    }
}

pub fn summary(report: &Report<'_>) {
    let mut rows = vec![
        ("Device", report.device.to_owned()),
        ("Recovery mode", report.mnemonic_scheme.to_owned()),
        ("Backend", format!("{:?}", report.backend)),
        (
            "Backend selection",
            format!("{:?}", report.requested_backend),
        ),
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
        ("Search time", estimate::duration(report.elapsed_seconds)),
        ("Startup + checks", estimate::duration(report.setup_seconds)),
        ("Total time", estimate::duration(report.total_wall_seconds)),
        ("VRAM workspace", bytes(report.vram_workspace_bytes)),
        (
            "Kernel cache",
            report
                .kernel_cache_hit
                .map_or("not used", |hit| if hit { "hit" } else { "miss" })
                .into(),
        ),
    ];
    if let Some(path) = &report.cuda_runtime_library {
        rows.push(("CUDA runtime", path.display().to_string()));
    }
    if let Some(reason) = &report.cuda_fallback_reason {
        rows.push(("CUDA fallback", reason.replace(['\r', '\n'], " ")));
    }
    if let Some(suffix) = report.requested_suffix {
        rows.push(("Requested suffix", suffix.into()));
    }
    if let Some(seconds) = report.expected_match_seconds {
        rows.push((
            "Mean search estimate",
            format!("{} (random; excludes startup)", estimate::duration(seconds)),
        ));
    }
    if let Some(result) = &report.result {
        rows.push(("Found address", result.address.clone()));
        rows.push((
            "Recovery words file",
            result.wallet_file.display().to_string(),
        ));
    }
    let mut builder = Builder::default();
    for (name, value) in rows {
        builder.push_record([name, &value]);
    }
    let mut table = builder.build();
    table
        .with(Style::ascii().remove_horizontals().remove_horizontal())
        .with(Alignment::left());
    println!("{table}");
}
