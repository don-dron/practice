#!/usr/bin/env python3
"""
Рисунки для п. 5.2: данные читаются из CSV в reports/registry_synthetic/ (синтетический «лог» для отчёта).

Запуск из корня репозитория:
  python3 scripts/plot_section_5_2_registry.py
  python3 scripts/plot_section_5_2_registry.py --data-dir reports/registry_synthetic --out reports/figures

Файлы данных:
  ctc_train_loss_epochs.csv              — средний train_loss по эпохам (CTC)
  attention_s2_normalized_ce_epochs.csv  — мониторинг CE для S2
  wall_clock_hours.csv                   — ориентировочный wall-clock 5 эпох
  proxy_ser_star.csv                     — условный ŜER* (пустое — пропуск)
  held_out_proxy_ser_by_epoch.csv       — траектория прокси SER по эпохам (СКО с CTC-связками)
  registry_run_summary.csv               — VRAM latency throughput сводка (иллюстрация для отчёта)

Создаваемые PNG (в каталог `--out`, по умолчанию reports/figures/):
  fig_5_1_ctc_training_montage.png  — четыре панели CTC-loss на одном макете (а–г)
  fig_5_s2_normalized_ce.png       — траектория CE для S2
  fig_5_wallclock_hours.png       — столбики wall-clock
  fig_5_proxy_ser_bar.png         — столбики прокси SER
  fig_5_heldout_ser_by_epoch.png  — траектория SER по эпохам
  fig_5_pareto_ser_vs_wallclock.png
  fig_5_infer_latency_ms.png
  fig_5_8_quality_aggregate_panel.png — две столбиковые панели (композит + ĈER*)

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WALL_COLOR_MAP = {"T0": "#1f77b4", "S2": "#9467bd", "S3": "#2ca02c", "S5": "#ff7f0e"}

SER_BAR_ORDER = ["T0", "E1", "E2", "S2", "S3", "S5"]

SER_BAR_COLORS = {
    "T0": "#1f77b4",
    "E1": "#aec7e8",
    "E2": "#9edae5",
    "S2": "#9467bd",
    "S3": "#2ca02c",
    "S5": "#ff7f0e",
}


def load_ctc_train_loss_epochs(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return _load_wide_first_col_epoch(path, "epoch")


def _load_wide_first_col_epoch(path: Path, first_col_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or reader.fieldnames[0].strip().lower() != first_col_name.strip().lower():
            raise ValueError(f"{path}: ожидается первая колонка {first_col_name!r}")
        keys = [k for k in reader.fieldnames if k and k.strip().lower() != first_col_name.strip().lower()]
        epochs: list[float] = []
        cols: dict[str, list[float]] = {k: [] for k in keys}
        for row in reader:
            epochs.append(float(row[reader.fieldnames[0]].strip()))
            for k in keys:
                cols[k].append(float(row[k].strip()))
    return np.array(epochs), {k: np.array(cols[k]) for k in keys}


def load_two_column_floats(path: Path, col_x: str, col_y: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        xs: list[float] = []
        ys: list[float] = []
        for row in reader:
            xs.append(float(row[col_x].strip()))
            ys.append(float(row[col_y].strip()))
    return np.array(xs), np.array(ys)


def load_wall_clock_hours(path: Path) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            run = row["run"].strip()
            out.append((run, float(row["hours_wallclock_epochs_1_to_5"].strip())))
    return out


def load_proxy_ser(path: Path) -> dict[str, float]:
    d: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            run = row["run"].strip()
            raw = row.get("proxy_ser_star", "").strip()
            if raw == "":
                continue
            d[run] = float(raw)
    return d


def _parse_opt_float(s: str) -> float | None:
    t = (s or "").strip()
    if t == "":
        return None
    return float(t)


def _parse_opt_int(s: str) -> int | None:
    t = (s or "").strip()
    if t == "":
        return None
    return int(float(t))


def load_run_summary_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            run = row["run"].strip()
            comparable = row.get("ctc_infer_comparable", "").strip().lower() == "true"
            rec: dict[str, Any] = {
                "run": run,
                "proxy_ser_star": _parse_opt_float(row.get("proxy_ser_star", "")),
                "proxy_cer_star": _parse_opt_float(row.get("proxy_cer_star", "")),
                "wallclock_h_5ep": _parse_opt_float(row.get("wallclock_h_5ep", "")),
                "peak_vram_gib": _parse_opt_float(row.get("peak_vram_gib", "")),
                "infer_latency_ms_mean": _parse_opt_float(row.get("infer_latency_ms_mean", "")),
                "epoch_first_loss_below_55": _parse_opt_int(row.get("epoch_first_loss_below_55", "")),
                "inference_throughput_chars_s_approx": _parse_opt_float(
                    row.get("inference_throughput_chars_s_approx", "")
                ),
                "ctc_infer_comparable": comparable,
            }
            rows.append(rec)
    return rows


def _min_max_norm(vals: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi - lo < 1e-9:
        return np.ones_like(vals)
    return (vals - lo) / (hi - lo)


def _style_axes(ax, epochs: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xticks(epochs)


def _panel_corner_label(ax: Any, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="#222222",
    )


def plot_ctc_training_montage(out: Path, epochs: np.ndarray, cols: dict[str, np.ndarray]) -> None:
    """Четыре панели на одном макете: заменяет четыре отдельных графика CTC train loss без дублирования вставок."""
    fig, axes = plt.subplots(2, 2, figsize=(9.9, 7.95), dpi=120)
    t0, s5, s3 = cols["T0"], cols["S5"], cols["S3"]

    ax = axes[0, 0]
    ax.plot(epochs, t0, "o-", color="#1f77b4", linewidth=2, markersize=6)
    _panel_corner_label(ax, "(а)")
    ax.set_title("Mean train loss — T0")
    _style_axes(ax, epochs, "Epoch", "CTC train loss", "")

    ax = axes[0, 1]
    ax.plot(epochs, t0, "s-", color="#1f77b4", linewidth=2, markersize=5, label="T0")
    ax.plot(epochs, s5, "^-", color="#ff7f0e", linewidth=2, markersize=5, label="S5")
    _panel_corner_label(ax, "(б)")
    ax.set_title("T0 vs S5")
    ax.legend(loc="upper right", fontsize=8)
    _style_axes(ax, epochs, "Epoch", "CTC train loss", "")

    ax = axes[1, 0]
    ax.plot(epochs, s3, "D-", color="#2ca02c", linewidth=2, markersize=5)
    _panel_corner_label(ax, "(в)")
    ax.set_title("S3 (transformer + CTC)")
    _style_axes(ax, epochs, "Epoch", "CTC train loss", "")

    ax = axes[1, 1]
    ax.plot(epochs, t0, "o-", label="T0", color="#1f77b4", linewidth=1.7)
    ax.plot(epochs, s3, "s-", label="S3", color="#2ca02c", linewidth=1.7)
    ax.plot(epochs, s5, "^-", label="S5", color="#ff7f0e", linewidth=1.7)
    _panel_corner_label(ax, "(г)")
    ax.set_title("CTC cohort overlay")
    ax.legend(loc="upper right", fontsize=8)
    _style_axes(ax, epochs, "Epoch", "CTC train loss", "")

    fig.suptitle("Synthetic registry: CTC mean train loss vs epoch", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_quality_aggregate_panel(out: Path, rows: list[dict[str, Any]]) -> None:
    runs_ord, scores = _composite_benchmark_scores(rows)
    if not runs_ord:
        raise ValueError("Нет строк для композитного рейтинга (CSV registry_run_summary)")
    colors_b = ["#ffd700" if r == "T0" else SER_BAR_COLORS.get(r, "#9e9e9e") for r in runs_ord]

    items = [(r["run"], r["proxy_cer_star"]) for r in rows if r["proxy_cer_star"] is not None]
    items.sort(key=lambda t: t[1])
    runs_t = [t[0] for t in items]
    vals = [t[1] for t in items]
    colors_c = ["#ffd700" if rr == "T0" else SER_BAR_COLORS.get(rr, "#7f7f7f") for rr in runs_t]

    fig, axes = plt.subplots(1, 2, figsize=(12.9, 4.55), dpi=120)
    ax0, ax1 = axes[0], axes[1]

    x = np.arange(len(runs_ord))
    bars = ax0.bar(x, scores, color=colors_b, edgecolor="#333333", linewidth=0.55)
    ax0.set_xticks(x)
    ax0.set_xticklabels(runs_ord)
    ax0.set_ylabel("Composite score (higher = better)")
    ax0.set_title("Multi-metric mix (SER · time · VRAM · latency · throughput)")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.bar_label(bars, labels=[f"{s:.3f}" for s in scores], fontsize=8, padding=2)
    _panel_corner_label(ax0, "(а)")

    x2 = np.arange(len(runs_t))
    ax1.bar(x2, vals, color=colors_c, edgecolor="#333333", linewidth=0.55)
    ax1.set_xticks(x2)
    ax1.set_xticklabels(runs_t)
    ax1.set_ylabel("Proxy character error rate (CTC cohort)")
    ax1.set_title("Illustrative proxy CER* (lower = better)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    _panel_corner_label(ax1, "(б)")

    fig.suptitle("Synthetic deployment-quality summary — registry_run_summary", fontsize=11, y=1.06)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_s2_ce(out: Path, epochs: np.ndarray, ce: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    ax.plot(epochs, ce, "v-", color="#9467bd", linewidth=2, markersize=6, label="S2 normalized CE monitor")
    _style_axes(ax, epochs, "Epoch", "Normalized mean CE", "S2 attention (CE objective — not comparable to CTC magnitude)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_wallclock(out: Path, pairs: list[tuple[str, float]]) -> None:
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [WALL_COLOR_MAP.get(n, "#7f7f7f") for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    x = np.arange(len(names))
    ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Wall-clock (hours, 5 epochs)")
    ax.set_title("Train duration from CSV (wall_clock_hours.csv)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ymax = max(vals) if vals else 1.0
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_proxy_ser_bars(out: Path, proxy: dict[str, float]) -> None:
    runs = [r for r in SER_BAR_ORDER if r in proxy]
    ser = [proxy[r] for r in runs]
    colors = [SER_BAR_COLORS.get(r, "#7f7f7f") for r in runs]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    x = np.arange(len(runs))
    ax.bar(x, ser, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylabel("Proxy SER (held-out 3.2k, illustrative)")
    ax.set_title("Proxy ŜER* from CSV (proxy_ser_star.csv)")
    ax.set_ylim(0, max(ser) * 1.35 if ser else 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_heldout_ser_trajectories(out: Path, epochs: np.ndarray, cols: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)
    specs = [
        ("T0", "o-", "#1f77b4", "reference CRNN"),
        ("S5", "^--", "#ff7f0e", "pretrained encoder"),
        ("S3", "s:", "#2ca02c", "transformer encoder"),
    ]
    for key, sty, clr, lbl in specs:
        if key not in cols:
            continue
        ax.plot(epochs, cols[key], sty, color=clr, linewidth=2, markersize=6, label=f"{key} ({lbl})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proxy SER (same held-out 3.2k protocol)")
    ax.set_title("Synthetic held-out SER vs epoch — CTC-comparable cohort (CSV)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xticks(epochs)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_quality_vs_wall(out: Path, rows: list[dict[str, Any]]) -> None:
    pts = [
        r
        for r in rows
        if r["ctc_infer_comparable"]
        and r["proxy_ser_star"] is not None
        and r["wallclock_h_5ep"] is not None
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=120)
    seen_labels: set[str] = set()
    for r in pts:
        x, y = r["wallclock_h_5ep"], r["proxy_ser_star"]
        run = r["run"]
        clr = SER_BAR_COLORS.get(run, "#555555")
        size = 150 if run == "T0" else 82
        label = run if run not in seen_labels else None
        if label:
            seen_labels.add(run)
        ax.scatter([x], [y], s=size, color=clr, alpha=0.88, edgecolors="#222", linewidths=0.6, label=label)
        ax.annotate(run, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Wall-clock train 5 epochs (h)")
    ax.set_ylabel("Proxy SER held-out")
    ax.set_title("Synthetic Pareto view: lower-left is preferable (CSV registry_run_summary)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, title="Runs")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_inference_latency_comparison(out: Path, rows: list[dict[str, Any]]) -> None:
    eligible = sorted(
        (r["run"], r["infer_latency_ms_mean"])
        for r in rows
        if r["ctc_infer_comparable"] and r["infer_latency_ms_mean"] is not None
    )
    runs = [x[0] for x in eligible]
    lats = [x[1] for x in eligible]
    colors = [SER_BAR_COLORS.get(rr, "#7f7f7f") for rr in runs]
    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=120)
    y = np.arange(len(runs))
    ax.barh(y, lats, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(runs)
    ax.set_xlabel("Mean line infer latency — synthetic bench (ms)")
    ax.set_title("Illustrative deployment latency subset (CSV registry_run_summary)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _composite_benchmark_scores(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    """Higher is better. Uses min–max inversion for cost-like metrics."""

    cand = []
    keys = []
    for r in rows:
        if not r["ctc_infer_comparable"]:
            continue
        if None in (
            r["proxy_ser_star"],
            r["wallclock_h_5ep"],
            r["infer_latency_ms_mean"],
            r["peak_vram_gib"],
            r["inference_throughput_chars_s_approx"],
        ):
            continue
        keys.append(r["run"])
        cand.append(r)
    if not cand:
        return [], np.array([])

    ser = np.array([r["proxy_ser_star"] for r in cand], dtype=float)
    wall = np.array([r["wallclock_h_5ep"] for r in cand], dtype=float)
    lat = np.array([r["infer_latency_ms_mean"] for r in cand], dtype=float)
    vram = np.array([r["peak_vram_gib"] for r in cand], dtype=float)
    thr = np.array([r["inference_throughput_chars_s_approx"] for r in cand], dtype=float)

    # lower cost metrics → invert
    parts = np.column_stack(
        [
            1 - _min_max_norm(ser),
            1 - _min_max_norm(wall),
            1 - _min_max_norm(lat),
            1 - _min_max_norm(vram),
            _min_max_norm(thr),
        ]
    )
    scores = parts.mean(axis=1)
    # sort descending
    order = np.argsort(-scores)
    return [keys[i] for i in order], scores[order]



def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="PNG figures from registry_synthetic CSVs.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=root / "reports" / "registry_synthetic",
        help="Directory with CSV tables",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "reports" / "figures",
        help="Output directory for PNG files",
    )
    args = p.parse_args()
    data_dir: Path = args.data_dir
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ctc_path = data_dir / "ctc_train_loss_epochs.csv"
    s2_path = data_dir / "attention_s2_normalized_ce_epochs.csv"
    wall_path = data_dir / "wall_clock_hours.csv"
    ser_path = data_dir / "proxy_ser_star.csv"
    held_path = data_dir / "held_out_proxy_ser_by_epoch.csv"
    summary_path = data_dir / "registry_run_summary.csv"

    for path in (ctc_path, s2_path, wall_path, ser_path, held_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Не найден файл данных: {path}")

    epochs, ctc = load_ctc_train_loss_epochs(ctc_path)
    ep_ce, ce = load_two_column_floats(s2_path, "epoch", "normalized_mean_ce")
    wall_pairs = load_wall_clock_hours(wall_path)
    proxy = load_proxy_ser(ser_path)
    ep_hld, held_ser = _load_wide_first_col_epoch(held_path, "epoch")
    summary_rows = load_run_summary_rows(summary_path)

    required_series = {"T0", "S3", "S5"}
    missing = required_series - set(ctc.keys())
    if missing:
        raise ValueError(f"В {ctc_path.name} нет колонок: {sorted(missing)}")

    outputs: list[tuple[str, Path]] = []
    outs = [
        ("fig_5_1_ctc_training_montage.png", lambda op: plot_ctc_training_montage(op, epochs, ctc)),
        ("fig_5_s2_normalized_ce.png", lambda op: plot_s2_ce(op, ep_ce, ce)),
        ("fig_5_wallclock_hours.png", lambda op: plot_wallclock(op, wall_pairs)),
        ("fig_5_proxy_ser_bar.png", lambda op: plot_proxy_ser_bars(op, proxy)),
        ("fig_5_heldout_ser_by_epoch.png", lambda op: plot_heldout_ser_trajectories(op, ep_hld, held_ser)),
        ("fig_5_pareto_ser_vs_wallclock.png", lambda op: plot_pareto_quality_vs_wall(op, summary_rows)),
        ("fig_5_infer_latency_ms.png", lambda op: plot_inference_latency_comparison(op, summary_rows)),
        ("fig_5_8_quality_aggregate_panel.png", lambda op: plot_quality_aggregate_panel(op, summary_rows)),
    ]
    for name, fn in outs:
        op = out_dir / name
        fn(op)
        outputs.append((name, op))

    for name, op in outputs:
        print(op)

    print("[plot_section_5_2_registry] Графики построены из CSV:", data_dir.resolve())


if __name__ == "__main__":
    main()
