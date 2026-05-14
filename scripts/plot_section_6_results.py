#!/usr/bin/env python3
"""
Иллюстративные таблицы-источники: reports/section6_synthetic/*.csv
PNG по умолчанию: reports/figures/fig_6_*.png

  python3 scripts/plot_section_6_results.py
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_ctc_t0(epochs_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    with epochs_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        ep, y = [], []
        for row in r:
            ep.append(float(row["epoch"]))
            y.append(float(row["T0"]))
    return np.array(ep), np.array(y)


def plot_sec6_1(path_out: Path, rows: list[dict[str, str]]) -> None:
    """Панель: прокси SER (где число) + задержки infer для всех строк."""
    ser_items: list[tuple[str, float]] = []
    lat_items: list[tuple[str, float]] = []
    for row in rows:
        run = row["run"].strip()
        sr = row.get("proxy_ser_star", "").strip()
        if sr != "":
            ser_items.append((run, float(sr)))
        lat = row["infer_latency_ms"].strip()
        if lat != "":
            lat_items.append((run, float(lat)))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.85), dpi=120)
    ax0, ax1 = axes[0], axes[1]
    labels_s = [t[0] for t in ser_items]
    vals_s = [t[1] for t in ser_items]
    y0 = np.arange(len(vals_s))
    ax0.barh(y0, vals_s, color="#3b6ea5", edgecolor="#222", linewidth=0.45)
    ax0.set_yticks(y0)
    ax0.set_yticklabels(labels_s, fontsize=9)
    ax0.set_xlabel("Proxy SER (hold-out subset, synthetic)")
    ax0.set_title("Section 6.1 — comparative proxy SER cohort")
    ax0.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax0.invert_yaxis()

    labels_l = [t[0] for t in lat_items]
    vals_l = [t[1] for t in lat_items]
    y1 = np.arange(len(vals_l))
    cmap = matplotlib.colormaps["viridis"](np.linspace(0.35, 0.82, len(vals_l)))
    ax1.barh(y1, vals_l, color=cmap, edgecolor="#222", linewidth=0.45)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(labels_l, fontsize=9)
    ax1.set_xlabel("Greedy infer latency (ms mean, synthetic bench)")
    ax1.set_title("Section 6.1 — illustrative latency stripe")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax1.invert_yaxis()
    fig.suptitle("Synthetic aggregate panel — sec6_1_aggregate_metrics.csv", fontsize=10)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_sec6_2(path_out: Path, rows: list[dict[str, str]]) -> None:
    labels = []
    eff, qual, fit = [], [], []
    for row in rows:
        labels.append(row["run"])
        eff.append(float(row["axis_efficiency_pct"]))
        qual.append(float(row["axis_quality_proxy_pct"]))
        fit.append(float(row["axis_resource_fit_pct"]))
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.9, 4.9), dpi=120)
    ax.bar(x - w, eff, width=w, label="Efficiency trajectory", color="#4a86c7")
    ax.bar(x, qual, width=w, label="Quality proxy", color="#2e8b57")
    ax.bar(x + w, fit, width=w, label="Resource fit VRAM-hours", color="#daa520")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Illustrative score 0–100")
    ax.set_title("Multi-axis comparison layer — sec6_2_normalized_axes.csv")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_sec6_3(path_out: Path, ep: np.ndarray, loss: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.05), dpi=120)
    ax.plot(ep, loss, "o-", color="#1f4e79", linewidth=2.1, markersize=7)
    ax.set_xticks(ep)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean CTC train loss (subset registry)")
    ax.set_title("Section 6.3 — linkage T0 loss curve echoed from CSV ctc_train_loss_epochs.csv")
    ax.grid(True, linestyle="--", alpha=0.35)
    for xi, yi in zip(ep, loss):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(5, -10), fontsize=8)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_sec6_4(path_out: Path, rows: list[dict[str, str]]) -> None:
    labs = [r["variant_label"] for r in rows]
    ser = [float(r["synthetic_holdout_proxy_ser"]) for r in rows]
    hrs = [float(r["approx_train_hours_note"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(8.9, 4.66), dpi=120)
    x = np.arange(len(labs))
    ax1.plot(x, ser, "s-", color="#a23434", linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labs, rotation=22, ha="right", fontsize=8)
    ax1.set_ylabel("Synthetic hold-out SER proxy")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(x, hrs, "^--", color="#2f5f2f", linewidth=1.6, markersize=6)
    ax2.set_ylabel("Approx train hours synthetic note")
    fig.suptitle("Section 6.4 — illustrative hyper-plane sweep (CSV)", fontsize=10, y=1.06)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_sec6_5(path_out: Path, rows: list[dict[str, str]]) -> None:
    labs = [r["sample_id"] for r in rows]
    acc = [float(r["char_accuracy_proxy_pct"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.95, 3.92), dpi=120)
    y = np.arange(len(labs))
    ax.barh(y, acc, color="#554488", edgecolor="#111", linewidth=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("Char-level proxy alignment % — synthetic illustrative")
    ax.set_xlim(88, 100)
    ax.set_title("Quality strip for sample anecdotes — Section 6.5")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_sec6_6(path_out: Path, rows: list[dict[str, str]]) -> None:
    lbl = [row["description"][:36] + "…" if len(row["description"]) > 36 else row["description"] for row in rows]
    sizes = [float(row["share_pct"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.95, 5.15), dpi=120)
    ax.pie(
        sizes,
        labels=lbl,
        autopct="%1.1f%%",
        startangle=40,
        textprops={"fontsize": 8},
    )
    ax.set_title("Synthetic error-share mix — Section 6.6")
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def plot_sec6_7(path_out: Path, rows: list[dict[str, str]]) -> None:
    lbl = [row["criterion"].replace("_", " ")[:32] for row in rows]
    w = np.array([float(row["weight_0_to_100"]) for row in rows])
    a = np.array([float(row["achievement_synthetic_pct"]) for row in rows])
    y = np.arange(len(lbl))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.95, 5.52), dpi=120)
    ax.barh(y - h / 2, w, height=h, label="Target emphasis", color="#b0bec5")
    ax.barh(y + h / 2, a, height=h, label="Synthetic fulfilment gap", color="#3949ab")
    ax.set_yticks(y)
    ax.set_yticklabels(lbl, fontsize=8)
    ax.set_xlabel("Score 0–100")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Interpretation checklist — Section 6.7 (sec6_7_interpretation_rubric.csv)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=root / "reports" / "section6_synthetic",
    )
    ap.add_argument(
        "--registry-ctc",
        type=Path,
        default=root / "reports" / "registry_synthetic" / "ctc_train_loss_epochs.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "reports" / "figures",
    )
    ns = ap.parse_args()
    d: Path = ns.data_dir
    out_dir: Path = ns.out
    out_dir.mkdir(parents=True, exist_ok=True)

    r1 = _read_csv_rows(d / "sec6_1_aggregate_metrics.csv")
    r2 = _read_csv_rows(d / "sec6_2_normalized_axes.csv")
    r4 = _read_csv_rows(d / "sec6_4_hypersensitivity_synthetic.csv")
    r5 = _read_csv_rows(d / "sec6_5_qualitative_line_samples.csv")
    r6 = _read_csv_rows(d / "sec6_6_error_taxonomy_counts.csv")
    r7 = _read_csv_rows(d / "sec6_7_interpretation_rubric.csv")

    ep_t0, loss_t0 = load_ctc_t0(ns.registry_ctc)

    plots: list[tuple[str, Any]] = [
        ("fig_6_1_aggregate_ser_latency.png", lambda p: plot_sec6_1(p, r1)),
        ("fig_6_2_multiaxis_comparison.png", lambda p: plot_sec6_2(p, r2)),
        ("fig_6_3_t0_epoch_loss_echo.png", lambda p: plot_sec6_3(p, ep_t0, loss_t0)),
        ("fig_6_4_hyper_sensitivity_tradeoff.png", lambda p: plot_sec6_4(p, r4)),
        ("fig_6_5_sample_char_alignment.png", lambda p: plot_sec6_5(p, r5)),
        ("fig_6_6_error_shares.png", lambda p: plot_sec6_6(p, r6)),
        ("fig_6_7_interpretation_rubric.png", lambda p: plot_sec6_7(p, r7)),
    ]
    for fn, fn_plot in plots:
        p = out_dir / fn
        fn_plot(p)
        print(p)
    print("[plot_section_6_results] Synthetic section-6 artefacts ready.")


if __name__ == "__main__":
    main()
