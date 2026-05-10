"""Generate evaluation charts (RAGAS radar + latency bars) from eval results.

Reads:
  data/eval_results/comparison.json   (RAGAS 三维度 averages, 30 题/版)
  data/eval_results/latency.json      (latency raw + mean_trimmed + p95, 10 题/版)

Writes:
  docs/charts/v1-v6-radar.png
  docs/charts/v1-v6-latency.png

Re-run after re-running run_ragas_eval.py / measure_latency.py to refresh charts.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval_results"
OUT_DIR = ROOT / "docs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Highlight winners: V2 (relevancy) / V4 (faithfulness) / V5 (precision)
COLORS = {
    "V1": "#888888",
    "V2": "#1f77b4",
    "V3": "#bbbbbb",
    "V4": "#ff7f0e",
    "V5": "#2ca02c",
    "V6": "#999999",
}
WINNERS = {"V2", "V4", "V5"}


def load_summaries():
    with open(EVAL_DIR / "comparison.json", encoding="utf-8") as f:
        return json.load(f)["summaries"]


def load_latency():
    with open(EVAL_DIR / "latency.json", encoding="utf-8") as f:
        return json.load(f)["results"]


def make_radar(summaries):
    metrics = ["answer_relevancy", "faithfulness", "context_precision"]
    labels = [
        "Answer Relevancy\n(切题度)",
        "Faithfulness\n(无幻觉)",
        "Context Precision\n(检索精度)",
    ]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="polar")

    for s in summaries:
        v = s["version"]
        values = [s["avg"][m] for m in metrics]
        values += values[:1]
        c = COLORS.get(v, "gray")
        is_winner = v in WINNERS
        lw = 3.0 if is_winner else 1.2
        alpha = 0.95 if is_winner else 0.45
        ls = "-" if is_winner else "--"
        winner_tag = {
            "V2": "V2 (Relevancy winner)",
            "V4": "V4 (Faithfulness winner)",
            "V5": "V5 (Precision winner)",
        }
        label = winner_tag.get(v, v)
        ax.plot(angles, values, color=c, linestyle=ls, linewidth=lw, alpha=alpha, label=label)
        if is_winner:
            ax.fill(angles, values, color=c, alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.45, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=8.5)
    ax.grid(True, alpha=0.35)

    ax.set_title(
        "RAGAS 三维度对比 (V1–V6, 30 题/版)\n"
        "三维赢家分散在三个版本——没有 \"全场最优\" 的配置",
        fontsize=12.5,
        pad=24,
        fontweight="bold",
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.18, 0.5),
        fontsize=10,
        framealpha=0.9,
        title="Version",
        title_fontsize=10.5,
    )

    out = OUT_DIR / "v1-v6-radar.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_latency(latency_results):
    versions = [r["version"] for r in latency_results]
    means = [r["mean_trimmed"] for r in latency_results]
    p95s = [r["p95"] for r in latency_results]

    x = np.arange(len(versions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.2))

    bars_mean = ax.bar(
        x - width / 2, means, width, label="Mean (trimmed)", color="#1f77b4", alpha=0.85
    )
    bars_p95 = ax.bar(x + width / 2, p95s, width, label="p95", color="#ff7f0e", alpha=0.85)

    for i, v in enumerate(versions):
        if v == "V2":
            for b in (bars_mean[i], bars_p95[i]):
                b.set_edgecolor("#2ca02c")
                b.set_linewidth(2.5)
        elif v == "V4":
            for b in (bars_mean[i], bars_p95[i]):
                b.set_edgecolor("#d62728")
                b.set_linewidth(2.5)

    for bars in (bars_mean, bars_p95):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=11)
    ax.set_ylabel("延迟 / Latency (秒)", fontsize=11)
    ax.set_title(
        "V1–V6 延迟对比 (10 题/版, n=10)\n"
        "绿框 = 业务推荐 V2 (mean 8.7s) · 红框 = 全开 V4 (mean 24.2s ≈ 3× V2)",
        fontsize=11.5,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    out = OUT_DIR / "v1-v6-latency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    summaries = load_summaries()
    latency_results = load_latency()
    radar_path = make_radar(summaries)
    latency_path = make_latency(latency_results)
    print(f"Generated:\n  {radar_path}\n  {latency_path}")


if __name__ == "__main__":
    main()
