"""Generate evaluation charts (RAGAS radar + latency bars) from eval results.

Reads:
  data/eval_results/comparison.json   (RAGAS 三维度 averages)
  data/eval_results/latency.json      (latency raw + mean_trimmed + p95)

Writes:
  docs/charts/v1-v6-radar.png
  docs/charts/v1-v7-latency.png

Re-run after re-running run_ragas_eval.py / measure_latency.py to refresh charts.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval_results"
OUT_DIR = ROOT / "docs" / "charts"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "V1": "#888888",
    "V2": "#1f77b4",
    "V3": "#bbbbbb",
    "V4": "#ff7f0e",
    "V5": "#2ca02c",
    "V6": "#999999",
}


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
    maxima = {metric:max((s['avg'][metric] for s in summaries),default=None) for metric in metrics}
    metric_names = ['Relevancy', 'Faithfulness', 'Precision']

    for s in summaries:
        v = s["version"]
        values = [s["avg"][m] for m in metrics]
        values += values[:1]
        c = COLORS.get(v, "gray")
        best = [name for metric,name in zip(metrics,metric_names) if s['avg'][metric]==maxima[metric]]
        is_winner = bool(best)
        lw = 3.0 if is_winner else 1.2
        alpha = 0.95 if is_winner else 0.45
        ls = "-" if is_winner else "--"
        label = v + (' (' + '/'.join(best) + ' max)' if best else '')
        ax.plot(angles, values, color=c, linestyle=ls, linewidth=lw, alpha=alpha, label=label)
        if is_winner:
            ax.fill(angles, values, color=c, alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5)
    ax.grid(True, alpha=0.35)

    ax.set_title(
        f"RAGAS 三维度对比 ({len(summaries)} 个版本)\n"
        + '题数：' + ' · '.join(f"{s['version']}={len(s['records']) if 'records' in s else '未知'}" for s in summaries)
        + '\nmax 表示本次输入的最高均值（含并列）',
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_latency(latency_results):
    versions = [r["version"] for r in latency_results]
    means = [r["mean_trimmed"] if r["mean_trimmed"] is not None else np.nan for r in latency_results]
    p95s = [r["p95"] if r["p95"] is not None else np.nan for r in latency_results]

    x = np.arange(len(versions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.2))

    bars_mean = ax.bar(
        x - width / 2, means, width, label="Mean (trimmed)", color="#1f77b4", alpha=0.85
    )
    bars_p95 = ax.bar(x + width / 2, p95s, width, label="p95", color="#ff7f0e", alpha=0.85)

    for bars in (bars_mean, bars_p95):
        for b in bars:
            h = b.get_height()
            if not np.isfinite(h):
                continue
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
    ax.set_xticklabels([f"{r['version']}\n{r.get('succeeded',r.get('n','?'))}/{r.get('attempted',r.get('n','?'))} 成功"
                        for r in latency_results], fontsize=11)
    for i,mean in enumerate(means):
        if not np.isfinite(mean):
            ax.text(i, 0, '无可用耗时', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("延迟 / Latency (秒)", fontsize=11)
    ax.set_title(
        f"延迟对比 ({len(versions)} 个版本)\n耗时统计仅包含成功请求；横轴显示成功数/尝试数",
        fontsize=11.5,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    out = OUT_DIR / "v1-v7-latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
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
