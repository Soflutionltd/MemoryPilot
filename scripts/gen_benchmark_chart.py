"""Generate the MemoryPilot vs competitors benchmark chart.

Produces a horizontal bar chart comparing R@5 / accuracy on LongMemEval-S
across the leading LLM memory layers as of v4.2.

Sources (all public, as of May 2026):
- MemoryPilot v4.2: this repo, --benchmark-longmemeval @470, May 2026
- MemPalace v3.3.3: MemPalace LongMemEval public release notes
- Mem0: https://mem0.ai/blog/benchmarked-openai-memory-... (94.4% accuracy)
- mcp-memory-service v10.34.0: release notes (R@5 80.4%, zero-LLM ONNX)
- Zep / Graphiti: arxiv 2501.13956 (63.8% LongMemEval accuracy)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

OUTPUT_PNG = "static/benchmark_chart.png"
OUTPUT_SVG = "static/benchmark_chart.svg"

# (label, score, color, is_us)
DATA = [
    ("MemoryPilot v4.2 (adaptive)",      99.1, "#84cc16", True),   # lime-500
    ("MemoryPilot v4.2 (fast, ~28 ms)",  98.7, "#a3e635", True),   # lime-400
    ("MemPalace v3.3.3 (hybrid)",        98.4, "#475569", False),  # slate-600
    ("Mem0 (cloud, OpenAI)",             94.4, "#64748b", False),  # slate-500
    ("mcp-memory-service v10.34",        80.4, "#94a3b8", False),  # slate-400
    ("Zep / Graphiti",                   63.8, "#cbd5e1", False),  # slate-300
]

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", "Arial",
]
rcParams["axes.edgecolor"] = "#0f172a"
rcParams["axes.linewidth"] = 0.0
rcParams["text.color"] = "#0f172a"

fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=200)
fig.patch.set_facecolor("#fafaf9")  # off-white
ax.set_facecolor("#fafaf9")

labels = [d[0] for d in DATA]
scores = [d[1] for d in DATA]
colors = [d[2] for d in DATA]
y_pos = list(range(len(DATA)))[::-1]

bars = ax.barh(
    y_pos,
    scores,
    color=colors,
    edgecolor="none",
    height=0.66,
    zorder=3,
)

for bar, (label, score, color, is_us) in zip(bars, DATA):
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    weight = "bold" if is_us else "semibold"
    ax.text(
        width + 0.7, y, f"{score:.1f}%",
        va="center", ha="left",
        fontsize=12.5,
        fontweight=weight,
        color="#0f172a" if is_us else "#334155",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(
    labels,
    fontsize=12,
    color="#0f172a",
)
for tick_label, (_, _, _, is_us) in zip(ax.get_yticklabels(), DATA):
    tick_label.set_fontweight("bold" if is_us else "normal")
    tick_label.set_color("#0f172a" if is_us else "#475569")

ax.set_xlim(0, 110)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color="#64748b")
ax.tick_params(axis="x", length=0, pad=8)
ax.tick_params(axis="y", length=0, pad=8)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=1)
ax.set_axisbelow(True)

fig.suptitle(
    "LongMemEval-S — R@5 / accuracy across LLM memory layers",
    fontsize=16,
    fontweight="bold",
    color="#0f172a",
    x=0.02, y=0.96,
    ha="left",
)
fig.text(
    0.02, 0.90,
    "Higher is better · 470 evaluable questions · all numbers from public sources (May 2026)",
    fontsize=10,
    color="#64748b",
)

fig.text(
    0.01, 0.02,
    "Sources: MemoryPilot --benchmark-longmemeval @470 · MemPalace v3.3.3 notes · "
    "mem0.ai blog · mcp-memory-service v10.34.0 · Zep arXiv 2501.13956",
    fontsize=8,
    color="#94a3b8",
)

plt.tight_layout(rect=[0, 0.05, 1, 0.85])

os.makedirs("static", exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Wrote {OUTPUT_PNG} and {OUTPUT_SVG}")
