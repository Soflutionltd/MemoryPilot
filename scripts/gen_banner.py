"""Generate the MemoryPilot README banner.

Wide premium hero (1600x500) — lime-on-dark, GitHub header friendly.
Linear / Vercel-style pure typographic hero. No blue-purple gradient,
no centered logo, no shadow-lg rounded-lg, no AI slop.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib import patches, rcParams

OUTPUT_PNG = "static/banner.png"
OUTPUT_SVG = "static/banner.svg"

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", "Arial",
]
rcParams["text.color"] = "#f8fafc"

WIDTH_IN = 16.0
HEIGHT_IN = 5.0
DPI = 200

fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
fig.patch.set_facecolor("#020617")
ax.set_facecolor("#020617")
ax.set_xlim(0, 1600)
ax.set_ylim(0, 500)
ax.set_axis_off()

# Subtle dark grid for depth — no blobs, no gradients.
for i in range(0, 1600, 80):
    ax.add_line(plt.Line2D(
        [i, i], [0, 500],
        linewidth=0.4, color="#0f172a", alpha=0.7, zorder=1,
    ))
for j in range(0, 500, 80):
    ax.add_line(plt.Line2D(
        [0, 1600], [j, j],
        linewidth=0.4, color="#0f172a", alpha=0.7, zorder=1,
    ))

# Single lime accent bar on the far left — the only "decoration".
ax.add_patch(patches.Rectangle(
    (80, 120), 6, 260,
    linewidth=0, facecolor="#a3e635", zorder=3,
))

# Eyebrow.
ax.text(
    120, 380,
    "MEMORYPILOT  ·  v4.2",
    fontsize=14,
    color="#a3e635",
    fontweight="bold",
    family="monospace",
    zorder=4,
)

# Main title — two lines, oversized, tight, asymmetric.
ax.text(
    120, 305,
    "The fastest local memory",
    fontsize=58,
    color="#f8fafc",
    fontweight="bold",
    zorder=4,
)
ax.text(
    120, 225,
    "layer for AI agents.",
    fontsize=58,
    color="#a3e635",
    fontweight="bold",
    zorder=4,
)

# Sub-tagline — muted, short, factual.
ax.text(
    120, 160,
    "Hybrid retrieval  ·  99.1% R@5 on LongMemEval-S  ·  35 MB Rust binary  ·  zero API calls",
    fontsize=16,
    color="#94a3b8",
    zorder=4,
)

# KPI row, lime numbers + slate labels.
stats = [
    ("99.1%",  "R@5 LongMemEval"),
    ("~28 ms", "p50 search"),
    ("100+",   "languages"),
    ("35 MB",  "single binary"),
    ("0",      "API calls"),
]
x_cursor = 120
gap = 250
for value, label in stats:
    ax.text(
        x_cursor, 95,
        value,
        fontsize=22,
        color="#f8fafc",
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x_cursor, 65,
        label,
        fontsize=11,
        color="#64748b",
        zorder=4,
    )
    x_cursor += gap

os.makedirs("static", exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0,
            facecolor=fig.get_facecolor())
plt.savefig(OUTPUT_SVG, bbox_inches="tight", pad_inches=0,
            facecolor=fig.get_facecolor())
print(f"Wrote {OUTPUT_PNG} and {OUTPUT_SVG}")
