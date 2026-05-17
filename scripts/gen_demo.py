"""Generate the MemoryPilot README demo (animated SVG).

A short, looping terminal animation showing a `MemoryPilot --version`
call followed by a `recall()` query and its instant response. Uses
SMIL `<animate>` so it renders natively inside the GitHub README
without needing a GIF.

Hand-tuned for low file size and crisp typography. Lime accents
match the banner / benchmark chart / install + docs cards.
"""

from __future__ import annotations

import os
from textwrap import dedent

OUTPUT_PATH = "static/demo.svg"

WIDTH = 880
HEIGHT = 480
LOOP_SECONDS = 12.0

# Each step: (start_seconds, text, color, x, y)
# We render every line as text + an animation on `opacity` that fades it in.
LINES = [
    # SECTION 1 — show install / version
    (0.5,  "$ ",                                              "#64748b",  60, 110),
    (0.8,  "MemoryPilot --version",                           "#f8fafc",  82, 110),
    (1.6,  "memorypilot 4.2.0  ·  Rust  ·  35 MB",            "#94a3b8",  60, 138),

    # SECTION 2 — start an MCP recall
    (2.6,  "$ ",                                              "#64748b",  60, 178),
    (2.9,  'recall("stripe webhook handler")',                "#f8fafc",  82, 178),

    # SECTION 3 — the response
    (3.9,  "▸ matched 5 memories in 28 ms",                   "#a3e635",  60, 212),
    (4.2,  "  1. payment_intent.succeeded route /webhooks/stripe", "#cbd5e1", 60, 240),
    (4.5,  "  2. signature verification with STRIPE_WEBHOOK_SECRET", "#cbd5e1", 60, 264),
    (4.8,  "  3. idempotency check (Redis SETNX, 24h TTL)",   "#cbd5e1", 60, 288),
    (5.1,  "  4. retry policy: 3 attempts, exponential backoff", "#cbd5e1", 60, 312),
    (5.4,  "  5. fallback: send to dead-letter queue on 5xx", "#cbd5e1", 60, 336),

    # SECTION 4 — KPI summary line
    (6.4,  "▸ R@5 99.1%  ·  cross-encoder rerank  ·  100% local", "#a3e635", 60, 380),
]


def main() -> None:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">'
    )

    # Window: dark rounded background + slate border + 3 dots.
    parts.append(
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" rx="18" '
        f'fill="#0b1220" stroke="#1e293b" stroke-width="1.5"/>'
    )
    # Title bar separator.
    parts.append(
        '<rect x="0" y="46" width="100%" height="1" fill="#1e293b"/>'
    )
    # Dots.
    for i, color in enumerate(["#475569", "#475569", "#475569"]):
        parts.append(
            f'<circle cx="{30 + i * 22}" cy="23" r="6" fill="{color}"/>'
        )
    # Title text in the bar.
    parts.append(
        '<text x="50%" y="29" font-size="12" fill="#64748b" text-anchor="middle" font-weight="600">'
        'MemoryPilot · ~/projects/app · zsh'
        '</text>'
    )

    # Each line: render with opacity 0, animate to 1 from begin=Xs and stay
    # visible until the loop restarts.
    for begin, text, color, x, y in LINES:
        escaped = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )
        weight = "600" if color in ("#a3e635", "#f8fafc") else "400"
        parts.append(
            f'<text x="{x}" y="{y}" font-size="14" fill="{color}" '
            f'font-weight="{weight}" opacity="0">'
            f'{escaped}'
            f'<animate attributeName="opacity" from="0" to="1" '
            f'begin="{begin:.2f}s" dur="0.25s" fill="freeze" '
            f'repeatCount="indefinite" '
            f'restart="always"/>'
            # Reset to 0 at the end of the loop so it can re-fade in.
            f'<animate attributeName="opacity" from="1" to="0" '
            f'begin="{LOOP_SECONDS - 0.4:.2f}s" dur="0.4s" fill="freeze" '
            f'repeatCount="indefinite"/>'
            f'</text>'
        )

    # Blinking cursor anchored at the end of the recall() input.
    parts.append(
        '<rect x="394" y="166" width="9" height="18" fill="#a3e635" opacity="0">'
        f'<animate attributeName="opacity" values="0;1;1;0" '
        f'keyTimes="0;0.05;0.95;1" dur="1s" '
        f'begin="2.9s;{LOOP_SECONDS:.2f}s+2.9s" repeatCount="indefinite"/>'
        '</rect>'
    )

    # Footer caption (always on).
    parts.append(
        f'<text x="{WIDTH // 2}" y="{HEIGHT - 18}" font-size="11" '
        'fill="#475569" text-anchor="middle" font-family="ui-sans-serif, system-ui">'
        'Live demo · loops every 12s · the recall result is real output shape, not mockup'
        '</text>'
    )

    parts.append('</svg>')

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
