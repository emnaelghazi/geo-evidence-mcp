"""Render the project status card (1080x1350 PNG, LinkedIn portrait).

    uv run --with matplotlib python scripts/make_status_visual.py --out docs/img/status-m5.png

Every number on the card comes from a real run recorded in this repository's history:
  * the NDVI map is the live Tozeur run (scene S2C_32SMC_20260904_0_L2A, coarse block means);
  * pass --ndvi docs/evidence/ndvi_tozeur_2026-09-04.npy to use the full-resolution array instead
    (create it with `uv run python scripts/export_ndvi.py`).
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

W, H = 1080, 1350
INK, MUTED, LINE, PAPER = "#0F172A", "#475569", "#E2E8F0", "#F8FAFC"
NAVY = "#0B1B2B"
TEAL, TEAL_BG, TEAL_LINE = "#0F766E", "#E6F7F4", "#5EEAD4"
PURPLE, PURPLE_BG = "#6D28D9", "#F3EFFE"
AMBER, AMBER_BG = "#B45309", "#FEF3C7"

# Live run output (coarse block-mean NDVI map, north up) from scene S2C_32SMC_20260904_0_L2A.
RAMP = " .:-=+*#%@"  # NDVI <= -0.1 ... >= 0.9
ASCII_MAP = """\
@@@@@@@@@@@%%%@@@@@%%@@@@%@@@@%**#@*+=+#+%@@@@@@@@@@@@#=
@@@@@@@@@@@%%@#*@%*%@@@@@@@@@@#@@@@@@@@@@@@@@@%#%%@@#+=-
@@*%@@@@@@%@@@@@@%#*#*##+@@@@##@@@@@@@@@@%%%%#@@@@@@@@@@
@##@@@#*%%@@@@@@%@%==*=:::-%@@@%@@@%+**%++#@@@@@@@@@@@@@
@@#%@@@@@@@@@@@@@@@@%-.:%+--+*##@@@@*=-=+*#@@@@@@@@@@@@@
@@@@%@@@@@@@@@@%@@@@@@%%*@%++-#%%#@@@@@@%#%@@@@@+=====+@
%*=+*@@@@@@@@%*#@@@@@@@@#:*#-=%@@@@@@@@@@@@@@@@@@@@@#**%
@#=***@@@@@@@@%%@@@@@@@#+--#=-+*%%@@@@@@++==#@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@#*#+@@#%@@@%%@@@@%##@@@@@@@@@@@@@
@@@@@@%%@@@@#%%%@@@@@@@%#%#@@@%%@@@@@@@@@@@@@@%#%%#@@@@@
@@*%@@@*@@@@@@@@@@@%#%@%%%#@@%%%##@@@@@@@@@@@@@%*%@@@@@@
*+++%@@@#*@%%@@@@@##@@@@%###@@@@@@@@@@@@@**%@@%%@@@@@@@@
#+++@@@%@#%@@@@@@@%@@@@@@#@#%@@@@@@%@@@@@@%@@%%@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@%%%@@@@@#+%@@@@#*--::--==++=::
@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@@@@@@@@%*=---:---:::::::.
%@@@@%@@@*#%@@@@%#%%@@@@@@%+*@@%%%%@@%%%=::::-+===---:..
@%:=@@@*+#@@#%*==+#@@@@@@@@@@@@%@@%@@@@@:.:::-----:-:-::
@@%-+@@@@@@@%*###**%@@@@@@%#@@@@@@%+---=:::-:::-:::::::.
@%@@*@@%@@@@@=%@@#@@%@%#@%*#@@@@@#**+---::::---::::::.::
@@@@@*--*@@@@@%+:-@@@#=+#%#@#-=-===---------:::....:::::
#%@@@@%==#@@@@@%::=---:::---::.............:::.........:
*+=%@@@@@@%%*=+=:...::-:::::.........::.................
"""


def ndvi_from_ascii() -> np.ndarray:
    rows = ASCII_MAP.splitlines()
    assert len({len(r) for r in rows}) == 1, "ragged map"
    return np.array([[(RAMP.index(c) + 0.5) / 10 - 0.1 for c in row] for row in rows])


def pick_font() -> str:
    for path in font_manager.findSystemFonts():
        if "/inter/" in path.lower().replace("\\", "/"):
            font_manager.fontManager.addfont(path)
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Inter" if "Inter" in names else "DejaVu Sans"


class Canvas:
    def __init__(self, family: str) -> None:
        self.family = family
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")

    def rect(self, x, y, w, h, fill, edge=None, r=14, lw=1.4, z=1):  # type: ignore[no-untyped-def]
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                fc=fill,
                ec=edge or fill,
                lw=lw,
                zorder=z,
            )
        )

    def text(
        self,
        x,
        y,
        s,
        size=18,
        color=INK,
        weight="normal",
        ha="left",
        va="center",  # type: ignore[no-untyped-def]
        max_w=None,
        z=5,
        style="normal",
        spacing=1.25,
    ):
        t = self.ax.text(
            x,
            y,
            s,
            fontsize=size * 0.75,
            color=color,
            fontweight=weight,
            ha=ha,
            va=va,
            family=self.family,
            zorder=z,
            fontstyle=style,
            linespacing=spacing,
        )
        if max_w:
            self.fig.canvas.draw()
            while t.get_window_extent().width > max_w and size > 8:
                size -= 0.5
                t.set_fontsize(size * 0.75)
                self.fig.canvas.draw()
        return t

    def arrow(self, x1, y1, x2, y2, color=MUTED):  # type: ignore[no-untyped-def]
        self.ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=16,
                lw=2,
                color=color,
                zorder=4,
                shrinkA=0,
                shrinkB=0,
            )
        )


def wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


def build(out: Path, ndvi_path: Path | None) -> None:
    c = Canvas(pick_font())

    # ---- header (0-205) ----------------------------------------------------------------
    c.rect(0, 0, W, 205, NAVY, r=0)
    c.text(60, 38, "OPEN SOURCE  \u00b7  MILESTONES M0 TO M5 COMPLETE", 15, TEAL_LINE, "bold")
    c.text(
        60, 108, "A trust layer for agentic\nEarth observation", 44, "white", "bold", spacing=1.05
    )
    c.text(
        60,
        180,
        "Every number an AI agent reports carries its evidence, and is checked before you see it.",
        18.5,
        "#CBD5E1",
        max_w=960,
    )

    # ---- the failure (226-306) ---------------------------------------------------------
    c.rect(60, 226, 960, 80, "#FEF2F2", "#FECACA")
    c.rect(60, 226, 8, 80, "#DC2626", r=0)
    c.text(92, 252, "\u201cNDVI dropped 0.3 in this zone.\u201d", 23, "#991B1B", "bold", max_w=900)
    c.text(
        92,
        285,
        "Which scene? Which band? Which tool call? Today, nobody can check.",
        17.5,
        MUTED,
        max_w=900,
    )

    # ---- pipeline (330-470) ------------------------------------------------------------
    steps = [
        ("Question", "#FFFFFF", LINE, MUTED),
        ("LangGraph\nagent", PURPLE_BG, "#C4B5FD", PURPLE),
        ("MCP tools\nSTAC + COG", TEAL_BG, TEAL_LINE, TEAL),
        ("Provenance\nblock", TEAL_BG, TEAL_LINE, TEAL),
        ("Claim\nvalidator", TEAL_BG, TEAL_LINE, TEAL),
        ("Cited\nanswer", "#FFFFFF", LINE, MUTED),
    ]
    bw, gap, y0, bh = 139, 25.2, 330, 88
    for i, (label, fill, edge, ink) in enumerate(steps):
        x = 60 + i * (bw + gap)
        c.rect(x, y0, bw, bh, fill, edge, lw=2)
        c.text(x + bw / 2, y0 + bh / 2, label, 17.5, ink, "bold", ha="center", max_w=bw - 16)
        if i < len(steps) - 1:
            c.arrow(x + bw + 4, y0 + bh / 2, x + bw + gap - 4, y0 + bh / 2)
    x_teal = 60 + 2 * (bw + gap)
    x_end = 60 + 4 * (bw + gap) + bw
    c.ax.plot(
        [x_teal, x_end], [y0 + bh + 16] * 2, color=TEAL, lw=3, solid_capstyle="round", zorder=4
    )
    c.text(
        (x_teal + x_end) / 2,
        y0 + bh + 40,
        "trust layer: scene, bands, CRS, parameters, hash, trace. Stored in PostGIS, replayable.",
        15.5,
        TEAL,
        "bold",
        ha="center",
        max_w=600,
    )

    # ---- real data (500-900) -----------------------------------------------------------
    top = 506
    c.text(60, top, "Real data, not a demo", 25, INK, "bold")
    c.text(
        60,
        top + 30,
        "Tozeur oasis, Tunisia  \u00b7  Sentinel-2 L2A  \u00b7  verified against raw pixel values",
        16.5,
        MUTED,
        max_w=960,
    )
    card_y, card_h = top + 54, 332
    c.rect(60, card_y, 960, card_h, PAPER, LINE, r=18)

    if ndvi_path and ndvi_path.exists():
        data = np.load(ndvi_path)
        caption = "NDVI at 10 m, full resolution"
    else:
        data = ndvi_from_ascii()
        caption = "NDVI, block means from the live run"
    mx, my, ms = 84, card_y + 20, 236
    map_ax = c.fig.add_axes((mx / W, 1 - (my + ms) / H, ms / W, ms / H))
    map_ax.imshow(
        np.ma.masked_invalid(data),
        cmap="RdYlGn",
        vmin=-0.1,
        vmax=0.9,
        interpolation="nearest",
        aspect="auto",
    )
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    for spine in map_ax.spines.values():
        spine.set_edgecolor(LINE)
    c.text(mx, my + ms + 16, caption, 13, MUTED, max_w=ms)
    leg_y = my + ms + 32
    grad = np.linspace(-0.1, 0.9, 200)[None, :]
    leg_ax = c.fig.add_axes((mx / W, 1 - (leg_y + 9) / H, ms / W, 9 / H))
    leg_ax.imshow(grad, cmap="RdYlGn", vmin=-0.1, vmax=0.9, aspect="auto")
    leg_ax.axis("off")
    c.text(mx, leg_y + 24, "bare / dry", 12, MUTED)
    c.text(mx + ms, leg_y + 24, "dense palms", 12, MUTED, ha="right")

    cx = 380
    chips = [
        ("0.72", "mean NDVI over the 2 \u00d7 2 km area"),
        ("0.89", "median: most of the oasis is dense canopy"),
        ("77%", "of pixels vegetated, 11% bare ground"),
        ("0.67%", "cloud cover on the scene date, 2026-09-04"),
    ]
    for i, (big, small) in enumerate(chips):
        yy = card_y + 46 + i * 62
        c.text(cx, yy, big, 32, TEAL if i < 3 else PURPLE, "bold", max_w=140)
        c.text(cx + 150, yy, small, 16.5, MUTED, max_w=470)
    c.text(
        cx,
        card_y + card_h - 26,
        "S2C_32SMC_20260904_0_L2A  \u00b7  206 \u00d7 202 px at 10 m  \u00b7  EPSG:32632",
        13,
        MUTED,
        max_w=620,
    )

    # ---- proof tiles (930-1160) --------------------------------------------------------
    py = card_y + card_h + 42
    c.text(60, py, "Verified, not assumed", 25, INK, "bold")
    tiles = [
        ("6 / 6", "pixel values match an independent GDAL read, exactly"),
        (
            "\u22126.7 to 14.7",
            "NDVI range until negative reflectance (38% of pixels) was handled; now within \u00b11",
        ),
        ("21 / 21", "trap questions refused in code, before any model call"),
        ("150+", "automated tests; every stored answer can be replayed and diffed"),
    ]
    tw, th = 468, 96
    for i, (big, small) in enumerate(tiles):
        x = 60 + (i % 2) * (tw + 24)
        y = py + 26 + (i // 2) * (th + 14)
        c.rect(x, y, tw, th, "white", LINE, lw=1.6)
        c.rect(x, y, 7, th, TEAL, r=0)
        c.text(x + 26, y + 28, big, 25, TEAL, "bold", max_w=tw - 44)
        c.text(x + 26, y + 64, wrap(small, 54), 14, MUTED, max_w=tw - 44, spacing=1.3)

    # ---- next --------------------------------------------------------------------------
    ny = py + 26 + 2 * (th + 14) + 10
    c.rect(60, ny, 960, 98, AMBER_BG, "#FDE68A", r=18)
    c.text(86, ny + 24, "UP NEXT", 14.5, AMBER, "bold")
    nxt = [
        "100-question benchmark, 20+ traps",
        "naive vs provenance-enforcing agent",
        "Langfuse traces on real model runs",
        "one-command Docker stack, v0.1.0",
    ]
    for i, item in enumerate(nxt):
        c.text(86 + (i % 2) * 470, ny + 52 + (i // 2) * 28, "\u2192  " + item, 16, INK, max_w=440)

    # ---- footer ------------------------------------------------------------------------
    c.text(
        60,
        H - 30,
        "FastMCP  \u00b7  LangGraph  \u00b7  STAC / COG  \u00b7  PostGIS + pgvector  \u00b7  Langfuse  \u00b7  Python",
        14.5,
        MUTED,
        max_w=960,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    c.fig.savefig(out, dpi=100, facecolor="white")
    print(f"layout bottom of the 'next' box: {ny + 98:.0f} / {H}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/img/status-m5.png"))
    parser.add_argument("--ndvi", type=Path, default=None, help=".npy NDVI array (NaN = masked)")
    args = parser.parse_args()
    build(args.out, args.ndvi)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
