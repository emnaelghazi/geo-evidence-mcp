"""Build the LinkedIn carousel (PDF, 8 slides of 1080 x 1350).

    uv run --with matplotlib python scripts/make_carousel.py --out docs/img/carousel.pdf

Every figure comes from a recorded run or a test in this repository. Optional: --ndvi PATH uses the
full-resolution NDVI array written by scripts/export_ndvi.py instead of the block-mean map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_status_visual import ndvi_from_ascii

W, H, TOTAL = 1080, 1350, 8
NAVY, NAVY_2, INK, SLATE = "#0B2545", "#13315C", "#1B2B41", "#5C6B7A"
LIGHT, RULE, WHITE = "#F3F5F8", "#D5DBE3", "#FFFFFF"
BLUE, GOLD, GREEN = "#1E5AA8", "#D4A72C", "#1B7F4B"
SERIF_CHOICES = ["Cambria", "Caladea", "Georgia", "DejaVu Serif"]
SANS_CHOICES = ["Calibri", "Carlito", "Segoe UI", "DejaVu Sans"]
BAD_DASHES = ("\u2014", "\u2013")  # em and en dashes are not allowed anywhere on the slides


def pick(choices: list[str]) -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((c for c in choices if c in names), choices[-1])


SERIF, SANS = pick(SERIF_CHOICES), pick(SANS_CHOICES)


class Slide:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=WHITE)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()  # type: ignore[attr-defined]

    # ---- primitives ----------------------------------------------------------------------
    def rect(self, x, y, w, h, fill, edge=None, r=6, lw=1.2, z=1):  # type: ignore[no-untyped-def]
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

    def line(self, x1, y1, x2, y2, color=RULE, lw=1.5, z=2):  # type: ignore[no-untyped-def]
        self.ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="butt", zorder=z)

    def _measure(self, s, size, family, weight, style):  # type: ignore[no-untyped-def]
        t = self.ax.text(
            0, 0, s, fontsize=size * 0.75, family=family, fontweight=weight, fontstyle=style
        )
        width = t.get_window_extent(self.renderer).width
        t.remove()
        return width

    def para(
        self,
        x,
        y,
        s,
        size,
        color=INK,
        family=None,
        weight="normal",
        style="normal",  # type: ignore[no-untyped-def]
        max_w=None,
        lh=1.28,
        ha="left",
        z=5,
    ):
        """Draw wrapped text with its top at y; returns the y of its bottom edge."""
        assert not any(d in s for d in BAD_DASHES), f"dash in slide text: {s!r}"
        family = family or SANS
        lines: list[str] = []
        for chunk in s.split("\n"):
            if not max_w:
                lines.append(chunk)
                continue
            cur = ""
            for word in chunk.split():
                trial = f"{cur} {word}".strip()
                if cur and self._measure(trial, size, family, weight, style) > max_w:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            lines.append(cur)
        for i, text in enumerate(lines):
            self.ax.text(
                x,
                y + i * size * lh,
                text,
                fontsize=size * 0.75,
                color=color,
                family=family,
                fontweight=weight,
                fontstyle=style,
                ha=ha,
                va="top",
                zorder=z,
            )
        return y + len(lines) * size * lh

    def check(self, cx, cy, size=15, color=GREEN):  # type: ignore[no-untyped-def]
        self.ax.add_patch(Circle((cx, cy), size * 1.25, fc=color, ec=color, zorder=4))
        self.ax.plot(
            [cx - size * 0.55, cx - size * 0.1, cx + size * 0.6],
            [cy + size * 0.02, cy + size * 0.5, cy - size * 0.45],
            color=WHITE,
            lw=3.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )

    def number(self, cx, cy, n, r=26, fill=NAVY, size=30):  # type: ignore[no-untyped-def]
        self.ax.add_patch(Circle((cx, cy), r, fc=fill, ec=fill, zorder=4))
        self.ax.text(
            cx,
            cy + 1,
            str(n),
            fontsize=size * 0.75,
            color=WHITE,
            family=SERIF,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )

    def arrow(self, x1, y1, x2, y2, color=SLATE, lw=2.6):  # type: ignore[no-untyped-def]
        self.ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=24,
                lw=lw,
                color=color,
                zorder=4,
                shrinkA=0,
                shrinkB=0,
            )
        )

    # ---- frame ---------------------------------------------------------------------------
    def frame(self, n: int, section: str) -> None:
        self.para(80, 56, "GEO-EVIDENCE-MCP", 20, SLATE, weight="bold")
        self.para(W - 80, 56, f"{n:02d} / {TOTAL:02d}", 20, SLATE, weight="bold", ha="right")
        self.line(80, 100, W - 80, 100)
        self.para(80, 1280, "Emna El Ghazi", 22, INK, weight="bold")
        self.para(W - 80, 1280, "Project review, M0 to M5", 22, SLATE, ha="right")
        self.line(80, 1262, W - 80, 1262)
        self.para(80, 150, section, 24, BLUE, weight="bold")

    def title(self, text: str, y: float = 200, size: int = 70) -> float:
        return self.para(80, y, text, size, NAVY, family=SERIF, weight="bold", lh=1.14, max_w=920)


def slide_cover() -> Slide:
    s = Slide()
    s.rect(0, 0, W, H, NAVY, r=0)
    for radius in (210, 330, 450, 570):  # graticule motif
        s.ax.add_patch(Circle((930, 1060), radius, fc="none", ec=NAVY_2, lw=2, zorder=1))
    s.line(930, 440, 930, 1350, color=NAVY_2, lw=2)
    s.para(80, 108, "PROJECT REVIEW  |  MILESTONES M0 TO M5", 24, GOLD, weight="bold")
    s.line(80, 164, 190, 164, color=GOLD, lw=4)
    s.para(
        80,
        270,
        "Verifiable answers\nfor agentic Earth\nobservation",
        96,
        WHITE,
        family=SERIF,
        weight="bold",
        lh=1.1,
    )
    s.para(
        80,
        700,
        "geo-evidence-mcp: an MCP server and agent where every number carries its evidence "
        "and is checked before you see it.",
        38,
        "#C9D4E3",
        max_w=800,
        lh=1.3,
    )
    s.line(80, 1070, 330, 1070, color=GOLD, lw=2)
    s.para(80, 1100, "Emna El Ghazi", 38, WHITE, weight="bold")
    s.para(80, 1156, "Geoinformatics Engineering  |  Geodata Science and GeoAI", 27, "#AFC0D6")
    s.para(80, 1210, "Open source  |  Python  |  MCP  |  LangGraph  |  PostGIS", 24, "#8FA4C0")
    return s


def slide_problem() -> Slide:
    s = Slide()
    s.frame(2, "01   THE PROBLEM")
    y = s.title("An answer nobody can audit is a liability", size=70)
    s.rect(80, y + 30, 920, 230, LIGHT, r=4)
    s.rect(80, y + 30, 10, 230, BLUE, r=0)
    s.para(
        126,
        y + 60,
        "\u201cNDVI dropped 0.3\nin this zone.\u201d",
        62,
        NAVY,
        family=SERIF,
        style="italic",
        lh=1.18,
    )
    rows = [
        "Which scene and which bands?",
        "Which cloud mask and parameters?",
        "Can anyone reproduce it tomorrow?",
    ]
    yy = y + 330
    for i, text in enumerate(rows, 1):
        s.number(108, yy + 30, i, r=30, size=34)
        s.para(166, yy + 2, text, 42, INK, max_w=840)
        yy += 112
    s.line(80, yy + 16, W - 80, yy + 16)
    s.para(
        80,
        yy + 50,
        "Tool access through MCP already exists.\nEvidence for the answers is the open problem.",
        36,
        NAVY,
        weight="bold",
        max_w=920,
        lh=1.28,
    )
    return s


def slide_architecture() -> Slide:
    s = Slide()
    s.frame(3, "02   ARCHITECTURE")
    y = s.title("Three layers, one rule", size=70)
    s.para(80, y + 8, "No number leaves without its evidence.", 38, SLATE, max_w=900)
    panels = [
        (
            "REASONING",
            "LangGraph agent",
            "Plans the request and writes the answer. Nothing else.",
            LIGHT,
            NAVY,
            SLATE,
            BLUE,
        ),
        (
            "EVIDENCE",
            "MCP tools, provenance, validator",
            "Computes every number, records how, checks every claim.",
            NAVY,
            WHITE,
            "#C9D4E3",
            GOLD,
        ),
        (
            "DATA",
            "Sentinel-2 over STAC and COG",
            "Windowed reads of GeoTIFFs and the cloud classification layer.",
            LIGHT,
            NAVY,
            SLATE,
            BLUE,
        ),
    ]
    top, ph, gap = 440, 214, 62
    for i, (label, name, desc, fill, ink, sub, accent) in enumerate(panels):
        py = top + i * (ph + gap)
        s.rect(80, py, 920, ph, fill, r=4)
        s.rect(80, py, 10, ph, accent, r=0)
        s.para(122, py + 24, label, 24, accent if fill == NAVY else BLUE, weight="bold")
        s.para(122, py + 62, name, 44, ink, family=SERIF, weight="bold", max_w=850, lh=1.1)
        s.para(122, py + 128, desc, 30, sub, max_w=850, lh=1.22)
        if i < 2:
            ay = py + ph
            s.arrow(150, ay + 6, 150, ay + gap - 6)
            s.para(178, ay + 16, "tool calls", 26, SLATE)
            s.arrow(640, ay + gap - 6, 640, ay + 6)
            s.para(668, ay + 16, "validated answer", 26, SLATE)
    return s


def slide_anatomy() -> Slide:
    s = Slide()
    s.frame(4, "03   THE EVIDENCE BLOCK")
    y = s.title("Anatomy of an auditable answer", size=70)
    rows = [
        ("SOURCE", "Scene, asset, band, processing baseline, cloud cover"),
        ("GRID", "CRS, 10 m pixels, window bounds, requested area"),
        ("METHOD", "Formula, clipping rule, cloud mask, resampling"),
        ("OUTPUT", "Every number, typed with its unit"),
        ("INTEGRITY", "SHA-256 of the call, trace id, run id"),
        ("REPLAY", "Re-run the call and diff every number"),
    ]
    top, rh = y + 36, 128
    s.line(80, top, W - 80, top, color=NAVY, lw=3)
    for i, (label, text) in enumerate(rows):
        ry = top + i * rh
        s.rect(80, ry + 48, 16, 16, BLUE, r=2)
        s.para(114, ry + 38, label, 30, NAVY, weight="bold")
        s.para(396, ry + 26, text, 32, INK, max_w=604, lh=1.18)
        s.line(80, ry + rh, W - 80, ry + rh)
    return s


def slide_real_data(ndvi_path: Path | None) -> Slide:
    s = Slide()
    s.frame(5, "04   EVIDENCE ON REAL DATA")
    y = s.title("Tozeur oasis, Sentinel-2", size=68)
    s.para(
        80,
        y + 6,
        "Scene S2C_32SMC_20260904_0_L2A\n4 September 2026, 0.67% cloud cover",
        28,
        SLATE,
        lh=1.25,
    )
    data = np.load(ndvi_path) if ndvi_path and ndvi_path.exists() else ndvi_from_ascii()
    caption = (
        "NDVI at 10 m"
        if ndvi_path and ndvi_path.exists()
        else "NDVI, block means from the live run"
    )
    mx, my, ms = 80, 450, 440
    ax = s.fig.add_axes((mx / W, 1 - (my + ms) / H, ms / W, ms / H))
    ax.imshow(
        np.ma.masked_invalid(data),
        cmap="RdYlGn",
        vmin=-0.1,
        vmax=0.9,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(RULE)
    s.para(mx, my + ms + 10, caption, 22, SLATE)
    gx = s.fig.add_axes((mx / W, 1 - (my + ms + 66) / H, ms / W, 12 / H))
    gx.imshow(np.linspace(-0.1, 0.9, 200)[None, :], cmap="RdYlGn", aspect="auto")
    gx.axis("off")
    s.para(mx, my + ms + 76, "bare, dry", 20, SLATE)
    s.para(mx + ms, my + ms + 76, "dense palms", 20, SLATE, ha="right")
    for i, (big, small) in enumerate(
        [
            ("0.72", "mean NDVI over\nthe 2 by 2 km area"),
            ("77%", "of pixels are\nvegetated"),
            ("11%", "of pixels are\nbare ground"),
        ]
    ):
        yy = 440 + i * 172
        s.para(572, yy, big, 84, NAVY, family=SERIF, weight="bold")
        s.para(572, yy + 94, small, 29, SLATE, lh=1.18)
    s.rect(80, 1040, 920, 190, NAVY, r=4)
    s.para(112, 1066, "6 of 6", 84, GOLD, family=SERIF, weight="bold")
    s.para(
        470, 1070, "pixel values equal an\nindependent read of\nthe raw files", 33, WHITE, lh=1.18
    )
    return s


def slide_findings() -> Slide:
    s = Slide()
    s.frame(6, "05   FINDINGS FROM REAL RUNS")
    y = s.title("What broke, what changed", size=68)
    rows = [
        (
            "REAL SCENE",
            "NDVI ranged from -6.7 to 14.7",
            "Negative reflectance, clipped to zero. The rule is stored in every block.",
        ),
        (
            "REAL MACHINE",
            "A global PROJ_LIB broke CRS lookups",
            "Bundled PROJ data is now preferred, with a regression test.",
        ),
        (
            "TEST SUITE",
            "Cloud pulled the NDVI minimum to 0",
            "A scene classification mask removes it and is part of the evidence.",
        ),
        (
            "TEST SUITE",
            "A refusal dropped its own evidence",
            "Searches and checks now stay on record, even when the answer is no.",
        ),
    ]
    top, rh = y + 24, 232
    for i, (tag, symptom, fix) in enumerate(rows):
        ry = top + i * rh
        s.line(80, ry, W - 80, ry, color=NAVY if i == 0 else RULE, lw=3 if i == 0 else 1.5)
        s.para(80, ry + 28, tag, 24, BLUE, weight="bold")
        s.para(80, ry + 70, symptom, 42, NAVY, family=SERIF, weight="bold", max_w=920, lh=1.1)
        s.para(80, ry + 130, fix, 31, INK, max_w=920, lh=1.2)
    return s


def slide_controls() -> Slide:
    s = Slide()
    s.frame(7, "06   CONTROLS")
    y = s.title("Controls built in", size=70)
    cards = [
        ("21 of 21", "trap questions refused in code, before any model call"),
        ("3 tries", "to fix an unsupported number, then code writes the answer"),
        ("10 m", "pixel size is the limit: finer requests are refused"),
        ("400 km2", "largest area accepted, checked before reading data"),
        ("Replay", "any stored answer is re-run and diffed"),
        ("150+", "automated tests, strict typing, CI on every push"),
    ]
    top, cw, ch, gx_, gy = y + 30, 450, 262, 20, 22
    for i, (big, small) in enumerate(cards):
        cx = 80 + (i % 2) * (cw + gx_)
        cy = top + (i // 2) * (ch + gy)
        s.rect(cx, cy, cw, ch, LIGHT, r=4)
        s.rect(cx, cy, 8, ch, BLUE, r=0)
        s.para(cx + 34, cy + 24, big, 62, NAVY, family=SERIF, weight="bold")
        s.para(cx + 34, cy + 112, small, 29, INK, max_w=cw - 62, lh=1.2)
    return s


def slide_next() -> Slide:
    s = Slide()
    s.frame(8, "07   STATUS AND NEXT STEPS")
    y = s.title("Verified, and what is next", size=68)
    top = y + 20
    s.para(80, top, "VERIFIED", 26, GREEN, weight="bold")
    for i, t in enumerate(
        ["Pixels on a real scene", "Live catalog and tools", "150+ offline tests"]
    ):
        s.check(98, top + 76 + i * 74, 12)
        s.para(138, top + 52 + i * 74, t, 32, INK, max_w=380)
    s.para(560, top, "IN PROGRESS", 26, GOLD, weight="bold")
    for i, t in enumerate(["Docker build in CI", "Real model runs", "Langfuse server traces"]):
        s.ax.add_patch(Circle((578, top + 76 + i * 74), 13, fc="none", ec=GOLD, lw=3.4, zorder=4))
        s.para(616, top + 52 + i * 74, t, 32, INK, max_w=384)
    ny = top + 76 + 3 * 74 + 4
    s.line(80, ny, W - 80, ny, color=NAVY, lw=3)
    nxt = [
        ("M6", "Benchmark: 100 questions, 20+ traps"),
        ("M7", "Naive vs evidence enforcing agent"),
        ("M8", "One command stack, version 0.1.0"),
    ]
    for i, (code, text) in enumerate(nxt):
        yy = ny + 26 + i * 74
        s.para(80, yy, code, 36, BLUE, family=SERIF, weight="bold")
        s.para(176, yy + 3, text, 32, INK, max_w=824)
    qy = ny + 26 + 3 * 74 + 10
    s.rect(80, qy, 920, 1236 - qy, NAVY, r=4)
    s.para(
        116,
        qy + 26,
        "For teams running EO agents in production: what do you log today to prove an answer was right?",
        40,
        WHITE,
        family=SERIF,
        weight="bold",
        max_w=850,
        lh=1.2,
    )
    s.para(116, 1236 - 58, "github.com/emnaelghazi/geo-evidence-mcp", 26, GOLD, weight="bold")
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/img/carousel.pdf"))
    parser.add_argument("--ndvi", type=Path, default=None)
    args = parser.parse_args()
    plt.rcParams["pdf.fonttype"] = 42
    slides = [
        slide_cover(),
        slide_problem(),
        slide_architecture(),
        slide_anatomy(),
        slide_real_data(args.ndvi),
        slide_findings(),
        slide_controls(),
        slide_next(),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        args.out,
        metadata={
            "Title": "geo-evidence-mcp: verifiable answers for agentic EO",
            "Author": "Emna El Ghazi",
        },
    ) as pdf:
        for slide in slides:
            pdf.savefig(slide.fig, facecolor=WHITE)
    print(f"wrote {args.out} ({len(slides)} slides, fonts: {SERIF} / {SANS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
