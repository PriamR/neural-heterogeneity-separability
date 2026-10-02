"""A light schematic of the task: the Spiking Heidelberg Digits as spike trains, twenty classes (ten digits in
two languages), collapsed to four classes by language and parity. Drawn, not measured, so it takes no data;
the raster is illustrative and the caption says so.

Writes docs/analysis_suite/figs/task_schematic.png.
Run: python code/analysis_suite/make_task_schematic.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

FIGS = HERE.parent.parent / "docs" / "analysis_suite" / "figs"

ROW_FILL = {"English": "#ececec", "German": "#d6d6d6"}


def main():
    S.apply()
    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], width_ratios=[1.05, 1.0],
                          hspace=0.5, wspace=0.28)

    # ---- top left: a small illustrative spike raster
    axr = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(7)
    n_ch, T = 28, 100
    for ch in range(n_ch):
        rate = 0.04 + 0.10 * np.exp(-((np.arange(T) - (20 + 3 * ch % 60)) ** 2) / (2 * 22 ** 2))
        t = np.where(rng.random(T) < rate)[0]
        axr.scatter(t, np.full_like(t, ch), s=2.2, marker="|", color=S.ACCENT, linewidths=0.7)
    axr.set_xlim(0, T)
    axr.set_ylim(-1, n_ch)
    axr.set_xlabel("Time (1 s trial)", fontsize=8.5)
    axr.set_ylabel("Input channels (700)", fontsize=8.5)
    S.panel_title(axr, "a", "Spike-train input (illustrative)", fontsize=9.5)
    axr.set_xticks([])
    axr.set_yticks([])
    axr.grid(False)
    for s in axr.spines.values():
        s.set_visible(True)
        s.set_color(S.AXIS)

    # ---- top right: the dataset description
    axt = fig.add_subplot(gs[0, 1])
    axt.axis("off")
    axt.text(0.0, 0.98, "(b) Spiking Heidelberg Digits", fontsize=9.5, fontweight="bold", va="top")
    axt.text(0.0, 0.72,
             "Spoken digits 0 to 9, in English and German.\n"
             "700 input channels, spike trains over 1 s.\n"
             "Twenty classes: 10 digits $\\times$ 2 languages.",
             fontsize=9.5, va="top", linespacing=1.5)
    axt.text(0.0, 0.06, "Collapsed by language $\\times$ parity into four classes:",
             fontsize=9.5, color=S.ACCENT, va="bottom")

    # ---- bottom: the 2x2 task grouping
    axg = fig.add_subplot(gs[1, :])
    axg.axis("off")
    axg.set_xlim(0, 4)
    axg.set_ylim(0, 3)

    cols = [("Even", "0  2  4  6  8"), ("Odd", "1  3  5  7  9")]
    rows = ["English", "German"]
    x0, y0, w, h = 1.15, 0.15, 1.25, 1.15
    axg.text(x0 - 0.12, y0 + 2 * h + 0.55, "(c) The four task classes", ha="left", va="bottom",
             fontsize=9.5, fontweight="bold")
    for c, (par, digs) in enumerate(cols):
        cx = x0 + w * (c + 0.5)
        axg.text(cx, y0 + 2 * h + 0.30, par, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        axg.text(cx, y0 + 2 * h + 0.10, digs, ha="center", va="bottom", fontsize=8.5, color=S.MUTED)
    for r, lang in enumerate(rows):
        ry = y0 + h * (1 - r)
        axg.text(x0 - 0.12, ry + h / 2, lang, ha="right", va="center", fontsize=10.5, fontweight="bold")
        for c, (par, _) in enumerate(cols):
            cx = x0 + w * c
            cls = r * 2 + c
            axg.add_patch(Rectangle((cx + 0.02, ry + 0.02), w - 0.04, h - 0.04, facecolor=ROW_FILL[lang],
                                    edgecolor="white", linewidth=2))
            axg.text(cx + w / 2, ry + h * 0.62, f"Class {cls}", ha="center", va="center", fontsize=11,
                     fontweight="bold")
            axg.text(cx + w / 2, ry + h * 0.28, f"{lang}, {par.lower()}", ha="center", va="center",
                     fontsize=8.5, color=S.MUTED)
    axg.text(x0 + 2 * w + 0.15, y0 + 2 * h + 0.30, "Chance 25%", ha="left", va="bottom", fontsize=8.5,
             color=S.MUTED)

    FIGS.mkdir(parents=True, exist_ok=True)
    S.save(fig, FIGS / "task_schematic.png")


if __name__ == "__main__":
    main()
