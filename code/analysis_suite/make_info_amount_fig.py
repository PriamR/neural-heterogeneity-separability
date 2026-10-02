"""The amount-of-information result as one simple figure, to sit beside the
O-information figure that carries the kind-of-information result.

The O-information figure (F4_5_minmax_oinfo_clean) shows that the *kind* of
interaction structure does not change: both arms are redundancy-dominated. This
figure shows the other half, that the heterogeneous code carries *more* class
information of every kind. It reads the pairwise decomposition (mixed-kNN mutual
information with dithering and a label-permutation null, one value per network)
and draws, for each of the four quantities, a dumbbell from the homogeneous mean
to the heterogeneous mean with the per-seed values behind it.

Reads data_cache/supervisor_extension/rsi_n20.json, writes
docs/supervisor_extension/figs/pairwise_info_amount.png.

Run: python code/analysis_suite/make_info_amount_fig.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

REPO = HERE.parent.parent
DATA = REPO / "data_cache" / "supervisor_extension" / "rsi_n20.json"
FIGS = REPO / "docs" / "supervisor_extension" / "figs"

# (metric key, row label, what it measures)
ROWS = [
    ("I_marginal", "single unit", r"$I(\mathrm{unit};\,\mathrm{class})$"),
    ("I_joint", "unit pair", r"$I(\mathrm{pair};\,\mathrm{class})$"),
    ("redundancy", "redundancy", r"$\min(I_i,I_j)$"),
    ("synergy", "synergy", r"$I_{ij}-\max(I_i,I_j)$"),
]


def main():
    d = json.load(open(DATA))
    seeds = d["seeds"]
    pn = d["per_network"]
    pr = d["paired"]

    S.apply()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))

    for i, (key, name, formula) in enumerate(ROWS):
        y = len(ROWS) - 1 - i  # first row on top
        het = np.array([pn[f"{s}_het"][key] for s in seeds], dtype=float)
        hom = np.array([pn[f"{s}_hom"][key] for s in seeds], dtype=float)
        hm, tm = hom.mean(), het.mean()

        # per-seed values behind the means, small vertical jitter for legibility
        jit = (np.arange(len(seeds)) % 5 - 2) * 0.028
        ax.scatter(hom, y + 0.15 + jit, s=9, color=S.COL["hom"], alpha=0.28, lw=0, zorder=2)
        ax.scatter(het, y - 0.15 + jit, s=9, color=S.COL["het"], alpha=0.28, lw=0, zorder=2)

        # the dumbbell of means
        ax.plot([hm, tm], [y, y], color="0.55", lw=2.0, zorder=3, solid_capstyle="round")
        ax.scatter([hm], [y], s=70, color=S.COL["hom"], zorder=4, edgecolor="white", lw=0.8)
        ax.scatter([tm], [y], s=70, color=S.COL["het"], zorder=4, edgecolor="white", lw=0.8)

        st = pr[key]
        ax.annotate(f"$\\Delta$+{st['delta']:.3f},  $d_z$ {st['dz']:.2f},  {st['wins']}/20",
                    xy=(max(tm, hm), y), xytext=(8, 9), textcoords="offset points",
                    fontsize=7.6, color="0.30", va="center")
        ax.text(-0.012, y, name, ha="right", va="center", fontsize=9.5,
                transform=ax.get_yaxis_transform())
        ax.text(-0.012, y - 0.30, formula, ha="right", va="center", fontsize=7.3,
                color="0.45", transform=ax.get_yaxis_transform())

    # legend proxies
    ax.scatter([], [], s=70, color=S.COL["het"], label="heterogeneous")
    ax.scatter([], [], s=70, color=S.COL["hom"], label="homogeneous")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("class information (bits)")
    ax.margins(x=0.16)
    ax.set_title("The heterogeneous code carries more class information of every kind",
                 fontweight="bold", fontsize=11, loc="left", pad=8)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)

    FIGS.mkdir(parents=True, exist_ok=True)
    S.save(fig, FIGS / "pairwise_info_amount.png")
    print("  wrote pairwise_info_amount.png")


if __name__ == "__main__":
    main()
