"""The within- and between-community weight of the two types of network across the community count K.
Forcing K communities on each functional graph, the mean edge weight inside communities and the mean edge
weight between them, each normalised by the network's own mean edge weight. Stars mark a paired t p below
0.05 at that K.

Reads data_cache/supervisor_extension/block_k_sweep.json, writes
docs/analysis_suite/figs/block_within_between_ksweep.png.

Run: python code/analysis_suite/make_block_ksweep_fig.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
DATA = REPO / "data_cache" / "supervisor_extension" / "block_k_sweep.json"
FIGS = REPO / "docs" / "analysis_suite" / "figs"


def main():
    S.apply()
    d = json.load(open(DATA))["by_k"]
    ks = sorted(int(k) for k in d)
    x = np.array(ks, dtype=float)
    n = d[str(ks[0])]["within"]["n"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    fig.subplots_adjust(wspace=0.28)
    panels = [("within", "a", "Within-community weight"),
              ("between", "b", "Between-community weight")]
    for ax, (key, letter, title) in zip(axes, panels):
        for arch in ("het", "hom"):
            m = np.array([d[str(k)][key][f"{arch}_mean"] for k in ks])
            h = 1.96 * np.array([d[str(k)][key][f"{arch}_sd"] for k in ks]) / np.sqrt(n)
            S.band(ax, x, m, h, S.COL[arch], label=S.LABEL[arch], marker="o", ms=4)
        top = ax.get_ylim()[1]
        for k in ks:
            if d[str(k)][key]["t_p"] < 0.05:
                ax.text(k, top, "*", ha="center", va="top", fontsize=13, color=S.ACCENT)
        S.panel_title(ax, letter, title)
        ax.set_xlabel("Number of communities, $K$")
        ax.set_ylabel("Mean edge weight / network mean")
        ax.set_xticks(ks)
    S.legend(axes[0], loc="lower right")
    FIGS.mkdir(parents=True, exist_ok=True)
    S.save(fig, FIGS / "block_within_between_ksweep.png")


if __name__ == "__main__":
    main()
