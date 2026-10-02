import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HET = "#d1495b"
HOM = "#30638e"
SCALED = "#e8a33d"
CHANCE = "#8d8d8d"
ACCENT = "#2f2f2f"

COL = {"het": HET, "hom": HOM, "scaled": SCALED}


LABEL = {
    "het": "heterogeneous",
    "hom": "homogeneous",
    "scaled": "scaled-init hom",
}

CHANCE_LEVEL = 0.25


def apply():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "errorbar.capsize": 3,
    })


def chance_line(ax, level=CHANCE_LEVEL, label="chance (4-class)"):
    ax.axhline(level, color=CHANCE, ls=":", lw=1.2, zorder=0, label=label)


def band(ax, x, mean, sd, color, label=None, alpha=0.15, **kw):
    ax.plot(x, mean, color=color, label=label, **kw)
    ax.fill_between(x, np.asarray(mean) - np.asarray(sd), np.asarray(mean) + np.asarray(sd),
                    color=color, alpha=alpha, lw=0, zorder=1)


def fmt_p(p, floor=None):
    if p != p:
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def save(fig, path, close=True):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"saved {path.name}", flush=True)
