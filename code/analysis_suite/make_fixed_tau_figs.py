"""Fixed-tau ladder figures.

Each figure carries the learned heterogeneous and learned homogeneous networks alongside the four fixed-tau
networks, so the reader sees directly how far a fixed timescale falls short of a learned spread on that measure.
The rungs exist for the ten original seeds only, so the learned networks are restricted to those same ten
seeds here: every bar and curve is a mean over the identical ten networks.

Rungs are named by their nominal value. The 2 ms rung actually sits at 3.26 ms because of the membrane-decay
clip; that is stated in the appendix and the caption rather than carried in the figure.

Writes F4_10_accuracy_ladder.png, F4_10b_fixed_tau_pca.png, F4_10c_fixed_tau_reconstruction.png.

Run: python code/analysis_suite/make_fixed_tau_figs.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "network_json"
FIGS = REPO / "docs" / "analysis_suite" / "figs"
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
CONFOUNDED = {202, 210, 340, 440, 550, 660, 710, 820, 930}
RUNGS = [2.0, 3.5, 15.0, 35.0]
TAG = {2.0: "fixed2ms", 3.5: "fixed3p5ms", 15.0: "fixed15ms", 35.0: "fixed35ms"}

R = {}
for f in CACHE.glob("*.json"):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if isinstance(r, dict) and "name" in r:
        R[r["name"]] = r


def hom_name(s):
    nowd = f"4class_seed{s}_hom_geommean_nowd_lognormal"
    return nowd if (s in CONFOUNDED and nowd in R) else f"4class_seed{s}_hom_geommean_lognormal"


arms = {
    "het": [R[f"4class_seed{s}_het_lognormal"] for s in SEEDS],
    "hom": [R[hom_name(s)] for s in SEEDS],
}
for t in RUNGS:
    arms[f"fixed{t}"] = [R[f"4class_seed{s}_hom_{TAG[t]}_lognormal"] for s in SEEDS
                         if f"4class_seed{s}_hom_{TAG[t]}_lognormal" in R]

ORDER = ["het", "hom"] + [f"fixed{t}" for t in RUNGS]
COLOURS = [S.HET, S.HOM] + [S.FIXED_RAMP[t] for t in RUNGS]
LABELS = [S.LABEL["het"], S.LABEL["hom"]] + [S.FIXED_LABEL[t] for t in RUNGS]


def vals(key, field):
    return np.array([r[field] for r in arms[key]], dtype=float)


def bar_panel(ax, field, ylabel, letter, title):
    rng = np.random.default_rng(0)
    means = []
    for i, k in enumerate(ORDER):
        v = vals(k, field)
        means.append(v.mean())
        S.bars(ax, i, v.mean(), COLOURS[i], width=0.62, err=S.ci95_mean(v))
        S.seed_dots(ax, i, v, width=0.62, rng=rng)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    S.panel_title(ax, letter, title)
    return means


# ------------------------------------------------------------------ accuracy ladder
fig, ax = plt.subplots(figsize=(6.6, 4.4))
means = bar_panel(ax, "acc", "Task accuracy", None, "Task accuracy on the fixed-tau ladder")
S.chance_line(ax)
ax.set_ylim(0.2, 0.8)
best_t = RUNGS[int(np.argmax(means[2:]))]
r_best = paired_triple(vals("het", "acc"), vals(f"fixed{best_t}", "acc"), "het vs best rung")
S.note(ax, f"Heterogeneous against the best fixed-tau network ({best_t:g} ms):\n"
           + S.stats_line(r_best, test="perm"), loc="upper right", fontsize=7.8)
S.save(fig, FIGS / "F4_10_accuracy_ladder.png")
print(f"  het - best rung ({best_t:g} ms): {r_best['delta']:+.3f}, dz {r_best['dz']:.2f}, "
      f"sign-flip p {r_best['perm_p']:.3g}, wins {r_best['wins']}/{r_best['n']}")

# ------------------------------------------------------------------ effective dimension across the ladder
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
fig.subplots_adjust(wspace=0.30)
bar_panel(axes[0], "n90_1ms", "Components for 90% of the variance, $n_{90}$", "a", "Components for 90% variance")
bar_panel(axes[1], "pr_1ms", "Participation ratio", "b", "Participation ratio")
S.save(fig, FIGS / "F4_10b_fixed_tau_pca.png")

# ------------------------------------------------------------------ reconstruction through the readout
# The learned networks are drawn over all twenty seeds here, as in the scree figure and the prose of sec:recon;
# the fixed-tau networks exist for the ten original seeds only.
SEEDS20 = SEEDS + list(range(1001, 1011))
learned20 = {
    "het": [R[f"4class_seed{s}_het_lognormal"] for s in SEEDS20],
    "hom": [R[hom_name(s)] for s in SEEDS20],
}
fig, ax = plt.subplots(figsize=(7.6, 4.6))
grid = np.array(arms["het"][0]["M_GRID"], dtype=float)
for k, lbl, col in zip(ORDER, LABELS, COLOURS):
    curves = np.array([r["pc_curve"] for r in (learned20[k] if k in learned20 else arms[k])], dtype=float)
    m = curves.mean(0)
    if k in ("het", "hom"):
        h = curves.std(0, ddof=1) / np.sqrt(curves.shape[0]) * 1.96
        S.band(ax, grid, m, h, col, label=lbl, marker="o", ms=3.5, zorder=3)
    else:
        ax.plot(grid, m, color=col, lw=1.5, ls=(0, (4, 2)), label=lbl, zorder=2)
S.chance_line(ax)
S.ref_line(ax, x=8, ls=(0, (1, 2)))
ax.set_xscale("log", base=2)
ax.set_xticks(grid)
ax.set_xticklabels([f"{int(g)}" for g in grid])
ax.set_xlabel("Retained principal components, $m$")
ax.set_ylabel("Task accuracy through the trained readout")
ax.set_ylim(0.2, 0.8)
S.panel_title(ax, None, "Accuracy from rank-$m$ reconstructions")
S.legend(ax, loc="upper left", ncol=2)
S.save(fig, FIGS / "F4_10c_fixed_tau_reconstruction.png")
for k in ("het", "hom"):
    c = np.array([r["pc_curve"] for r in learned20[k]], dtype=float).mean(0)
    print(f"  recon {k} (n=20) at m=8 {c[list(grid).index(8)]:.3f}, m=32 {c[list(grid).index(32)]:.3f}")

print("\nmeans over the ten shared seeds:")
for k, lbl in zip(ORDER, LABELS):
    print(f"  {lbl:16s} acc {vals(k,'acc').mean():.3f}   n90 {vals(k,'n90_1ms').mean():5.1f}   "
          f"PR {vals(k,'pr_1ms').mean():5.2f}   n={len(arms[k])}")
