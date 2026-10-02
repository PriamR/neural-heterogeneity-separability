"""Redraw the five report figures that build_report.py produces, from the per-network records alone.

build_report.py also audits the checkpoints with torch and writes the markdown tables. For the figures none of
that is needed: every number they show is in data_cache/network_json (one record per trained network), so this
script redraws them from those records with no torch and no checkpoints, using stats_core for the paired
statistics so the figure cannot drift from the tables.

  F4_1_paired_accuracy            (a) paired task accuracy, (b) per-seed gain
  F4_2_tau_solutions              learned time constants, pooled heterogeneous histogram and homogeneous lines
  F4_3_recruitment                (a) per-seed active-unit difference, (b) active units against accuracy
  F4_12_scaled_init_dissociation  the scaled-init control on five measures, ten seeds
  T4_10_gain                      (a) paired recurrent gain, (b) gain against accuracy with the controls

Run: python code/analysis_suite/redraw_report_figs.py
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
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_NEW = list(range(1001, 1011))
SEEDS = SEEDS_ORIG + SEEDS_NEW
CONFOUNDED = {202, 210, 340, 440, 550, 660, 710, 820, 930}
LONG_TAU_HOM = {1004, 1007}

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


het = {s: R[f"4class_seed{s}_het_lognormal"] for s in SEEDS}
hom = {s: R[hom_name(s)] for s in SEEDS}
scaled = {s: R[f"4class_seed{s}_hom_scaledinit_inputrec_lognormal"] for s in SEEDS_ORIG
          if f"4class_seed{s}_hom_scaledinit_inputrec_lognormal" in R}
V = lambda d, k, seeds=SEEDS: np.array([d[s][k] for s in seeds], float)


# ===================================================================== F4.1 paired accuracy
acc_h, acc_o = V(het, "acc"), V(hom, "acc")
r_acc = paired_triple(acc_h, acc_o, "Task accuracy")
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})

ax = axes[0]
S.paired_slopes(ax, acc_h, acc_o)
ax.set_ylabel("Task accuracy")
ax.set_ylim(0.5, 0.8)
S.panel_title(ax, "a", "Task accuracy across pairs")
S.legend(ax, loc="upper right")
S.note_below(ax, S.stats_line(r_acc))

ax = axes[1]
d = acc_h - acc_o
order = np.argsort(d)
labels = [f"{SEEDS[i]}{'*' if SEEDS[i] in LONG_TAU_HOM else ''}" for i in order]
S.bars(ax, np.arange(len(d)), d[order], [S.HET if v > 0 else S.HOM for v in d[order]], width=0.7)
ax.axhline(0, color=S.ACCENT, lw=0.8)
ax.set_xticks(range(len(d)))
ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Accuracy gain (heterogeneous $-$ homogeneous)")
ax.set_xlabel("Seed, sorted by gain")
S.panel_title(ax, "b", "Accuracy gain per seed")
S.save(fig, FIGS / "F4_1_paired_accuracy.png")

# ===================================================================== F4.2 learned time constants
allt = np.concatenate([np.asarray(het[s]["tau_all_ms"], float) for s in SEEDS])
homt = np.array([hom[s]["tau_median_ms"] for s in SEEDS])
fig, ax = plt.subplots(figsize=(8.2, 4.6))
bins = np.logspace(np.log10(3.0), np.log10(210.0), 46)
ax.hist(allt, bins=bins, color=S.HET, alpha=0.85, edgecolor="white", linewidth=0.5,
        label=f"Heterogeneous, pooled per-neuron $\\tau$ ($n$ = {allt.size:,})")
for t in homt:
    ax.axvline(t, color=S.HOM, alpha=0.75, lw=1.4, zorder=3)
ax.axvline(np.nan, color=S.HOM, lw=1.4, label="Homogeneous, single learned $\\tau$ (one line per seed)")
HOM_SUMMARY = "#3a9d5d"        # summary of the twenty homogeneous constants
HOM_SUMMARY_GEO = "#d4a017"    # geometric mean, the natural centre on a log axis
homt_geo = float(np.exp(np.log(homt).mean()))
ax.axvline(homt.mean(), color=HOM_SUMMARY, lw=2.2, zorder=4,
           label=f"Homogeneous, arithmetic mean of the twenty constants ({homt.mean():.1f} ms)")
ax.axvline(homt_geo, color=HOM_SUMMARY_GEO, lw=2.2, zorder=4,
           label=f"Homogeneous, geometric mean of the twenty constants ({homt_geo:.1f} ms)")
S.ref_line(ax, x=3.26, label="Clip floor, 3.26 ms", ls=(0, (1, 2)))
S.ref_line(ax, x=199.5, label="Clip ceiling, 199.5 ms", ls=(0, (4, 2)))
ax.set_xscale("log")
S.plain_log_ticks(ax, [3, 5, 10, 20, 50, 100, 200])
ax.set_xlabel("Membrane time constant (ms, log scale)")
ax.set_ylabel("Number of hidden units")
S.panel_title(ax, None, "Learned membrane time constants")
# boxed legend: the homogeneous lines run the full height of the axes, so an unboxed legend is crossed by them
ax.legend(loc="upper right", ncol=1, bbox_to_anchor=(0.94, 0.99), frameon=True, framealpha=1.0,
          facecolor="white", edgecolor=S.GRID, fancybox=False)
print(f"  homogeneous constants: mean {homt.mean():.2f}, median {np.median(homt):.2f}, "
      f"geometric mean {np.exp(np.log(homt).mean()):.2f} ms")
S.save(fig, FIGS / "F4_2_tau_solutions.png")

# ===================================================================== F4.3 recruitment
act_h, act_o = V(het, "active"), V(hom, "active")
r_act = paired_triple(act_h, act_o, "Active units")
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})

ax = axes[0]
# the same paired-slope idiom as the accuracy figure: each network's count, pairs connected, so the reader
# sees both how many units are active (out of sixty-four) and that the two members of a pair barely differ
# two bars from zero to the full population, 95% CI whiskers and one dot per network: the message is that the
# two types of network switch on the same share of their sixty-four units, so the axis must not be truncated
S.paired_bars(ax, [""], {"het": act_h[None, :], "hom": act_o[None, :]}, None, width=0.34)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 64)
ax.set_yticks([0, 16, 32, 48, 64])
ax.set_xticks([-0.19, 0.19])
ax.set_xticklabels([S.LABEL["het"], S.LABEL["hom"]])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Active hidden units (of 64)")
S.panel_title(ax, "a", "Active units by network type")
S.note_below(ax, S.stats_line(r_act), dy=-0.12)

ax = axes[1]
xs = np.concatenate([act_h, act_o])
ys = np.concatenate([acc_h, acc_o])
S.dots(ax, act_h, acc_h, S.HET, label=S.LABEL["het"])
S.dots(ax, act_o, acc_o, S.HOM, label=S.LABEL["hom"])
rr = np.corrcoef(xs, ys)[0, 1]
z = np.polyfit(xs, ys, 1)
xg = np.linspace(xs.min(), xs.max(), 40)
ax.plot(xg, np.polyval(z, xg), color=S.ACCENT, ls=(0, (4, 2)), lw=1.3, label=f"Linear fit, $r$ = {rr:.2f}")
ax.set_xlabel("Active hidden units")
ax.set_ylabel("Task accuracy")
S.panel_title(ax, "b", "Active units against accuracy")
S.legend_below(ax, ncol=3, dy=-0.2)
S.save(fig, FIGS / "F4_3_recruitment.png")
print(f"  recruitment r = {rr:.3f}")

# ===================================================================== F4.12 scaled-init control
if scaled:
    ss = sorted(scaled)
    PANELS = [("mean_rate", "Mean firing rate\n(spikes per unit per ms)"),
              ("gain", "Recurrent gain\n(spectral radius)"),
              ("acc", "Task accuracy"),
              ("active", "Active hidden units"),
              ("n90_1ms", "Components for 90% variance, $n_{90}$")]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14.0, 3.9))
    fig.subplots_adjust(wspace=0.45)
    rng = np.random.default_rng(0)
    for i, (ax, (k, lab)) in enumerate(zip(axes, PANELS)):
        vals = [np.array([het[s][k] for s in ss], float), np.array([hom[s][k] for s in ss], float),
                np.array([scaled[s][k] for s in ss], float)]
        for j, (v, c) in enumerate(zip(vals, [S.HET, S.HOM, S.SCALED])):
            S.bars(ax, j, v.mean(), c, width=0.62, err=S.ci95_mean(v))
            S.seed_dots(ax, j, v, width=0.62, rng=rng)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(lab)
        if k == "acc":
            S.chance_line(ax)
        S.panel_title(ax, "abcde"[i], lab.split("\n")[0].replace(", $n_{90}$", ""))
    S.arch_legend(fig, extra=[(S.SCALED, S.LABEL["scaled"])], bbox=(0.5, -0.06))
    S.save(fig, FIGS / "F4_12_scaled_init_dissociation.png")

# ===================================================================== T4.10 recurrent gain
g_h, g_o = V(het, "gain"), V(hom, "gain")
r_g = paired_triple(g_h, g_o, "Recurrent gain")
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

ax = axes[0]
S.paired_slopes(ax, g_h, g_o)
S.ref_line(ax, y=1.0, label="Unit gain")
ax.set_ylabel("Recurrent spectral radius")
ax.set_ylim(0.8, 3.35)
S.panel_title(ax, "a", "Recurrent gain across pairs")
S.legend(ax, loc="upper right")
S.note_below(ax, S.stats_line(r_g))

ax = axes[1]
S.dots(ax, g_h, acc_h, S.HET, label=S.LABEL["het"])
S.dots(ax, g_o, acc_o, S.HOM, label=S.LABEL["hom"])
r40 = np.corrcoef(np.concatenate([g_h, g_o]), np.concatenate([acc_h, acc_o]))[0, 1]
r50 = r40
if scaled:
    ss = sorted(scaled)
    gs = np.array([scaled[s]["gain"] for s in ss]); as_ = np.array([scaled[s]["acc"] for s in ss])
    S.dots(ax, gs, as_, S.SCALED, label=S.LABEL["scaled"], marker="D", s=30)
    r50 = np.corrcoef(np.concatenate([g_h, g_o, gs]), np.concatenate([acc_h, acc_o, as_]))[0, 1]
ax.set_xlabel("Recurrent spectral radius")
ax.set_ylabel("Task accuracy")
S.panel_title(ax, "b", "Recurrent gain against accuracy")
S.legend_below(ax, ncol=3, dy=-0.2)
S.note(ax, f"$r$ = {r40:.2f} over the forty matched networks\n$r$ = {r50:.2f} with the ten controls", loc="upper left")
S.save(fig, FIGS / "T4_10_gain.png")
print(f"  gain: matched r = {r40:.3f}, with controls r = {r50:.3f}")
print("done")
