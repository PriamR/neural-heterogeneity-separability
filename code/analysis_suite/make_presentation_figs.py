"""Presentation figure set for the supervisor review deck.

Builds, from the cached per-network records in data_cache/analysis_suite (all 100 nets), the figures the
presentation markdown embeds, plus a stats JSON (presentation_stats.json) with the paired significance for each.
Figures go to docs/analysis_suite/figs/presentation/. Reuses the O-info line figure and the ladder figure
already produced by the suite.

Conventions:
  * het and clean hom are the n=20 arm (no-WD retrains for seeds 202..930).
  * scaled-init and the fixed-tau rungs are the n=10 original seeds. For a single consistent reference across the
    deck they are compared to the clean learned hom (not the WD hom). They carry weight decay 1e-4, which the
    confound analysis shows changes accuracy by ~0.04, so the control-vs-hom direction is unaffected.
  * Error bars / bands are 95% confidence intervals (mean +/- t_{.975,n-1} * SE).

Run: python code/analysis_suite/make_presentation_figs.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "analysis_suite"
FIGS = REPO / "docs" / "analysis_suite" / "figs" / "presentation"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_ALL = SEEDS_ORIG + list(range(1001, 1011))
CONFOUNDED = [202, 210, 340, 440, 550, 660, 710, 820, 930]
RUNGS = [2.0, 3.5, 15.0, 35.0]
TAG = {2.0: "fixed2ms", 3.5: "fixed3p5ms", 15.0: "fixed15ms", 35.0: "fixed35ms"}
RUNG_LABEL = {2.0: "2 ms*", 3.5: "3.5 ms", 15.0: "15 ms", 35.0: "35 ms"}
ANOM = {1004, 1007}


def rec(name):
    return json.load(open(CACHE / f"{name}.json"))


def hom_name(s):
    return (f"4class_seed{s}_hom_geommean_nowd_lognormal" if s in CONFOUNDED
            else f"4class_seed{s}_hom_geommean_lognormal")


het = {s: rec(f"4class_seed{s}_het_lognormal") for s in SEEDS_ALL}
hom = {s: rec(hom_name(s)) for s in SEEDS_ALL}
scaled = {s: rec(f"4class_seed{s}_hom_scaledinit_inputrec_lognormal") for s in SEEDS_ORIG}
fixed = {t: {s: rec(f"4class_seed{s}_hom_{TAG[t]}_lognormal") for s in SEEDS_ORIG} for t in RUNGS}

FIXED_COL = {2.0: S.FIXED_RAMP[2.0], 3.5: S.FIXED_RAMP[3.5], 15.0: S.FIXED_RAMP[15.0], 35.0: S.FIXED_RAMP[35.0]}
V = lambda d, k, seeds=SEEDS_ALL: np.array([d[s][k] for s in seeds], float)


def ci95(x):
    x = np.asarray(x, float); n = x.size
    if n < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(st.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def pcm(d, m, seeds):
    return np.array([d[s]["pc_curve"][d[s]["M_GRID"].index(m)] for s in seeds])


STATS = {}


def sig_row(label, r):
    return {"comparison": label, "a_mean": r["het_mean"], "b_mean": r["hom_mean"], "delta": r["delta"],
            "dz": r["dz"], "t_p": r["t_p"], "w_p": r["w_p"], "perm_p": r["perm_p"],
            "perm_floor": r["perm_floor"], "wins": r["wins"], "n": r["n"]}


# =====================================================================================
# G1a  per-seed accuracy delta (het - hom), clean n=20
# =====================================================================================
acc_h, acc_o = V(het, "acc"), V(hom, "acc")
d = acc_h - acc_o
order = np.argsort(d)
lbl = [str(SEEDS_ALL[i]) + ("*" if SEEDS_ALL[i] in ANOM else "") for i in order]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(range(len(d)), d[order], color=[S.HET if v > 0 else S.HOM for v in d[order]])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(d))); ax.set_xticklabels(lbl, rotation=90, fontsize=7.5)
ax.set_ylabel("accuracy gain (het - hom)")
ax.set_xlabel("seed (sorted; * = long-tau hom)")
ax.set_title("Per-seed accuracy difference, het minus hom (n = 20)")
r = paired_triple(acc_h, acc_o, "accuracy het vs hom")
ax.annotate(f"het wins {r['wins']}/20\nmean delta = {r['delta']:+.3f}\nsign-flip p = {S.fmt_p(r['perm_p'], r['perm_floor'])}",
            xy=(0.03, 0.72), xycoords="axes fraction", fontsize=9, color=S.ACCENT)
S.save(fig, FIGS / "G1a_accuracy_per_seed.png")
STATS["G1a"] = sig_row("Task accuracy: het vs clean hom (n=20)", r)


# =====================================================================================
# helper: grouped-by-type bar with 95% CI, and the paired stats table vs the matched baseline
# =====================================================================================
def grouped_bar(metric, title, ylabel, fname, fmt="{:.3f}", chance=None, higher=True):
    groups = [("het", V(het, metric), S.HET, 20),
              ("hom", V(hom, metric), S.HOM, 20),
              ("scaled", V(scaled, metric, SEEDS_ORIG), S.SCALED, 10)]
    for t in RUNGS:
        groups.append((RUNG_LABEL[t], V(fixed[t], metric, SEEDS_ORIG), FIXED_COL[t], 10))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.arange(len(groups))
    for i, (nm, vals, col, n) in enumerate(groups):
        m, e = ci95(vals)
        ax.bar(i, m, yerr=e, color=col, width=0.68, zorder=2, capsize=3)
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).uniform(-.12, .12, len(vals)), vals,
                   color="k", s=8, alpha=0.45, zorder=3)
    if chance is not None:
        ax.axhline(chance, color=S.CHANCE, ls=":", lw=1.2, zorder=0)
    # het and hom carry all 20 seeds, the controls only the 10 original ones, so every bar states its own n
    ax.set_xticks(xs); ax.set_xticklabels([f"{g[0]}\n(n={g[3]})" for g in groups], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title)
    S.save(fig, FIGS / fname)
    # Control comparisons against the clean learned hom (n=10 original seeds), one baseline throughout the deck.
    # The het-vs-hom row is shown once in the per-seed section, so it is not repeated here.
    rows = [sig_row("scaled vs learned hom (n=10)",
                    paired_triple(V(scaled, metric, SEEDS_ORIG), V(hom, metric, SEEDS_ORIG), "scaled", higher))]
    for t in RUNGS:
        rows.append(sig_row(f"fixed {RUNG_LABEL[t]} vs learned hom (n=10)",
                            paired_triple(V(fixed[t], metric, SEEDS_ORIG), V(hom, metric, SEEDS_ORIG), "fixed", higher)))
    return rows


# =====================================================================================
# G1b accuracy grouped, G2 active neurons, G4 firing rate
# =====================================================================================
STATS["G1b"] = grouped_bar("acc", "Task accuracy by network type", "task accuracy",
                           "G1b_accuracy_grouped.png", chance=0.25)

# G2a per-seed active delta
act_h, act_o = V(het, "active"), V(hom, "active")
da = act_h - act_o
order = np.argsort(da)
lbl = [str(SEEDS_ALL[i]) + ("*" if SEEDS_ALL[i] in ANOM else "") for i in order]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.bar(range(len(da)), da[order], color=[S.HET if v > 0 else S.HOM for v in da[order]])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(da))); ax.set_xticklabels(lbl, rotation=90, fontsize=7.5)
ax.set_ylabel("active-unit difference (het - hom)")
ax.set_xlabel("seed (sorted; * = long-tau hom)")
ax.set_title("Per-seed active-neuron difference, het minus hom (n = 20)")
r = paired_triple(act_h, act_o, "active het vs hom")
ax.annotate(f"het higher on {r['wins']}/20\nmean delta = {r['delta']:+.2f}\nsign-flip p = {S.fmt_p(r['perm_p'], r['perm_floor'])}",
            xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9, color=S.ACCENT)
S.save(fig, FIGS / "G2a_active_per_seed.png")
STATS["G2a"] = sig_row("Active neurons: het vs clean hom (n=20)", r)

STATS["G2b"] = grouped_bar("active", "Active neurons by network type", "active hidden units (>=1 spike)",
                           "G2b_active_grouped.png")

# =====================================================================================
# G3  averaged het tau distribution across seeds
# =====================================================================================
fig, ax = plt.subplots(figsize=(9, 4.8))
bins = np.logspace(np.log10(3), np.log10(210), 41)
centers = np.sqrt(bins[:-1] * bins[1:])
# per-seed normalised histogram, then mean + 95% CI across the 20 het seeds
H = np.stack([np.histogram(np.array(het[s]["tau_all_ms"]), bins=bins, density=False)[0] / 64.0 for s in SEEDS_ALL])
m = H.mean(0); e = st.t.ppf(0.975, 19) * H.std(0, ddof=1) / np.sqrt(20)
ax.bar(centers, m, width=np.diff(bins), align="center", color=S.HET, alpha=0.75, label="het per-neuron tau (mean over 20 seeds)")
ax.errorbar(centers, m, yerr=e, fmt="none", ecolor=S.ACCENT, elinewidth=0.7, alpha=0.6)
hom_taus = V(hom, "tau_median_ms")
# The homs are drawn as a rug along the top rather than as full-height lines: twenty full-height rules were
# covering the histogram they are meant to be compared against. The rug also makes the spread visible, which
# matters, because the hom solutions are not all short.
ytop = ax.get_ylim()[1]
ax.set_ylim(0, ytop * 1.30)
ytop = ax.get_ylim()[1]
ax.vlines(hom_taus, ytop * 0.90, ytop * 0.99, color=S.HOM, lw=1.6, alpha=0.8)
ax.vlines([], 0, 0, color=S.HOM, lw=1.6, label="hom single tau (one tick per seed)")
het_med = float(np.median([het[s]["tau_median_ms"] for s in SEEDS_ALL]))
ax.axvline(het_med, color=S.ACCENT, ls="--", lw=1.6, label=f"het pooled median = {het_med:.1f} ms")
ax.axvline(float(np.median(hom_taus)), color=S.HOM, ls="-.", lw=1.4,
           label=f"hom median across seeds = {np.median(hom_taus):.1f} ms")
ax.axvline(199.5, color=S.CHANCE, ls="--", lw=1.2, label="clip ceiling 199.5 ms")
ax.axvline(3.26, color=S.CHANCE, ls=":", lw=1.2, label="clip floor 3.26 ms")
_long = sorted(float(t) for t in hom_taus if t > 20)
ax.annotate(f"{len(_long)} of 20 homs land long\n({', '.join(f'{t:.0f}' for t in _long)} ms)",
            xy=(0.58, 0.55), xycoords="axes fraction", fontsize=8.5, color=S.HOM)
ax.set_xscale("log")
ax.set_xlabel("membrane time constant (ms, log scale)")
ax.set_ylabel("fraction of hidden units per bin")
ax.set_title("Learned time-constant distribution, het (averaged over seeds)")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.88))
S.save(fig, FIGS / "G3_tau_distribution.png")
# Does a long homogeneous constant predict a small gap? Rank correlation across the twenty seeds, reported
# because the two seeds where hom reaches het-level accuracy are the two longest, which invites the claim.
_rho, _rho_p = st.spearmanr(hom_taus, acc_h - acc_o)
STATS["G3"] = {"het_pooled_median_ms": het_med,
               "gap_vs_homtau_spearman": float(_rho), "gap_vs_homtau_p": float(_rho_p),
               "gap_by_long_seed": {str(s): float(acc_h[i] - acc_o[i]) for i, s in enumerate(SEEDS_ALL)
                                    if hom_taus[i] > 20},
               "hom_median_ms": float(np.median(hom_taus)),
               "hom_tau_min_ms": float(np.min(hom_taus)), "hom_tau_max_ms": float(np.max(hom_taus)),
               "hom_taus_sorted": {str(s): float(t) for s, t in
                                   sorted(zip(SEEDS_ALL, hom_taus), key=lambda kv: kv[1])},
               "hom_n_over_20ms": int(np.sum(hom_taus > 20)),
               "het_frac_on_ceiling": float(np.mean([np.mean(np.array(het[s]["tau_all_ms"]) > 199.0) for s in SEEDS_ALL]))}

# =====================================================================================
# G4  firing rate: per-seed paired + grouped
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
S.paired_slopes(axes[0], V(het, "mean_rate"), V(hom, "mean_rate"))
axes[0].set_ylabel("mean firing rate (spikes/step)")
r = paired_triple(V(het, "mean_rate"), V(hom, "mean_rate"), "rate het vs hom")
axes[0].set_title("Firing rate, het vs hom")
axes[0].legend(loc="upper right")
groups = [("het", V(het, "mean_rate"), S.HET), ("hom", V(hom, "mean_rate"), S.HOM),
          ("scaled", V(scaled, "mean_rate", SEEDS_ORIG), S.SCALED)] + \
         [(RUNG_LABEL[t], V(fixed[t], "mean_rate", SEEDS_ORIG), FIXED_COL[t]) for t in RUNGS]
for i, (nm, vals, col) in enumerate(groups):
    mm, ee = ci95(vals)
    axes[1].bar(i, mm, yerr=ee, color=col, width=0.68, capsize=3, zorder=2)
    axes[1].scatter(np.full(len(vals), i) + np.random.default_rng(0).uniform(-.12, .12, len(vals)), vals,
                    color="k", s=8, alpha=0.45, zorder=3)
axes[1].set_xticks(range(len(groups))); axes[1].set_xticklabels([g[0] for g in groups], fontsize=9)
axes[1].set_ylabel("mean firing rate"); axes[1].set_title("Firing rate by network type")
S.save(fig, FIGS / "G4_firing_rate.png")
STATS["G4"] = {"paired": sig_row("Firing rate: het vs clean hom (n=20)", r),
               "grouped": grouped_bar("mean_rate", "Firing rate by type", "rate", "G4b_rate_grouped_tmp.png", higher=True)}
(FIGS / "G4b_rate_grouped_tmp.png").unlink(missing_ok=True)   # only kept the two-panel G4

# =====================================================================================
# G6  fixed-tau ladder, per-seed trajectories only (the deck's single-panel version of F4.10; the paper keeps
#     the two-panel figure). One line per control seed across its learned pair and the four frozen rungs, so
#     every seed's learned spread standing above its whole fixed arm is visible directly.
# =====================================================================================
fseeds6 = [s for s in SEEDS_ORIG if all(s in fixed[t] for t in RUNGS)]
fig, ax = plt.subplots(figsize=(8.6, 5.2))
xpos = np.arange(6)
Mtr = np.zeros((len(fseeds6), 6))
for i, s in enumerate(fseeds6):
    Mtr[i, 0] = het[s]["acc"]; Mtr[i, 1] = hom[s]["acc"]
    for j, t in enumerate(RUNGS):
        Mtr[i, 2 + j] = fixed[t][s]["acc"]
ax.axvline(1.5, color=S.CHANCE, lw=0.8, alpha=0.4, zorder=0)
ax.text(0.5, 0.795, "learned", ha="center", va="top", fontsize=8, color=S.ACCENT, alpha=0.7)
ax.text(3.5, 0.795, "fixed tau", ha="center", va="top", fontsize=8, color=S.ACCENT, alpha=0.7)
for i in range(len(fseeds6)):
    ax.plot(xpos, Mtr[i], color="#b8b8b8", lw=0.8, alpha=0.7, zorder=1)
ax.plot(xpos, Mtr.mean(0), color=S.ACCENT, lw=2.4, marker="o", ms=6, zorder=4, label="mean over seeds")
ax.scatter([0] * len(fseeds6), Mtr[:, 0], color=S.HET, s=26, zorder=3)
ax.scatter([1] * len(fseeds6), Mtr[:, 1], color=S.HOM, s=26, zorder=3)
for j, t in enumerate(RUNGS):
    ax.scatter([2 + j] * len(fseeds6), Mtr[:, 2 + j], color=FIXED_COL[t], s=26, zorder=3)
S.chance_line(ax)
ax.set_xticks(xpos)
ax.set_xticklabels(["learned\nhet", "learned\nhom", "3.26*", "3.5", "15", "35"], fontsize=9)
ax.set_xlim(-0.4, 5.4); ax.set_ylim(0.20, 0.82)
ax.set_xlabel("network (learned, then fixed tau in ms)")
ax.set_ylabel("task accuracy")
ax.set_title(f"Fixed-tau ladder: per-seed accuracy (n = {len(fseeds6)} control seeds)")
ax.legend(loc="upper right", fontsize=8)
S.save(fig, FIGS / "G6_fixed_ladder_perseed.png")

# =====================================================================================
# G5b  O-information at k=8, bar (line graph reuses F4_5_minmax_oinfo_clean.png)
# =====================================================================================
OI = json.load(open(REPO / "docs" / "analysis_suite" / "oinfo_summary_clean.json"))
# k = 8 is the reliability ceiling; k = 10 sits beside it as a robustness check, both on all twenty pairs.
KS = [k for k in (8, 10) if OI.get(f"k{k}")]
cats = ["min O\n(most synergistic)", "max O\n(most redundant)", "spread\n(max - min)"]
xs = np.arange(3)
fig, axes = plt.subplots(1, len(KS), figsize=(6.2 * len(KS), 4.8), squeeze=False)
for ax, kk in zip(axes[0], KS):
    kd = OI[f"k{kk}"]
    for off, arch, col in [(-0.19, "het", S.HET), (0.19, "hom", S.HOM)]:
        ax.bar(xs + off, [kd["min"][arch], kd["max"][arch], kd["spread"][arch]], 0.36,
               color=col, label=S.LABEL[arch])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("O-information (bits)")
    ax.set_title(f"k = {kk} (n = {OI['n_by_k'][str(kk)]})")
    ax.legend(fontsize=9)
_ymax = max(a.get_ylim()[1] for a in axes[0]); _ymin = min(a.get_ylim()[0] for a in axes[0])
for a in axes[0]:
    a.set_ylim(_ymin, _ymax)                       # shared scale, or the two panels are not comparable
fig.suptitle("O-information of neuron subsets", fontweight="bold", y=1.01)
S.save(fig, FIGS / "G5b_oinfo_k8_bar.png")
STATS["G5b"] = {f"k{k}": OI[f"k{k}"] for k in KS}
STATS["G5b"].update({"frac_pos_k8": OI["frac_pos_k8"], "k8_table": OI["k8_table"],
                     "frac_pos_k10": OI.get("frac_pos_k10", {}), "k10_table": OI.get("k10_table", []),
                     "n_by_k": OI["n_by_k"]})

# =====================================================================================
# G7a PCA scree het vs hom (95% CI);  G7b PCA scree fixed-tau ladder (95% CI)
# =====================================================================================
def scree_ci(seeds_dict, seeds):
    E = np.stack([np.array(seeds_dict[s]["evr_1ms"]) for s in seeds])
    m = E.mean(0); e = st.t.ppf(0.975, len(seeds) - 1) * E.std(0, ddof=1) / np.sqrt(len(seeds))
    return m, e


x = np.arange(1, 65)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for dd, col, name in [(het, S.HET, "heterogeneous"), (hom, S.HOM, "homogeneous")]:
    m, e = scree_ci(dd, SEEDS_ALL)
    axes[0].plot(x, np.clip(m, 1e-7, None), color=col, marker="o", ms=2.5, label=name)
    axes[0].fill_between(x, np.clip(m - e, 1e-7, None), m + e, color=col, alpha=0.18, lw=0)
    cm = np.cumsum(np.stack([np.array(dd[s]["evr_1ms"]) for s in SEEDS_ALL]), 1)
    cmm = cm.mean(0); cme = st.t.ppf(0.975, 19) * cm.std(0, ddof=1) / np.sqrt(20)
    axes[1].plot(x, cmm, color=col, marker="o", ms=2.5, label=name)
    axes[1].fill_between(x, cmm - cme, cmm + cme, color=col, alpha=0.18, lw=0)
axes[0].set_yscale("log"); axes[0].set_ylim(1e-6, 1.5)
axes[0].set_xlabel("principal component"); axes[0].set_ylabel("explained variance ratio")
axes[0].set_title("PCA scree, het vs hom"); axes[0].legend()
axes[1].axhline(0.9, color=S.CHANCE, ls=":", lw=1.2)
axes[1].set_xlabel("principal component"); axes[1].set_ylabel("cumulative explained variance")
axes[1].set_title("Cumulative variance, het vs hom"); axes[1].legend()
S.save(fig, FIGS / "G7a_pca_scree_hom_het.png")
STATS["G7a"] = {"n90": sig_row("PCA n90 (1 ms): het vs hom", paired_triple(V(het, "n90_1ms"), V(hom, "n90_1ms"), "n90")),
                "pr": sig_row("Participation ratio (1 ms): het vs hom", paired_triple(V(het, "pr_1ms"), V(hom, "pr_1ms"), "pr")),
                "top5": sig_row("Top-5 variance: het vs hom", paired_triple(V(het, "top5_1ms"), V(hom, "top5_1ms"), "top5", higher_is_better=False))}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
# reference het/hom mean scree faint (labelled, or a reader cannot tell what the dashed curves are)
for dd, col, nm in [(het, S.HET, "het (learned, reference)"), (hom, S.HOM, "hom (learned, reference)")]:
    m, _ = scree_ci(dd, SEEDS_ALL)
    axes[0].plot(x, np.clip(m, 1e-7, None), color=col, lw=1.2, alpha=0.55, ls="--", label=nm)
    cm = np.cumsum(np.stack([np.array(dd[s]["evr_1ms"]) for s in SEEDS_ALL]), 1).mean(0)
    axes[1].plot(x, cm, color=col, lw=1.2, alpha=0.55, ls="--", label=nm)
# scaled-init is a control alongside the fixed rungs, so it belongs on the all-networks panel too
_scree_nets = [(scaled, S.SCALED, "scaled", "D")] + [(fixed[t], FIXED_COL[t], RUNG_LABEL[t], "o") for t in RUNGS]
for dd, col, name, mk in _scree_nets:
    m, e = scree_ci(dd, SEEDS_ORIG)
    axes[0].plot(x, np.clip(m, 1e-7, None), color=col, marker=mk, ms=2, label=name)
    axes[0].fill_between(x, np.clip(m - e, 1e-7, None), m + e, color=col, alpha=0.15, lw=0)
    cm = np.cumsum(np.stack([np.array(dd[s]["evr_1ms"]) for s in SEEDS_ORIG]), 1)
    cmm = cm.mean(0); cme = st.t.ppf(0.975, 9) * cm.std(0, ddof=1) / np.sqrt(10)
    axes[1].plot(x, cmm, color=col, marker=mk, ms=2, label=name)
    axes[1].fill_between(x, cmm - cme, cmm + cme, color=col, alpha=0.15, lw=0)
axes[0].set_yscale("log"); axes[0].set_ylim(1e-6, 1.5)
axes[0].set_xlabel("principal component"); axes[0].set_ylabel("explained variance ratio")
axes[0].set_title("PCA scree, all network types"); axes[0].legend(fontsize=8, ncol=2)
axes[1].axhline(0.9, color=S.CHANCE, ls=":", lw=1.2)
axes[1].set_xlabel("principal component"); axes[1].set_ylabel("cumulative explained variance")
axes[1].set_title("Cumulative variance, all network types"); axes[1].legend(fontsize=8, ncol=2)
S.save(fig, FIGS / "G7b_pca_scree_fixed.png")
# n90 and participation ratio for every control, so the all-networks dimension table matches G7a's columns
rows = [sig_row("n90 scaled vs learned hom",
                paired_triple(V(scaled, "n90_1ms", SEEDS_ORIG), V(hom, "n90_1ms", SEEDS_ORIG), "n90"))]
pr_rows = [sig_row("PR scaled vs learned hom",
                   paired_triple(V(scaled, "pr_1ms", SEEDS_ORIG), V(hom, "pr_1ms", SEEDS_ORIG), "pr"))]
for t in RUNGS:
    rows.append(sig_row(f"n90 fixed {RUNG_LABEL[t]} vs learned hom",
                        paired_triple(V(fixed[t], "n90_1ms", SEEDS_ORIG), V(hom, "n90_1ms", SEEDS_ORIG), "n90")))
    pr_rows.append(sig_row(f"PR fixed {RUNG_LABEL[t]} vs learned hom",
                           paired_triple(V(fixed[t], "pr_1ms", SEEDS_ORIG), V(hom, "pr_1ms", SEEDS_ORIG), "pr")))
STATS["G7b"] = {"n90_vs_wdhom": rows, "pr_vs_hom": pr_rows,
                "n90_means": {"scaled": float(V(scaled, "n90_1ms", SEEDS_ORIG).mean()),
                              **{RUNG_LABEL[t]: float(V(fixed[t], "n90_1ms", SEEDS_ORIG).mean()) for t in RUNGS}}}

# =====================================================================================
# G9a PC-reconstruction het vs hom (95% CI);  G9b all types
# =====================================================================================
MG = het[SEEDS_ALL[0]]["M_GRID"]


def recon_ci(dd, seeds):
    P = np.stack([np.array(dd[s]["pc_curve"]) for s in seeds])
    m = P.mean(0); e = st.t.ppf(0.975, len(seeds) - 1) * P.std(0, ddof=1) / np.sqrt(len(seeds))
    return m, e


fig, ax = plt.subplots(figsize=(8, 5))
for dd, col, name in [(het, S.HET, "heterogeneous"), (hom, S.HOM, "homogeneous")]:
    m, e = recon_ci(dd, SEEDS_ALL)
    ax.plot(MG, m, color=col, marker="o", ms=4, label=name)
    ax.fill_between(MG, m - e, m + e, color=col, alpha=0.18, lw=0)
S.chance_line(ax)
ax.set_xscale("log", base=2); ax.set_xticks(MG); ax.set_xticklabels(MG)
ax.set_xlabel("number of principal components in the reconstruction")
ax.set_ylabel("task accuracy through the real readout")
ax.set_title("PC-reconstruction, het vs hom")
ax.legend(loc="lower right")
S.save(fig, FIGS / "G9a_pc_recon_hom_het.png")
STATS["G9a"] = {f"m={m}": sig_row(f"PC-recon m={m}: het vs hom", paired_triple(pcm(het, m, SEEDS_ALL), pcm(hom, m, SEEDS_ALL), f"m{m}"))
                for m in [4, 8, 16, 32, 64]}

# Seven curves with seven CI bands is unreadable, so the bands stay only on het and hom (the paired n=20
# claim); the controls are drawn as lines with error bars at the grid points, and the legend moves outside.
fig, ax = plt.subplots(figsize=(9.6, 5))
for dd, col, name, seeds in [(het, S.HET, "het", SEEDS_ALL), (hom, S.HOM, "hom", SEEDS_ALL)]:
    m, e = recon_ci(dd, seeds)
    ax.plot(MG, m, color=col, marker="o", ms=4, lw=2.4, label=name, zorder=5)
    ax.fill_between(MG, m - e, m + e, color=col, alpha=0.18, lw=0, zorder=1)
for dd, col, name, mk in [(scaled, S.SCALED, "scaled", "D")] + \
                         [(fixed[t], FIXED_COL[t], f"fixed {RUNG_LABEL[t]}", "s") for t in RUNGS]:
    m, e = recon_ci(dd, SEEDS_ORIG)
    ax.errorbar(MG, m, yerr=e, color=col, marker=mk, ms=3.4, lw=1.4, elinewidth=0.8, capsize=2,
                alpha=0.9, label=name, zorder=3)
S.chance_line(ax)
ax.set_xscale("log", base=2); ax.set_xticks(MG); ax.set_xticklabels(MG)
ax.set_xlabel("number of principal components")
ax.set_ylabel("task accuracy through the real readout")
ax.set_title("PC-reconstruction by network type")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
          title="bands: 95% CI (het, hom)\nbars: 95% CI (controls)", title_fontsize=7.5)
S.save(fig, FIGS / "G9b_pc_recon_all.png")
# Where each control's reconstruction curve saturates, against the learned hom at the same m.
STATS["G9b"] = {"ceiling": {"het": float(pcm(het, 64, SEEDS_ALL).mean()), "hom": float(pcm(hom, 64, SEEDS_ALL).mean()),
                            "scaled": float(pcm(scaled, 64, SEEDS_ORIG).mean()),
                            **{RUNG_LABEL[t]: float(pcm(fixed[t], 64, SEEDS_ORIG).mean()) for t in RUNGS}},
                "m16_vs_hom": [sig_row("PC-recon m=16: scaled vs learned hom",
                                       paired_triple(pcm(scaled, 16, SEEDS_ORIG), pcm(hom, 16, SEEDS_ORIG), "m16"))] +
                              [sig_row(f"PC-recon m=16: fixed {RUNG_LABEL[t]} vs learned hom",
                                       paired_triple(pcm(fixed[t], 16, SEEDS_ORIG), pcm(hom, 16, SEEDS_ORIG), "m16"))
                               for t in RUNGS],
                "m64_vs_hom": [sig_row("PC-recon m=64: scaled vs learned hom",
                                       paired_triple(pcm(scaled, 64, SEEDS_ORIG), pcm(hom, 64, SEEDS_ORIG), "m64"))] +
                              [sig_row(f"PC-recon m=64: fixed {RUNG_LABEL[t]} vs learned hom",
                                       paired_triple(pcm(fixed[t], 64, SEEDS_ORIG), pcm(hom, 64, SEEDS_ORIG), "m64"))
                               for t in RUNGS]}

json.dump(STATS, open(REPO / "docs" / "analysis_suite" / "presentation_stats.json", "w"), indent=1)
print("wrote presentation_stats.json and all G-figures to figs/presentation/")
