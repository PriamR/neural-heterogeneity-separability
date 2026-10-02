"""Build every figure and table for the paper from the cached per-network records, and write the report.

Figures and tables come from this one script so a number in the report and a number in a figure cannot drift
apart. Run extract_all.py first.

Run: python code/analysis_suite/build_report.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple, holm, signflip_p

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "analysis_suite"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_NEW = list(range(1001, 1011))
SEEDS_ALL = SEEDS_ORIG + SEEDS_NEW
RUNGS = [2.0, 3.5, 15.0, 35.0]
ANOM = {1004, 1007}

R = {}
for f in CACHE.glob("*.json"):
    r = json.load(open(f))
    R[r["name"]] = r
print(f"loaded {len(R)} network records")

# The 9 seeds whose original homogeneous net carried weight decay 1e-4 while its het twin used none. These
# have been retrained at weight decay 0 (the "nowd" checkpoints). The main comparison uses the clean homs;
# the original WD homs are retained as hom_wd only for the confound before/after (Section 2); every other
# comparison, controls included, uses the corrected no-WD hom.
CONFOUNDED_SEEDS = [202, 210, 340, 440, 550, 660, 710, 820, 930]

het = {s: R[f"4class_seed{s}_het_lognormal"] for s in SEEDS_ALL}
hom_wd = {s: R[f"4class_seed{s}_hom_geommean_lognormal"] for s in SEEDS_ALL}   # as originally trained
hom = {s: (R[f"4class_seed{s}_hom_geommean_nowd_lognormal"] if s in CONFOUNDED_SEEDS
           and f"4class_seed{s}_hom_geommean_nowd_lognormal" in R
           else R[f"4class_seed{s}_hom_geommean_lognormal"]) for s in SEEDS_ALL}   # clean where available
scaled = {s: R[f"4class_seed{s}_hom_scaledinit_inputrec_lognormal"] for s in SEEDS_ORIG
          if f"4class_seed{s}_hom_scaledinit_inputrec_lognormal" in R}
TAG = {2.0: "fixed2ms", 3.5: "fixed3p5ms", 15.0: "fixed15ms", 35.0: "fixed35ms"}
fixed = {t: {s: R[f"4class_seed{s}_hom_{TAG[t]}_lognormal"] for s in SEEDS_ORIG
             if f"4class_seed{s}_hom_{TAG[t]}_lognormal" in R} for t in RUNGS}

V = lambda d, k, seeds=SEEDS_ALL: np.array([d[s][k] for s in seeds], dtype=float)
tables = {}

# ---------------------------------------------------------------------------------------
# Regularisation audit. The paired design claims the only free difference within a seed is the hidden time
# constant. That is false for part of the set: in the original batch the homogeneous net was trained with L2
# weight decay 1e-4 while its heterogeneous twin got none. Read it straight off the checkpoints rather than
# trusting the recipe in seeded_runs_common.py, which says 0.0 for both.
# ---------------------------------------------------------------------------------------
import torch
CKDIR = REPO / "checkpoints" / "all_networks_4class_64n"


def prm_of(name, key):
    ck = torch.load(CKDIR / f"{name}.pt", map_location="cpu", weights_only=False)
    return dict(ck["prms"]).get(key), len(ck.get("history", {}).get("test_acc", []))


WD = {}
for s in SEEDS_ALL:
    WD[(s, "het")], _ = prm_of(het[s]["name"], "weight_decay")
    WD[(s, "hom")], _ = prm_of(hom[s]["name"], "weight_decay")
MATCHED = [s for s in SEEDS_ALL if WD[(s, "het")] == WD[(s, "hom")]]
CONFOUND = [s for s in SEEDS_ALL if WD[(s, "het")] != WD[(s, "hom")]]
print(f"\nregularisation audit (after retrain): {len(MATCHED)} matched pairs, {len(CONFOUND)} still confounded {CONFOUND}")
assert not CONFOUND, f"expected all pairs matched after the retrain, still confounded: {CONFOUND}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def stat_row(label, h, o, higher_better=True, fmt="{:.3f}"):
    r = paired_triple(h, o, label, higher_better)
    return r, [label, fmt.format(r["het_mean"]), fmt.format(r["hom_mean"]), f"{r['delta']:+.3f}",
               f"{r['dz']:+.2f}", S.fmt_p(r["t_p"]), S.fmt_p(r["w_p"], r["w_floor"]),
               S.fmt_p(r["perm_p"], r["perm_floor"]), f"{r['wins']}/{r['n']}"]


# =====================================================================================
# T4.1  THE SCOREBOARD
# =====================================================================================
print("\n--- T4.1 scoreboard ---")
pcm = lambda d, m, seeds=SEEDS_ALL: np.array([d[s]["pc_curve"][d[s]["M_GRID"].index(m)] for s in seeds])

SCORE = [
    ("Task accuracy", V(het, "acc"), V(hom, "acc"), "{:.3f}"),
    ("Active neurons", V(het, "active"), V(hom, "active"), "{:.2f}"),
    ("Mean firing rate", V(het, "mean_rate"), V(hom, "mean_rate"), "{:.4f}"),
    ("Recurrent gain (spectral radius)", V(het, "gain"), V(hom, "gain"), "{:.3f}"),
    ("PCA n90 (1 ms)", V(het, "n90_1ms"), V(hom, "n90_1ms"), "{:.2f}"),
    ("PCA participation ratio (1 ms)", V(het, "pr_1ms"), V(hom, "pr_1ms"), "{:.2f}"),
    ("PCA top-5 variance", V(het, "top5_1ms"), V(hom, "top5_1ms"), "{:.3f}"),
    ("PCA top-10 variance", V(het, "top10_1ms"), V(hom, "top10_1ms"), "{:.3f}"),
    ("Linear decoder accuracy", V(het, "lin_decode"), V(hom, "lin_decode"), "{:.3f}"),
    ("PC-reconstruction @ m=4", pcm(het, 4), pcm(hom, 4), "{:.3f}"),
    ("PC-reconstruction @ m=16", pcm(het, 16), pcm(hom, 16), "{:.3f}"),
    ("PC-reconstruction @ m=32", pcm(het, 32), pcm(hom, 32), "{:.3f}"),
]
rows, recs = [], []
for lbl, h, o, fmt in SCORE:
    r, row = stat_row(lbl, h, o, fmt=fmt)
    recs.append(r)
    rows.append(row)
hp = holm([r["perm_p"] for r in recs])
for row, a in zip(rows, hp):
    row.append(f"{a:.1e}" if a < 1e-3 else f"{a:.3f}")
tables["T4.1"] = md_table(
    ["metric", "het", "hom", "delta", "dz", "paired-t p", "Wilcoxon p", "sign-flip p", "wins/n", "Holm p"], rows)
print(tables["T4.1"])
scoreboard = {r["metric"]: r for r in recs}

# =====================================================================================
# F4.1  paired accuracy
# =====================================================================================
acc_h, acc_o = V(het, "acc"), V(hom, "acc")
r_acc = scoreboard["Task accuracy"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
S.paired_slopes(axes[0], acc_h, acc_o)
S.chance_line(axes[0])
axes[0].set_ylabel("task accuracy")
axes[0].set_title(f"Paired accuracy, n = {len(acc_h)}")
axes[0].annotate(f"het wins {r_acc['wins']}/{r_acc['n']}\ndelta = {r_acc['delta']:+.3f}\n"
                 f"sign-flip p = {S.fmt_p(r_acc['perm_p'], r_acc['perm_floor'])}",
                 xy=(0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=9, color=S.ACCENT)
axes[0].legend(loc="upper right")
d = acc_h - acc_o
o_ = np.argsort(d)
lbl = [str(SEEDS_ALL[i]) + ("*" if SEEDS_ALL[i] in ANOM else "") for i in o_]
axes[1].bar(range(len(d)), d[o_], color=[S.HET if v > 0 else S.HOM for v in d[o_]])
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xticks(range(len(d)))
axes[1].set_xticklabels(lbl, rotation=90, fontsize=7)
axes[1].set_ylabel("accuracy gain (het - hom)")
axes[1].set_xlabel("seed (sorted by gain; * = long-tau hom)")
axes[1].set_title("Per-seed accuracy gain")
S.save(fig, FIGS / "F4_1_paired_accuracy.png")

# =====================================================================================
# F4.2 + T4.2  tau solutions
# =====================================================================================
# Single panel: the two learned solutions. The former right panel (hom tau vs accuracy gain) was dropped; the
# hom-tau/gain relationship is weak (Spearman rho = -0.44, p = 0.05) and the scatter over-implied a dose-response.
fig, ax = plt.subplots(figsize=(8.2, 4.8))
allt = np.concatenate([np.array(het[s]["tau_all_ms"]) for s in SEEDS_ALL])
bins = np.logspace(np.log10(3), np.log10(210), 40)
ax.hist(allt, bins=bins, color=S.HET, alpha=0.75, label=f"het, pooled per-neuron tau (n = {allt.size})")
for s in SEEDS_ALL:
    ax.axvline(hom[s]["tau_median_ms"], color=S.HOM, alpha=0.5, lw=1.2)
ax.axvline(np.nan, color=S.HOM, lw=1.2, label="hom, single learned tau (one line per seed)")
ax.axvline(199.5, color=S.CHANCE, ls="--", lw=1.4, label="clip ceiling 199.5 ms")
ax.axvline(3.26, color=S.CHANCE, ls=":", lw=1.4, label="clip floor 3.26 ms")
ax.set_xscale("log")
ax.set_xlabel("membrane time constant (ms, log scale)")
ax.set_ylabel("number of hidden units")
ax.set_title("Learned time-constant solutions")
ax.legend(fontsize=8)
med_h = V(het, "tau_median_ms")
med_o = V(hom, "tau_median_ms")
S.save(fig, FIGS / "F4_2_tau_solutions.png")

rows = []
for s in SEEDS_ALL:
    h, o = het[s], hom[s]
    rows.append([str(s) + ("*" if s in ANOM else ""), f"{o['tau_median_ms']:.2f}", f"{h['tau_median_ms']:.2f}",
                 f"{h['tau_mean_ms']:.2f}", f"{h['tau_sd_ms']:.2f}", f"{h['tau_cv']:.2f}",
                 f"{h['tau_min_ms']:.1f} to {h['tau_max_ms']:.1f}", f"{h['acc']-o['acc']:+.3f}"])
tables["T4.2"] = md_table(["seed", "hom tau (ms)", "het median", "het mean", "het SD", "het CV",
                           "het range", "delta acc"], rows)

# =====================================================================================
# T4.3  the tau-diversity null
# =====================================================================================
gain_acc = acc_h - acc_o
rows = []
for lbl, x in [("het tau CV", V(het, "tau_cv")), ("het tau mean (ms)", V(het, "tau_mean_ms")),
               ("het tau SD (ms)", V(het, "tau_sd_ms")), ("het tau median (ms)", med_h),
               ("hom tau (ms)", med_o)]:
    r = np.corrcoef(x, gain_acc)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
    from scipy import stats as _st
    p = 2 * (1 - _st.t.cdf(abs(t), n - 2))
    rows.append([lbl, f"{r:+.3f}", f"{p:.3f}", "n.s." if p >= 0.05 else S.sig_stars(p)])
tables["T4.3"] = md_table(["predictor", "r with accuracy gain", "p", ""], rows)
print("\n--- T4.3 tau-diversity null ---")
print(tables["T4.3"])

# =====================================================================================
# F4.3  recruitment, now a null (both networks matched on training)
# =====================================================================================
act_h, act_o = V(het, "active"), V(hom, "active")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
# Paired slopes were unreadable here: with the two networks level the connecting lines cross at random and the
# null looks like noise rather than like a null. Sorted per-seed differences about zero show it directly.
r_act = scoreboard["Active neurons"]
d_act = act_h - act_o
order = np.argsort(d_act)
d_sorted = d_act[order]
bar_cols = [S.HET if v > 0 else S.HOM for v in d_sorted]
axes[0].bar(np.arange(len(d_sorted)), d_sorted, color=bar_cols, zorder=3)
axes[0].axhline(0, color="0.25", lw=1.0, zorder=4)
axes[0].axhline(d_act.mean(), color=S.ACCENT, ls="--", lw=1.3, zorder=5,
                label=f"mean {d_act.mean():+.2f} (n.s., $p$ = {r_act['perm_p']:.3f})")
axes[0].set_xlabel("seed, sorted by difference")
axes[0].set_xticks(np.arange(0, len(d_sorted), 4))
axes[0].set_ylabel("active units, het $-$ hom")
axes[0].set_title("Recruitment difference per seed")
axes[0].legend(loc="upper left", fontsize=9)
xs = np.concatenate([act_h, act_o])
ys = np.concatenate([acc_h, acc_o])
cols = [S.HET] * len(act_h) + [S.HOM] * len(act_o)
axes[1].scatter(xs, ys, c=cols, s=30, zorder=3)
rr = np.corrcoef(xs, ys)[0, 1]
z = np.polyfit(xs, ys, 1)
xg = np.linspace(xs.min(), xs.max(), 50)
axes[1].plot(xg, np.polyval(z, xg), color=S.ACCENT, ls="--", lw=1.3, label=f"r = {rr:.2f}")
S.chance_line(axes[1])
axes[1].set_xlabel("active hidden units")
axes[1].set_ylabel("task accuracy")
axes[1].set_title("Active neurons vs accuracy")
axes[1].legend()
S.save(fig, FIGS / "F4_3_recruitment.png")
r_active_acc = rr

# =====================================================================================
# F4.7 + T4.6  corrected PCA scree
# =====================================================================================
E = {a: np.stack([np.array(d[s]["evr_1ms"]) for s in SEEDS_ALL]) for a, d in [("het", het), ("hom", hom)]}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(1, 65)
for a in ["het", "hom"]:
    m, sd = E[a].mean(0), E[a].std(0)
    S.band(axes[0], x, np.clip(m, 1e-7, None), sd, S.COL[a], label=S.LABEL[a], marker="o", ms=2.5)
    cm = np.cumsum(E[a], 1)
    S.band(axes[1], x, cm.mean(0), cm.std(0), S.COL[a], label=S.LABEL[a], marker="o", ms=2.5)
axes[0].set_yscale("log")
axes[0].set_ylim(1e-6, 1.5)
axes[0].set_xlabel("principal component")
axes[0].set_ylabel("explained variance ratio")
axes[0].set_title("Variance decay, time-resolved [(trials x time) x 64] at 1 ms")
axes[0].legend()
axes[1].axhline(0.9, color=S.CHANCE, ls=":", lw=1.2)
axes[1].set_xlabel("principal component")
axes[1].set_ylabel("cumulative explained variance")
n9h, n9o = V(het, "n90_1ms"), V(hom, "n90_1ms")
axes[1].set_title(f"Cumulative variance (n90: het {n9h.mean():.1f}, hom {n9o.mean():.1f})")
axes[1].legend()
S.save(fig, FIGS / "F4_7_pca_scree.png")

rows = []
for lbl, k, fmt in [("n90 (1 ms)", "n90_1ms", "{:.2f}"), ("participation ratio (1 ms)", "pr_1ms", "{:.2f}"),
                    ("top-5 variance", "top5_1ms", "{:.3f}"), ("top-10 variance", "top10_1ms", "{:.3f}"),
                    ("n90 (4 ms stride)", "n90_4ms", "{:.2f}"), ("participation ratio (4 ms)", "pr_4ms", "{:.2f}")]:
    _, row = stat_row(lbl, V(het, k), V(hom, k), fmt=fmt)
    rows.append(row)
tables["T4.6"] = md_table(["metric", "het", "hom", "delta", "dz", "paired-t p", "Wilcoxon p",
                           "sign-flip p", "wins/n"], rows)

# =====================================================================================
# F4.8 + T4.7  PC reconstruction, the spine
# =====================================================================================
MG = het[SEEDS_ALL[0]]["M_GRID"]
P = {a: np.stack([np.array(d[s]["pc_curve"]) for s in SEEDS_ALL]) for a, d in [("het", het), ("hom", hom)]}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for a in ["het", "hom"]:
    S.band(axes[0], MG, P[a].mean(0), P[a].std(0), S.COL[a], label=S.LABEL[a], marker="o", ms=4)
S.chance_line(axes[0])
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(MG)
axes[0].set_xticklabels(MG)
axes[0].set_xlabel("number of principal components in the reconstruction")
axes[0].set_ylabel("task accuracy through the real readout")
axes[0].set_title("PC-reconstruction accuracy, 1 ms, all 2,264 test trials")
axes[0].legend(loc="lower right")
gapc = P["het"].mean(0) - P["hom"].mean(0)
axes[1].plot(MG, gapc, color=S.ACCENT, marker="o", ms=4)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(MG)
axes[1].set_xticklabels(MG)
axes[1].set_xlabel("number of principal components")
axes[1].set_ylabel("het advantage (het - hom)")
# the meaningful crossover is where het takes a DURABLE lead: the last m at which hom is still ahead, +1 step.
# (the gap is briefly positive at m=1, dips negative through the m=4 hom reversal, then turns over for good.)
neg_idx = [i for i in range(len(MG)) if gapc[i] < 0]
cross = MG[max(neg_idx) + 1] if neg_idx and max(neg_idx) + 1 < len(MG) else MG[0]
if cross:
    axes[1].axvline(cross, color=S.CHANCE, ls="--", lw=1.2)
    axes[1].annotate(f"het takes the lead\nat m = {cross}", xy=(cross, 0), xytext=(8, -28),
                     textcoords="offset points", fontsize=9, color=S.ACCENT)
    axes[1].annotate("hom ahead in\nlow-D (m = 4)", xy=(4, gapc[MG.index(4)]), xytext=(-4, -34),
                     textcoords="offset points", fontsize=8, color=S.HOM, ha="center")
axes[1].set_title("Het minus hom accuracy vs number of PCs")
S.save(fig, FIGS / "F4_8_pc_reconstruction.png")


def pcs_to_ceiling(curve, frac=0.9):
    ceil = curve[-1]
    thr = 0.25 + frac * (ceil - 0.25)
    return next((MG[i] for i in range(len(MG)) if curve[i] >= thr), 64)


rows = []
for m in [4, 8, 12, 16, 24, 32, 64]:
    _, row = stat_row(f"m = {m}", pcm(het, m), pcm(hom, m))
    rows.append(row)
p90h = np.array([pcs_to_ceiling(het[s]["pc_curve"]) for s in SEEDS_ALL], float)
p90o = np.array([pcs_to_ceiling(hom[s]["pc_curve"]) for s in SEEDS_ALL], float)
_, row = stat_row("PCs to 90% of own ceiling", p90h, p90o, higher_better=True, fmt="{:.2f}")
rows.append(row)
tables["T4.7"] = md_table(["quantity", "het", "hom", "delta", "dz", "paired-t p", "Wilcoxon p",
                           "sign-flip p", "wins/n"], rows)

# =====================================================================================
# F4.9  separability vs accuracy
# =====================================================================================
fig, ax = plt.subplots(figsize=(6.4, 4.8))
allnets = [(het, S.HET, "heterogeneous"), (hom, S.HOM, "homogeneous")]
xs, ys = [], []
for d, c, lab in allnets:
    x_ = V(d, "lin_decode")
    y_ = V(d, "acc")
    ax.scatter(x_, y_, color=c, s=30, zorder=3, label=lab)
    xs.append(x_)
    ys.append(y_)
if scaled:
    ss = sorted(scaled)
    x_ = np.array([scaled[s]["lin_decode"] for s in ss])
    y_ = np.array([scaled[s]["acc"] for s in ss])
    ax.scatter(x_, y_, color=S.SCALED, s=30, marker="^", zorder=3, label="scaled-init hom")
    xs.append(x_)
    ys.append(y_)
xs, ys = np.concatenate(xs), np.concatenate(ys)
r_sep = np.corrcoef(xs, ys)[0, 1]
z = np.polyfit(xs, ys, 1)
xg = np.linspace(xs.min(), xs.max(), 50)
ax.plot(xg, np.polyval(z, xg), color=S.ACCENT, ls="--", lw=1.3, label=f"r = {r_sep:.2f}")
S.chance_line(ax)
ax.set_xlabel("linear decodability of class from the rate code (5-fold)")
ax.set_ylabel("task accuracy")
ax.set_title("Linear decodability vs accuracy")
ax.legend(loc="lower right")
S.save(fig, FIGS / "F4_9_separability_vs_accuracy.png")

# =====================================================================================
# F4.10 + T4.8  THE BOWL
# =====================================================================================
print("\n--- T4.8 fixed-tau ladder ---")
# Two-panel comparison. Left: the bowl as a clean line, each rung mean +/- SD, with the learned het and
# learned hom means as reference lines and bands. Right: per-seed trajectories across [het, hom, the four
# fixed rungs], one line per original seed, so no seed's fixed arm reaching its own het is visible directly.
# The controls are compared to the corrected (no-WD) learned hom, the same hom used everywhere else.
fseeds = [s for s in SEEDS_ORIG if all(s in fixed[t] for t in RUNGS)]
rung_m = [np.array([fixed[t][s]["acc"] for s in fseeds]).mean() for t in RUNGS]
rung_sd = [np.array([fixed[t][s]["acc"] for s in fseeds]).std(ddof=1) for t in RUNGS]
het_ref_m, het_ref_sd = acc_h.mean(), acc_h.std(ddof=1)
acc_o_clean = V(hom, "acc", fseeds)
hom_ref_m, hom_ref_sd = acc_o_clean.mean(), acc_o_clean.std(ddof=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
# --- left: clean bowl
ax = axes[0]
xg = [1.7, 60]
ax.fill_between(xg, het_ref_m - het_ref_sd, het_ref_m + het_ref_sd, color=S.HET, alpha=0.10, lw=0, zorder=1)
ax.axhline(het_ref_m, color=S.HET, lw=2.0, zorder=3, label=f"learned het (n={len(acc_h)}): {het_ref_m:.3f}")
ax.fill_between(xg, hom_ref_m - hom_ref_sd, hom_ref_m + hom_ref_sd, color=S.HOM, alpha=0.10, lw=0, zorder=1)
ax.axhline(hom_ref_m, color=S.HOM, lw=2.0, ls="--", zorder=3,
           label=f"learned hom (n={len(fseeds)}): {hom_ref_m:.3f}")
jr = np.random.default_rng(0)
# The nominal 2 ms rung clips at the 3.26 ms floor, so it is plotted where it actually sits. That puts it
# almost on top of the 3.5 ms rung, which is the honest picture: those two rungs are one timescale trained
# twice, and the gap between them measures run-to-run spread at fixed tau rather than a timescale effect.
XPOS = {2.0: 3.26, 3.5: 3.5, 15.0: 15.0, 35.0: 35.0}
for t in RUNGS:
    a = np.array([fixed[t][s]["acc"] for s in fseeds])
    ax.scatter(XPOS[t] * (1 + jr.uniform(-0.03, 0.03, len(a))), a, color=S.FIXED_RAMP[t], s=20, alpha=0.55,
               zorder=2, edgecolor="none")
xs_rung = [XPOS[t] for t in RUNGS]
ax.plot(xs_rung, rung_m, color=S.ACCENT, lw=2.2, zorder=5)
ax.errorbar(xs_rung, rung_m, yerr=rung_sd, fmt="s", color=S.ACCENT, ms=8, mfc="white", mew=1.8, capsize=4,
            zorder=6, label=f"fixed-tau hom (mean +/- SD, n={len(fseeds)})")
S.chance_line(ax)
ax.set_xscale("log")
ax.set_xlim(2.6, 60)
ax.set_xticks([3.38, 15, 35])
ax.set_xticklabels(["3.26 and 3.5", "15", "35"])
ax.annotate("the nominal 2 ms rung clips to 3.26 ms,\nso these two rungs are one timescale", xy=(4.6, 0.295),
            ha="left", va="bottom", fontsize=7.5, color="#7a7a7a")
ax.set_ylim(0.20, 0.78)
ax.set_xlabel("hidden membrane time constant, fixed (ms, log scale)")
ax.set_ylabel("task accuracy")
ax.set_title("Accuracy vs fixed time constant")
_best = RUNGS[int(np.argmax(rung_m))]
ax.annotate(f"best rung {max(rung_m):.3f}\nat {_best:g} ms", xy=(_best, max(rung_m)), xytext=(0, -34),
            textcoords="offset points", ha="center", fontsize=8.5, color=S.ACCENT,
            arrowprops=dict(arrowstyle="->", color=S.ACCENT, lw=1))
ax.legend(loc="upper right", fontsize=8)
# --- right: per-seed trajectories
ax = axes[1]
xpos = np.arange(6)
Mtr = np.zeros((len(fseeds), 6))
for i, s in enumerate(fseeds):
    Mtr[i, 0] = het[s]["acc"]
    Mtr[i, 1] = hom[s]["acc"]
    for j, t in enumerate(RUNGS):
        Mtr[i, 2 + j] = fixed[t][s]["acc"]
ax.axvline(1.5, color=S.CHANCE, lw=0.8, alpha=0.4, zorder=0)
ax.text(0.5, 0.795, "learned", ha="center", va="top", fontsize=8, color=S.ACCENT, alpha=0.7)
ax.text(3.5, 0.795, "fixed tau", ha="center", va="top", fontsize=8, color=S.ACCENT, alpha=0.7)
for i in range(len(fseeds)):
    ax.plot(xpos, Mtr[i], color="#b8b8b8", lw=0.8, alpha=0.7, zorder=1)
ax.plot(xpos, Mtr.mean(0), color=S.ACCENT, lw=2.4, marker="o", ms=6, zorder=4, label="mean over seeds")
ax.scatter([0] * len(fseeds), Mtr[:, 0], color=S.HET, s=26, zorder=3)
ax.scatter([1] * len(fseeds), Mtr[:, 1], color=S.HOM, s=26, zorder=3)
for j, t in enumerate(RUNGS):
    ax.scatter([2 + j] * len(fseeds), Mtr[:, 2 + j], color=S.FIXED_RAMP[t], s=26, zorder=3)
S.chance_line(ax)
ax.set_xticks(xpos)
ax.set_xticklabels(["learned\nhet", "learned\nhom", "3.26*", "3.5", "15", "35"], fontsize=9)
ax.set_xlim(-0.4, 5.4)
ax.set_ylim(0.20, 0.82)
ax.set_xlabel("network (learned, then fixed tau in ms)")
ax.set_ylabel("task accuracy")
# Only the seeds that have the full fixed-tau set appear here, so this panel is n=10 throughout, including
# its het column; the left panel's het reference line is the full n=20.
ax.set_title(f"Per-seed accuracy: learned and fixed (n={len(fseeds)} control seeds)")
ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Fixed-tau ladder", fontweight="bold", y=1.01)
S.save(fig, FIGS / "F4_10_fixed_tau_bowl.png")

rows = []
for t in RUNGS:
    if not fixed[t]:
        continue
    ss = sorted(fixed[t])
    a = np.array([fixed[t][s]["acc"] for s in ss])
    ac = np.array([fixed[t][s]["active"] for s in ss], float)
    n9 = np.array([fixed[t][s]["n90_1ms"] for s in ss], float)
    pr_ = np.array([fixed[t][s]["pr_1ms"] for s in ss], float)
    g_ = np.array([fixed[t][s]["gain"] for s in ss], float)
    b = np.array([hom[s]["acc"] for s in ss])    # corrected (no-WD) learned hom baseline
    r = paired_triple(a, b, f"fixed {t:g} ms vs learned hom")
    # the nominal 2 ms rung trains at the 3.26 ms clip floor, so the table states the constant it actually ran at
    rows.append([f"{XPOS[t]:g}" + ("*" if XPOS[t] != t else ""),
                 f"{a.mean():.3f} +/- {a.std(ddof=1):.3f}", f"{ac.mean():.1f}", f"{n9.mean():.1f}",
                 f"{pr_.mean():.2f}", f"{g_.mean():.2f}", f"{r['delta']:+.3f}", f"{r['dz']:+.2f}",
                 S.fmt_p(r["t_p"]), S.fmt_p(r["w_p"], r["w_floor"]), S.fmt_p(r["perm_p"], r["perm_floor"]),
                 f"{r['wins']}/{r['n']}"])
tables["T4.8"] = md_table(["fixed tau (ms)", "accuracy", "active", "n90", "PR", "gain",
                           "delta vs learned hom", "dz", "paired-t p", "Wilcoxon p", "sign-flip p", "wins/n"], rows)
print(tables["T4.8"])

# =====================================================================================
# F4.12 + T4.9  the scaled-init dissociation
# =====================================================================================
if scaled:
    print("\n--- T4.9 scaled-init ---")
    ss = sorted(scaled)
    cons = json.load(open(REPO / "data_cache" / "init_scaling" / "ratematch_constants_inputrec.json"))
    per_seed = {int(k): v for k, v in cons.get("per_seed", cons).items()}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    PANELS = [("mean_rate", "mean firing rate", "{:.3f}"), ("active", "active hidden units", "{:.0f}"),
              ("acc", "task accuracy", "{:.3f}"), ("pr_1ms", "participation ratio", "{:.1f}")]
    for ax, (k, lab, _f) in zip(axes, PANELS):
        # the hom bar is the corrected (no-WD) learned hom, the same hom used everywhere else
        vals = [np.array([het[s][k] for s in ss], float), np.array([hom[s][k] for s in ss], float),
                np.array([scaled[s][k] for s in ss], float)]
        for i, (v, c) in enumerate(zip(vals, [S.HET, S.HOM, S.SCALED])):
            ax.bar(i, v.mean(), yerr=v.std(ddof=1), color=c, width=0.62, zorder=2)
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.13, .13, len(v)), v,
                       color="k", s=9, alpha=0.55, zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["het", "hom", "scaled\nhom"], fontsize=9)
        ax.set_title(lab)
        if k == "acc":
            S.chance_line(ax)
    fig.suptitle("Scaled-init control: het, hom, scaled-hom", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "F4_12_scaled_init_dissociation.png")

    rows = []
    for s in ss:
        rows.append([str(s), f"{per_seed.get(s, float('nan')):.2f}", f"{het[s]['acc']:.3f}", f"{hom[s]['acc']:.3f}",
                     f"{scaled[s]['acc']:.3f}", f"{scaled[s]['acc']-hom[s]['acc']:+.3f}",
                     f"{scaled[s]['active']}", f"{scaled[s]['mean_rate']:.3f}",
                     f"{scaled[s]['n90_1ms']}", f"{scaled[s]['pr_1ms']:.1f}", f"{scaled[s]['gain']:.2f}"])
    tables["T4.9"] = md_table(["seed", "c", "het acc", "hom acc", "scaled acc", "delta vs hom",
                               "scaled active", "scaled rate", "scaled n90", "scaled PR", "scaled gain"], rows)
    rows = []
    for lbl, k, fmt in [("Task accuracy", "acc", "{:.3f}"), ("Mean firing rate", "mean_rate", "{:.4f}"),
                        ("Active neurons", "active", "{:.1f}"), ("PCA n90", "n90_1ms", "{:.1f}"),
                        ("Participation ratio", "pr_1ms", "{:.2f}"), ("Recurrent gain", "gain", "{:.2f}")]:
        a = np.array([scaled[s][k] for s in ss], float)
        b = np.array([hom[s][k] for s in ss], float)   # corrected (no-WD) learned hom baseline
        r = paired_triple(a, b, lbl)
        rows.append([lbl, fmt.format(r["het_mean"]), fmt.format(r["hom_mean"]), f"{r['delta']:+.3f}",
                     f"{r['dz']:+.2f}", S.fmt_p(r["t_p"]), S.fmt_p(r["w_p"], r["w_floor"]),
                     S.fmt_p(r["perm_p"], r["perm_floor"]), f"{r['wins']}/{r['n']}"])
    tables["T4.9b"] = md_table(["metric (scaled vs learned hom)", "scaled", "hom", "delta", "dz",
                                "paired-t p", "Wilcoxon p", "sign-flip p", "scaled wins/n"], rows)
    print(tables["T4.9b"])

# =====================================================================================
# T4.10  gain
# =====================================================================================
g_h, g_o = V(het, "gain"), V(hom, "gain")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
S.paired_slopes(axes[0], g_h, g_o)
axes[0].axhline(1.0, color=S.CHANCE, ls=":", lw=1.2, label="unit gain")
r_g = scoreboard["Recurrent gain (spectral radius)"]
axes[0].set_ylabel("recurrent spectral radius")
axes[0].set_title("Recurrent gain, het vs hom")
axes[0].legend(loc="upper right")
xs2 = np.concatenate([g_h, g_o])
ys2 = np.concatenate([acc_h, acc_o])
c2 = [S.HET] * len(g_h) + [S.HOM] * len(g_o)
if scaled:
    xs2 = np.concatenate([xs2, [scaled[s]["gain"] for s in ss]])
    ys2 = np.concatenate([ys2, [scaled[s]["acc"] for s in ss]])
    c2 += [S.SCALED] * len(ss)
axes[1].scatter(xs2, ys2, c=c2, s=30, zorder=3)
r_gain_acc = np.corrcoef(xs2, ys2)[0, 1]
S.chance_line(axes[1])
axes[1].set_xlabel("recurrent spectral radius")
axes[1].set_ylabel("task accuracy")
axes[1].set_title(f"Recurrent gain vs accuracy (r = {r_gain_acc:.2f})")
S.save(fig, FIGS / "T4_10_gain.png")

rows = []
for lbl, arr in [("het", g_h), ("hom", g_o)] + ([("scaled hom", np.array([scaled[s]["gain"] for s in ss]))] if scaled else []):
    rows.append([lbl, f"{arr.mean():.3f}", f"{arr.std(ddof=1):.3f}", f"{arr.min():.2f} to {arr.max():.2f}",
                 f"{int((arr > 1).sum())}/{len(arr)}"])
tables["T4.10"] = md_table(["network", "mean spectral radius", "SD", "range", "nets above unit gain"], rows)

# =====================================================================================
# A.8  full per-seed table
# =====================================================================================
rows = []
for s in SEEDS_ALL:
    h, o = het[s], hom[s]
    rows.append([str(s) + ("*" if s in ANOM else ""),
                 f"{h['acc']:.3f}", f"{o['acc']:.3f}", f"{h['acc']-o['acc']:+.3f}",
                 f"{h['active']}", f"{o['active']}", f"{h['n90_1ms']}", f"{o['n90_1ms']}",
                 f"{h['pr_1ms']:.1f}", f"{o['pr_1ms']:.1f}", f"{h['gain']:.2f}", f"{o['gain']:.2f}",
                 f"{h['tau_median_ms']:.1f}", f"{o['tau_median_ms']:.1f}"])
tables["A8"] = md_table(["seed", "het acc", "hom acc", "delta", "het act", "hom act", "het n90", "hom n90",
                         "het PR", "hom PR", "het gain", "hom gain", "het tau med", "hom tau"], rows)

# =====================================================================================
# batch check: do the two training batches differ?
# =====================================================================================
print("\n--- batch check ---")
rows = []
for lbl, k, fmt in [("het accuracy", "acc", "{:.3f}"), ("hom accuracy", "acc", "{:.3f}"),
                    ("het active", "active", "{:.1f}"), ("hom active", "active", "{:.1f}"),
                    ("het gain", "gain", "{:.2f}"), ("hom gain", "gain", "{:.2f}"),
                    ("het n90", "n90_1ms", "{:.1f}"), ("hom n90", "n90_1ms", "{:.1f}")]:
    d = het if lbl.startswith("het") else hom
    a = np.array([d[s][k] for s in SEEDS_ORIG], float)
    b = np.array([d[s][k] for s in SEEDS_NEW], float)
    from stats_core import unpaired_triple
    r = unpaired_triple(a, b, lbl)
    rows.append([lbl, fmt.format(a.mean()), fmt.format(b.mean()), f"{r['delta']:+.2f}",
                 S.fmt_p(r["welch_p"]), S.fmt_p(r["mw_p"])])
tables["BATCH"] = md_table(["metric", "original batch (10)", "new batch (10)", "delta", "Welch p", "Mann-Whitney p"], rows)
print(tables["BATCH"])

# =====================================================================================
# The regularisation confound, before and after the retrain. On the 9 seeds whose homogeneous net originally
# carried weight decay 1e-4, compare that WD net against its no-WD retrain (both share the het twin's init).
# This shows what the penalty did, and confirms the corrected homs are what the clean scoreboard uses.
# =====================================================================================
print("\n--- confound before/after (9 retrained seeds) ---")
rows = []
for lbl, k, fmt in [("Task accuracy", "acc", "{:.3f}"), ("Active neurons", "active", "{:.1f}"),
                    ("Recurrent gain", "gain", "{:.2f}"), ("PCA n90 (1 ms)", "n90_1ms", "{:.1f}"),
                    ("PCA participation ratio", "pr_1ms", "{:.2f}"), ("Mean firing rate", "mean_rate", "{:.4f}")]:
    wdv = V(hom_wd, k, CONFOUNDED_SEEDS)
    nwv = V(hom, k, CONFOUNDED_SEEDS)
    r = paired_triple(nwv, wdv, lbl)     # no-WD minus WD
    rows.append([lbl, fmt.format(wdv.mean()), fmt.format(nwv.mean()), f"{nwv.mean()-wdv.mean():+.3f}",
                 f"{r['dz']:+.2f}", S.fmt_p(r["perm_p"], r["perm_floor"]), f"{r['wins']}/{r['n']}"])
tables["CONFOUND"] = md_table(["metric (9 retrained seeds)", "hom with WD 1e-4", "hom no WD (retrain)",
                               "change", "dz", "sign-flip p", "no-WD higher in"], rows)
print(tables["CONFOUND"])

# how the het gap looks against each hom version, on the 9 seeds
gap_rows = []
for lbl, k, fmt in [("Task accuracy", "acc", "{:.3f}"), ("Active neurons", "active", "{:.1f}"),
                    ("Recurrent gain", "gain", "{:.2f}"), ("PCA n90 (1 ms)", "n90_1ms", "{:.1f}")]:
    rw = paired_triple(V(het, k, CONFOUNDED_SEEDS), V(hom_wd, k, CONFOUNDED_SEEDS), lbl)
    rn = paired_triple(V(het, k, CONFOUNDED_SEEDS), V(hom, k, CONFOUNDED_SEEDS), lbl)
    gap_rows.append([lbl, f"{rw['delta']:+.3f}", f"{rw['wins']}/{rw['n']}",
                     f"{rn['delta']:+.3f}", f"{rn['wins']}/{rn['n']}"])
tables["CONFOUND_GAP"] = md_table(["metric", "het - hom(WD) delta", "wins", "het - hom(noWD) delta", "wins"], gap_rows)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (k, lab, fmt) in zip(axes, [("acc", "task accuracy", "{:.3f}"), ("gain", "recurrent gain", "{:.2f}"),
                                    ("active", "active hidden units", "{:.0f}")]):
    het9 = V(het, k, CONFOUNDED_SEEDS)
    wd9 = V(hom_wd, k, CONFOUNDED_SEEDS)
    nw9 = V(hom, k, CONFOUNDED_SEEDS)
    for i, (v, c, nm) in enumerate([(het9, S.HET, "het"), (wd9, S.HOM, "hom (WD 1e-4)"),
                                    (nw9, S.SCALED, "hom (no WD, retrain)")]):
        ax.bar(i, v.mean(), yerr=v.std(ddof=1), color=c, width=0.66, zorder=2, label=nm)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-.12, .12, len(v)), v,
                   color="k", s=9, alpha=0.5, zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["het", "hom\nWD", "hom\nno WD"], fontsize=8)
    ax.set_title(lab)
    if k == "acc":
        S.chance_line(ax)
fig.suptitle("Het, weight-decay hom, and no-WD retrain (9 seeds)", fontweight="bold", y=1.03)
S.save(fig, FIGS / "F4_0_regularisation_confound.png")

conf_summary = {}
for k in ["acc", "active", "gain", "n90_1ms", "pr_1ms", "mean_rate"]:
    conf_summary[k] = {
        "wd": paired_triple(V(het, k, CONFOUNDED_SEEDS), V(hom_wd, k, CONFOUNDED_SEEDS), k),
        "nowd": paired_triple(V(het, k, CONFOUNDED_SEEDS), V(hom, k, CONFOUNDED_SEEDS), k),
        "before_after": paired_triple(V(hom, k, CONFOUNDED_SEEDS), V(hom_wd, k, CONFOUNDED_SEEDS), k),
    }

json.dump({k: v for k, v in tables.items()}, open(OUT / "tables.json", "w"), indent=1)
summary = {
    "n_pairs": len(SEEDS_ALL),
    "acc": {"het": float(acc_h.mean()), "hom": float(acc_o.mean()), "delta": float((acc_h - acc_o).mean()),
            "wins": int(r_acc["wins"]), "perm_p": float(r_acc["perm_p"]), "perm_floor": float(r_acc["perm_floor"]),
            "dz": float(r_acc["dz"]), "t_p": float(r_acc["t_p"]), "w_p": float(r_acc["w_p"])},
    "r_active_acc": float(r_active_acc),
    "r_sep_acc": float(r_sep),
    "r_gain_acc": float(r_gain_acc),
    "pc_crossover": cross,
    "pcs_to_ceiling": {"het": float(p90h.mean()), "hom": float(p90o.mean())},
    "pc_ceiling": {"het": float(P["het"].mean(0)[-1]), "hom": float(P["hom"].mean(0)[-1])},
    "scoreboard": {k: v for k, v in scoreboard.items()},
    "all_matched": len(CONFOUND) == 0,
    "retrained_seeds": CONFOUNDED_SEEDS,
    "wd": {f"{s}_{t}": WD[(s, t)] for s in SEEDS_ALL for t in ("het", "hom")},
    "confound": conf_summary,
    "n_trials": int(het[SEEDS_ALL[0]]["n_trials"]),
}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print(f"\nwrote {OUT/'tables.json'} and {OUT/'summary.json'}")
print(f"figures in {FIGS}")
