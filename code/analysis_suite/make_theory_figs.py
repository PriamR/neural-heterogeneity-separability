"""Figures for the minimal tier of the temporal-basis account (the mechanism appendix), plus a theory-vs-data
check that is not in the paper.

Everything here follows from the proven proposition and needs no fitting:
  C_ij = sech((ln tau_i - ln tau_j) / 2)          filter correlation
  PR(C) = N^2 / (N + 2 sum_{i<j} C_ij^2)          effective dimension

  F5.1  the filter bank: identical kernels against a spread of widths (illustrative tau values)
  F5.2  the sech correlation kernel against log-tau distance
  F5.3  PR(s) as a real learned configuration is dilated about its mean, from 1 toward N
  F5.4  minimal-tier prediction against measured dimensionality, per network (not in the paper)

Reads the per-network records in data_cache/network_json.
Run: python code/analysis_suite/make_theory_figs.py
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
CACHE = REPO / "data_cache" / "network_json"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
CONFOUNDED = [202, 210, 340, 440, 550, 660, 710, 820, 930]   # use the no-WD retrain for these
N = 64


def hom_json(s):
    """The clean homogeneous record: the no-WD retrain for the originally-confounded seeds."""
    p = CACHE / f"4class_seed{s}_hom_geommean_nowd_lognormal.json"
    if s in CONFOUNDED and p.exists():
        return json.load(open(p))
    return json.load(open(CACHE / f"4class_seed{s}_hom_geommean_lognormal.json"))


def corr_matrix(tau):
    u = np.log(np.asarray(tau, dtype=np.float64))
    return 1.0 / np.cosh((u[:, None] - u[None, :]) / 2.0)


def pr_bank(tau):
    C = corr_matrix(tau)
    n = C.shape[0]
    return (n ** 2) / (n + 2 * np.triu(C ** 2, 1).sum())


# ---------------------------------------------------------------- F5.1 the filter bank
t = np.linspace(0, 300, 900)
HOM_TAU = 5.0
HET_TAUS = np.geomspace(3.3, 199.5, 6)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
fig.subplots_adjust(wspace=0.12)
for ax, taus, letter, title, col in [
    (axes[0], [HOM_TAU] * 6, "a", f"Homogeneous population, $\\tau$ = {HOM_TAU:g} ms", S.HOM),
    (axes[1], HET_TAUS, "b", "Heterogeneous population, $\\tau$ from 3.3 to 199.5 ms", S.HET),
]:
    for i, tt in enumerate(taus):
        k = np.exp(-t / tt) / tt
        ax.plot(t, k / k.max(), color=col, alpha=0.35 + 0.11 * i, lw=1.8)
    ax.set_xlabel("Time since input (ms)")
    S.panel_title(ax, letter, title)
axes[0].set_ylabel("Normalised kernel $k_\\tau(t)$")
S.save(fig, FIGS / "F5_1_filter_bank.png")

# ---------------------------------------------------------------- F5.2 the sech kernel
d = np.linspace(0, 8, 400)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.plot(d, 1 / np.cosh(d / 2), color=S.ACCENT, lw=2.2)
# the three ratios annotated are the ones the appendix quotes: 10x is "an order of magnitude apart",
# 35x is the ratio that reaches a third
for r, lab in [(np.log(2), "Ratio 2"), (np.log(10), "Ratio 10"), (np.log(35), "Ratio 35")]:
    S.dots(ax, [r], [1 / np.cosh(r / 2)], S.HET, s=40)
    ax.annotate(f"{lab}: $C$ = {1/np.cosh(r/2):.2f}", (r, 1 / np.cosh(r / 2)),
                xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=S.ACCENT)
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"Log-$\tau$ distance, $|\ln\tau_i - \ln\tau_j|$")
ax.set_ylabel(r"Filter correlation, $C_{ij}$")
S.panel_title(ax, None, r"Filter correlation against log-$\tau$ distance")
S.save(fig, FIGS / "F5_2_sech_kernel.png")

# ---------------------------------------------------------------- F5.3 PR(s) under dilation
tau_ref = np.array(json.load(open(CACHE / "4class_seed202_het_lognormal.json"))["tau_all_ms"])
u = np.log(tau_ref)
ubar = u.mean()
svals = np.linspace(0, 20, 400)
prs = np.array([pr_bank(np.exp(ubar + s * (u - ubar))) for s in svals])
pr1 = pr_bank(tau_ref)
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(svals, prs, color=S.ACCENT, lw=2.2, zorder=3)
S.ref_line(ax, y=N, label=f"$N$ = {N}, fully decorrelated bank", ls=(0, (4, 2)))
S.ref_line(ax, y=1, label="PR = 1, homogeneous population ($s$ = 0)", ls=(0, (1, 2)))
S.dots(ax, [1.0], [pr1], S.HET, s=60, zorder=5)
ax.annotate(f"Learned spread, $s$ = 1: PR = {pr1:.1f}", (1.0, pr1), xytext=(10, -2), textcoords="offset points",
            fontsize=8.5, color=S.ACCENT, va="center")
ax.set_yscale("log")
ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64"])
ax.set_ylim(0.85, 80)
ax.set_xlim(0, 20)
ax.set_xlabel("Dilation of the log-$\\tau$ configuration about its mean, $s$")
ax.set_ylabel("Effective dimension of the filter bank, PR (log scale)")
S.panel_title(ax, None, "Effective dimension of the filter bank under dilation")
S.legend(ax, loc="lower right")
S.save(fig, FIGS / "F5_3_pr_vs_spread.png")
print(f"  PR at s=1 (seed 202): {pr1:.2f}; at s=6: {np.interp(6, svals, prs):.2f}; at s=20: {prs[-1]:.2f}")

# ---------------------------------------------------------------- F5.4 minimal tier vs data (not in the paper)
rows = []
for s in SEEDS:
    h = json.load(open(CACHE / f"4class_seed{s}_het_lognormal.json"))
    o = hom_json(s)
    rows.append(dict(seed=s,
                     het_pred=pr_bank(np.array(h["tau_all_ms"])), het_pr=h["pr_1ms"], het_n90=h["n90_1ms"],
                     hom_pred=pr_bank(np.array(o["tau_all_ms"])), hom_pr=o["pr_1ms"], hom_n90=o["n90_1ms"]))
hp = np.array([r["het_pred"] for r in rows])
hm = np.array([r["het_pr"] for r in rows])
hn = np.array([r["het_n90"] for r in rows])
op = np.array([r["hom_pred"] for r in rows])
om = np.array([r["hom_pr"] for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
S.dots(ax, hp, hm, S.HET, label="Heterogeneous, measured PR")
S.dots(ax, hp, hn, S.HET, marker="^", alpha=0.6, label="Heterogeneous, measured $n_{90}$")
S.dots(ax, op, om, S.HOM, label="Homogeneous, measured PR")
lim = [0, 66]
ax.plot(lim, lim, color=S.CHANCE, ls=(0, (4, 2)), lw=1.2, label="Perfect prediction")
ax.set_xlim(lim)
ax.set_ylim(0, 40)
ax.set_xlabel("Predicted PR of the bare filter bank")
ax.set_ylabel("Measured dimensionality of the state")
S.panel_title(ax, "a", "Prediction against measurement")
S.legend(ax, loc="upper left")

ax = axes[1]
r_pred = np.corrcoef(hp, hn)[0, 1]
S.dots(ax, hp, hn, S.HET)
z = np.polyfit(hp, hn, 1)
xg = np.linspace(hp.min(), hp.max(), 40)
ax.plot(xg, np.polyval(z, xg), color=S.ACCENT, ls=(0, (4, 2)), lw=1.3, label=f"Linear fit, $r$ = {r_pred:.2f}")
ax.set_xlabel("Predicted PR of the bare filter bank")
ax.set_ylabel("Measured $n_{90}$, heterogeneous networks")
S.panel_title(ax, "b", "Ordering of the heterogeneous networks")
S.legend(ax)
S.save(fig, FIGS / "F5_4_theory_vs_data.png")

res = {
    "het_pred_pr_mean": float(hp.mean()), "het_measured_pr_mean": float(hm.mean()),
    "het_measured_n90_mean": float(hn.mean()),
    "hom_pred_pr_mean": float(op.mean()), "hom_measured_pr_mean": float(om.mean()),
    "r_pred_vs_n90_het": float(r_pred),
    "r_pred_vs_pr_het": float(np.corrcoef(hp, hm)[0, 1]),
    "pr_at_s1_seed202": float(pr1),
}
json.dump(res, open(OUT / "theory_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
