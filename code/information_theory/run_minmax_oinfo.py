"""WS2 + WS3 (Hardik's O-Info feedback):
  WS2  min / max O-Information per (network, k) instead of the average — the most synergistic and the most
       redundant subset, and how far they move between het and hom.
  WS3  scatter: per het subset, x = mean pairwise |tau_i - tau_j| among its neurons, y = the subset's O-info.
       Tests whether more timescale-diverse subsets are less redundant / more synergistic.

Runs on the existing os_info_records.jsonl (per-subset O/S + neuron lists) + checkpoint tau_mem. No retrain.
Run: python run_minmax_oinfo.py
"""
import sys, json, itertools
from pathlib import Path
import numpy as np, torch
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REC = ROOT / "Project Files" / "O-S Information" / "outputs" / "os_info_records.jsonl"
SPK = ROOT / "Project Files" / "Discrete Zscore Pipeline" / "outputs" / "spk_cache"
CK = ROOT / "Project Files" / "Checkpoints" / "SeededRuns" / "seeded_run_1_64n" / "4class_lognormal"
FIGS = ROOT / "Project Files" / "Explanation files" / "figures"
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
COL = {"het": "#d1495b", "hom": "#30638e"}
RK = [2, 4, 6, 8, 10, 12]
tau_ms = lambda b: -1.0 / np.log(np.clip(b, 1e-9, 1 - 1e-9))

ok = [json.loads(l) for l in open(REC)]; ok = [r for r in ok if r.get("status") == "ok"]

# ----------------------------------------------------------------- WS2: min / max O
def vals(arch, seed, k):
    return np.array([r["O_info"] for r in ok if r["arch"] == arch and r["seed"] == seed and r["k"] == k])

print("=== WS2: min / max / spread of O-information per (network, k) ===")
print(f"{'k':>3} | {'het minO':>9} {'het maxO':>9} {'het spread':>10} | {'hom minO':>9} {'hom maxO':>9} {'hom spread':>10}")
agg = {a: {st: {} for st in ["min", "max", "spread"]} for a in ["het", "hom"]}
for k in RK:
    for a in ["het", "hom"]:
        mins, maxs = [], []
        for s in SEEDS:
            v = vals(a, s, k)
            if v.size: mins.append(v.min()); maxs.append(v.max())
        agg[a]["min"][k] = np.array(mins); agg[a]["max"][k] = np.array(maxs)
        agg[a]["spread"][k] = np.array(maxs) - np.array(mins)
    print(f"{k:>3} | {agg['het']['min'][k].mean():>+9.3f} {agg['het']['max'][k].mean():>+9.3f} "
          f"{agg['het']['spread'][k].mean():>10.3f} | {agg['hom']['min'][k].mean():>+9.3f} "
          f"{agg['hom']['max'][k].mean():>+9.3f} {agg['hom']['spread'][k].mean():>10.3f}")

print("\n--- significance at k=8 (paired het vs hom, 10 seeds) ---")
for st in ["min", "max", "spread"]:
    h, o = agg["het"][st][8], agg["hom"][st][8]
    dz = (h - o).mean() / (h - o).std(ddof=1)
    print(f"O {st:>6}: het={h.mean():+.3f} hom={o.mean():+.3f} diff={h.mean()-o.mean():+.3f} "
          f"t p={stats.ttest_rel(h, o).pvalue:.3f} Wilcox p={stats.wilcoxon(h, o).pvalue:.3f} dz={dz:+.2f}")

# fig: min & max O vs k
fig, ax = plt.subplots(figsize=(9, 5.5))
for a in ["het", "hom"]:
    mn = [agg[a]["min"][k].mean() for k in RK]; mx = [agg[a]["max"][k].mean() for k in RK]
    ax.plot(RK, mx, color=COL[a], lw=2, marker="^", label=f"{a} max O (most redundant)")
    ax.plot(RK, mn, color=COL[a], lw=2, ls="--", marker="v", label=f"{a} min O (most synergistic)")
    ax.fill_between(RK, mn, mx, color=COL[a], alpha=0.08)
ax.axhline(0, color="black", lw=1); ax.set_xlabel("subset size k"); ax.set_ylabel("O-information")
ax.set_title("Min / Max O-information of subsets (het vs hom)\n(range = how far the most synergistic vs most redundant subsets sit)",
             fontweight="bold")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / "fig_minmax_oinfo.png", dpi=150, bbox_inches="tight"); plt.close()
print("SAVED fig_minmax_oinfo.png")

# ----------------------------------------------------------------- WS3: timescale-diversity vs info (het)
def het_tau_active(seed):
    spk = np.load(SPK / f"4class_seed{seed}_het_spk.npy"); active = np.where(spk.sum(1) > 0)[0]
    beta = torch.load(CK / f"4class_seed{seed}_het_lognormal.pt", map_location="cpu",
                      weights_only=False)["model_state_dict"]["network.0.beta"].numpy().ravel()
    return tau_ms(beta)[active]                          # tau (ms) of the active neurons, in active order

pts = {6: ([], []), 8: ([], [])}
for seed in SEEDS:
    tau_a = het_tau_active(seed)
    for r in ok:
        if r["arch"] != "het" or r["seed"] != seed or r["k"] not in pts: continue
        ns = r.get("neurons");
        if not ns: continue
        taus = tau_a[np.array(ns)]
        dtau = np.mean([abs(taus[i] - taus[j]) for i, j in itertools.combinations(range(len(taus)), 2)])
        pts[r["k"]][0].append(dtau); pts[r["k"]][1].append(r["O_info"])

print("\n=== WS3: het — mean pairwise |Δτ| vs subset O-info ===")
fig, ax = plt.subplots(figsize=(8.5, 6))
for k, mk in [(6, "o"), (8, "s")]:
    x = np.array(pts[k][0]); y = np.array(pts[k][1])
    pr = stats.pearsonr(x, y); sp = stats.spearmanr(x, y)
    ax.scatter(x, y, s=10, alpha=0.35, marker=mk, label=f"k={k}  (Pearson r={pr.statistic:+.2f}, p={pr.pvalue:.1e}; Spearman ρ={sp.statistic:+.2f})")
    b, a0 = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 50); ax.plot(xx, a0 + b * xx, lw=2)
    print(f"k={k}: n={x.size}  Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.2e})  Spearman rho={sp.statistic:+.3f} (p={sp.pvalue:.2e})")
ax.axhline(0, color="black", lw=1, alpha=0.5)
ax.set_xlabel("mean pairwise |Δτ_mem| among the subset's neurons (ms)")
ax.set_ylabel("subset O-information  (lower = more synergistic)")
ax.set_title("het: do more timescale-diverse subsets carry less-redundant information?", fontweight="bold")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
plt.tight_layout(); plt.savefig(FIGS / "fig_delay_vs_info.png", dpi=150, bbox_inches="tight"); plt.close()
print("SAVED fig_delay_vs_info.png")
