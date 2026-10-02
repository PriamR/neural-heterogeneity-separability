"""Three visuals for the corrected manifold-capacity result, each answering a different question:

  F4_18  orchestrated reduction   -- the Cohen 2020 story in one panel: capacity up, radius/dimension/correlation
                                     down. Paired effect sizes (dz) with wins, both condition sets.
  F4_19  center-correlation maps   -- het vs hom 20x20 class-centroid correlation matrices (mean |off-diagonal|),
                                     the mechanism illustrator: het decorrelates its class centres.
  F4_20  the capacity bridge       -- manifold capacity vs task accuracy across all 40 nets, the empirical form
                                     of the theorem's capacity -> accuracy link (mechanism Sec 5.3).

Pulls corrected numbers from capacity_corr_n20_summary.json and recomputes the center-correlation matrices and
per-net accuracy from the representation cache.

Run: python code/analysis_suite/make_capacity_visuals.py
"""
import json, sys
from pathlib import Path
import numpy as np
import h5py
from scipy.linalg import qr
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
CAP = json.load(open(OUT / "capacity_corr_n20_summary.json"))

with h5py.File(SHD_TEST, "r") as _f:
    RAW = np.asarray(_f["labels"][:]).astype(int)


# ================================================================= F4.18 orchestrated reduction
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
order = [("alpha_M", "capacity $\\alpha_M$"), ("R_M", "radius $R_M$"),
         ("D_M", "dimension $D_M$"), ("center_corr", "centroid corr.")]
for ax, (setk, nice) in zip(axes, [("A_8cond", "8 conditions"), ("B_20cond", "20 conditions")]):
    labs = [n for _, n in order]
    dz = [CAP["paired"][setk][k]["dz"] for k, _ in order]
    wins = [f"{CAP['paired'][setk][k]['wins']}/{CAP['paired'][setk][k]['n']}" for k, _ in order]
    cols = [S.HET if d > 0 else S.HOM for d in dz]
    y = np.arange(len(order))[::-1]
    ax.barh(y, dz, color=cols, height=0.6)
    ax.axvline(0, color="#444", lw=1)
    for yi, d, w in zip(y, dz, wins):
        ax.text(d + (0.06 if d > 0 else -0.06), yi, w, va="center",
                ha="left" if d > 0 else "right", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    ax.set_xlabel("paired effect size $d_z$ (het $-$ hom)")
    ax.set_title(nice)
    ax.set_xlim(-2.2, 2.2)
axes[0].text(0.02, 1.02, "capacity rises", transform=axes[0].transAxes, color=S.HET, fontsize=9, ha="left")
fig.suptitle("Orchestrated reduction: het raises capacity by shrinking radius, dimension and correlation",
             fontweight="bold", y=1.02)
S.save(fig, FIGS / "F4_18_orchestrated_reduction.png")


# ================================================================= F4.19 center-correlation matrices
def center_corr_matrix(centers):
    """Signed correlation matrix of class centroids in the mean-subtracted (P-1) basis."""
    N, P = centers.shape
    Xb = centers.T - centers.T.mean(0, keepdims=True)
    q, _ = qr(Xb.T, mode="economic")
    X = Xb @ q[:, : P - 1]
    xn = np.sqrt(np.square(X).sum(1, keepdims=True))
    return (X @ X.T) / (xn @ xn.T)


mats = {"het": [], "hom": []}
for arch in ("het", "hom"):
    for s in SEEDS:
        d = np.load(CACHE / f"repr_seed{s}_{arch}.npz")
        X = d["rate"].astype(float)
        centers = np.stack([X[RAW == c].mean(0) for c in range(20)], axis=1)   # 20 digit x language classes
        mats[arch].append(np.abs(center_corr_matrix(centers)))
mean_abs = {a: np.mean(mats[a], axis=0) for a in mats}
offdiag = {a: (mean_abs[a].sum() - np.trace(mean_abs[a])) / (20 * 19) for a in mats}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
vmax = max(np.max(mean_abs["het"] - np.eye(20)), np.max(mean_abs["hom"] - np.eye(20)))
for ax, arch in zip(axes, ("het", "hom")):
    M = mean_abs[arch].copy(); np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"{S.LABEL[arch]}\nmean |off-diag| = {offdiag[arch]:.3f}", fontsize=10)
    ax.set_xlabel("class (digit $\\times$ language)"); ax.set_ylabel("class")
fig.colorbar(im, ax=axes, fraction=0.045, label="| centroid correlation |")
fig.suptitle("Class-centroid correlation, het vs hom (mean over 20 seeds): het decorrelates its centres",
             fontweight="bold", y=1.02)
S.save(fig, FIGS / "F4_19_center_correlation_maps.png")


# ================================================================= F4.20 capacity -> accuracy bridge
acc = {"het": {}, "hom": {}}
for arch in ("het", "hom"):
    for s in SEEDS:
        d = np.load(CACHE / f"repr_seed{s}_{arch}.npz")
        acc[arch][s] = float((d["pred"] == d["y"]).mean())
cap20 = CAP["per_network"]["B_20cond"]
xs, ys, cs = [], [], []
for arch, col in [("het", S.HET), ("hom", S.HOM)]:
    for s in SEEDS:
        xs.append(cap20[arch][str(s)]["alpha_M"]); ys.append(acc[arch][s]); cs.append(col)
xs, ys = np.array(xs), np.array(ys)
r, p = pearsonr(xs, ys)
fig, ax = plt.subplots(figsize=(6.6, 5))
for arch, col in [("het", S.HET), ("hom", S.HOM)]:
    m = [cap20[arch][str(s)]["alpha_M"] for s in SEEDS]
    a = [acc[arch][s] for s in SEEDS]
    ax.scatter(m, a, color=col, s=42, label=S.LABEL[arch], zorder=3, edgecolor="white", linewidth=0.5)
b1, b0 = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min(), xs.max(), 50)
ax.plot(xx, b1 * xx + b0, color=S.ACCENT, lw=1.5, ls="--", zorder=2, label=f"$r$ = {r:.2f}")
ax.set_xlabel("manifold capacity $\\alpha_M$")
ax.set_ylabel("task accuracy")
ax.set_title("The capacity bridge: measured $\\alpha_M$ tracks accuracy across all 40 nets")
ax.legend(loc="lower right", fontsize=9)
S.save(fig, FIGS / "F4_20_capacity_bridge.png")

print(f"center-corr off-diagonal: het {offdiag['het']:.3f}  hom {offdiag['hom']:.3f}")
print(f"capacity-accuracy: r = {r:.3f}, p = {p:.2e}")
print("wrote F4_18_orchestrated_reduction.png, F4_19_center_correlation_maps.png, F4_20_capacity_bridge.png")
