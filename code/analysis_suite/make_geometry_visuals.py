"""Direct pictures of the class geometry, het vs hom, rather than scalar summaries of it.

Every other measure in the suite reduces the geometry to a number. This draws it. Two figures:

  F4_23  class manifolds: trials of a representative pair projected into their own top two principal
         components, coloured by task class, with 95% concentration ellipses per class. The claim the scalars
         make (tighter, less overlapping, better separated classes for the heterogeneous network) should be
         visible here without any statistic.
  F4_24  centroid geometry: the twenty digit-by-language class centroids of the same pair, which is what the
         centroid-correlation result is about.

Each network is projected in its own basis, because two different networks have no shared coordinate system.
Neurons are z-scored before the projection, the same normalisation the decoders use, so the two panels are on
comparable scales. The seed shown is the one whose accuracy gain is closest to the median, so the picture is
representative rather than the best case; it is named in the caption and set by REPRESENTATIVE below.

Run: python code/analysis_suite/make_geometry_visuals.py
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
FIGS = REPO / "docs" / "analysis_suite" / "figs"
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
CLASS_COLS = [plt.cm.viridis(x) for x in (0.05, 0.35, 0.62, 0.90)]
CLASS_NAMES = ["EN even", "EN odd", "DE even", "DE odd"]

with h5py.File(SHD_TEST, "r") as _f:
    RAW = np.asarray(_f["labels"][:]).astype(int)
DIGIT, LANG = RAW % 10, RAW // 10


def load(seed, arch):
    d = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")
    return d["rate"].astype(float), d["y"].astype(int), float((d["pred"] == d["y"]).mean())


def project(X, n=2):
    """Z-score neurons, then project onto the network's own top principal components."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Z = Z - Z.mean(0)
    U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
    var = (Sv ** 2) / (Sv ** 2).sum()
    return Z @ Vt[:n].T, var[:n]


def ellipse(ax, pts, color, nsd=2.0):
    """95% concentration ellipse (2 SD) of a 2D point cloud."""
    mu = pts.mean(0)
    C = np.cov(pts.T)
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    ang = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    e = Ellipse(mu, 2 * nsd * np.sqrt(max(w[0], 1e-12)), 2 * nsd * np.sqrt(max(w[1], 1e-12)),
                angle=ang, facecolor=color, alpha=0.18, edgecolor=color, lw=2.0, zorder=2)
    ax.add_patch(e)
    ax.plot(*mu, marker="X", color=color, ms=11, mec="white", mew=1.4, zorder=5)


def overlap_index(P, y):
    """Within-class spread relative to between-class spread. Lower means better separated classes."""
    cents = np.stack([P[y == c].mean(0) for c in range(4)])
    within = np.mean([np.linalg.norm(P[y == c] - cents[c], axis=1).mean() for c in range(4)])
    between = np.mean([np.linalg.norm(cents[i] - cents[j])
                       for i in range(4) for j in range(i + 1, 4)])
    return within / between


# choose the representative seed: accuracy gain closest to the median gain
gains = {s: load(s, "het")[2] - load(s, "hom")[2] for s in SEEDS}
med = np.median(list(gains.values()))
REPRESENTATIVE = min(gains, key=lambda s: abs(gains[s] - med))
print(f"representative seed {REPRESENTATIVE} (gain {gains[REPRESENTATIVE]:+.3f}, median {med:+.3f})")


# ================================================================= F4.23 class manifolds
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
lims = []
panels = {}
for arch in ("het", "hom"):
    X, y, acc = load(REPRESENTATIVE, arch)
    P, var = project(X, 2)
    panels[arch] = (P, y, acc, var)
    lims.append(np.abs(P).max())
lim = float(max(lims)) * 1.05

for ax, arch in zip(axes, ("het", "hom")):
    P, y, acc, var = panels[arch]
    for c in range(4):
        m = y == c
        ax.scatter(P[m, 0], P[m, 1], s=7, color=CLASS_COLS[c], alpha=0.30, linewidths=0, zorder=1)
    for c in range(4):
        ellipse(ax, P[y == c], CLASS_COLS[c])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} var)")
    ax.set_title(f"{S.LABEL[arch]}  (seed {REPRESENTATIVE}, accuracy {acc:.3f})", fontsize=10)
    ax.text(0.02, 0.02, f"within/between spread = {overlap_index(P, y):.2f}",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLS[c], label=CLASS_NAMES[c]) for c in range(4)]
axes[1].legend(handles=handles, loc="upper right", fontsize=8, title="class", title_fontsize=8)
fig.suptitle("Class manifolds in the network's own principal components (ellipses are 2 SD)",
             fontweight="bold", y=1.00)
S.save(fig, FIGS / "F4_23_class_manifolds.png")


# ================================================================= F4.24 centroid geometry
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
cent_lims = []
cpanels = {}
for arch in ("het", "hom"):
    X, y, acc = load(REPRESENTATIVE, arch)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    cents = np.stack([Z[RAW == c].mean(0) for c in range(20)])
    cents = cents - cents.mean(0)
    U, Sv, Vt = np.linalg.svd(cents, full_matrices=False)
    Pc = cents @ Vt[:2].T
    # mean absolute pairwise centroid correlation in the full space
    n = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
    C = np.abs(n @ n.T)
    mc = (C.sum() - np.trace(C)) / (20 * 19)
    cpanels[arch] = (Pc, mc, (Sv ** 2 / (Sv ** 2).sum())[:2])
    cent_lims.append(np.abs(Pc).max())
clim = float(max(cent_lims)) * 1.15

for ax, arch in zip(axes, ("het", "hom")):
    Pc, mc, var = cpanels[arch]
    for c in range(20):
        col = plt.cm.tab10(c % 10)
        mk = "o" if c < 10 else "^"
        ax.scatter(Pc[c, 0], Pc[c, 1], s=110, color=col, marker=mk, edgecolor="white",
                   linewidth=1.0, zorder=3)
        ax.annotate(str(c % 10), (Pc[c, 0], Pc[c, 1]), fontsize=7, ha="center", va="center",
                    color="white", zorder=4, fontweight="bold")
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xlim(-clim, clim); ax.set_ylim(-clim, clim)
    ax.set_xlabel(f"centroid PC1 ({var[0]:.0%} var)")
    ax.set_ylabel(f"centroid PC2 ({var[1]:.0%} var)")
    ax.set_title(f"{S.LABEL[arch]}   mean |centroid correlation| = {mc:.3f}", fontsize=10)
mk_handles = [plt.Line2D([], [], marker="o", ls="", color="#777", label="English"),
              plt.Line2D([], [], marker="^", ls="", color="#777", label="German")]
axes[1].legend(handles=mk_handles, loc="upper right", fontsize=8)
fig.suptitle(f"Class centroids, twenty digit-by-language classes (seed {REPRESENTATIVE}); "
             f"labels are the digit", fontweight="bold", y=1.00)
S.save(fig, FIGS / "F4_24_centroid_geometry.png")

for arch in ("het", "hom"):
    P, y, acc, _ = panels[arch]
    print(f"{arch}: within/between = {overlap_index(P, y):.3f}, "
          f"mean |centroid corr| = {cpanels[arch][1]:.3f}")
print("wrote F4_23_class_manifolds.png, F4_24_centroid_geometry.png")
