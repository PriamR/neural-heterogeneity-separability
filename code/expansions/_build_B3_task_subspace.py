"""Builder: emits 'B3 — Task-Subspace Alignment.ipynb'.

Avenue B3: we know het has more *functional* PCs (PC-reconstruction crossover). Here we ask WHICH task axis
lives in WHICH dimensions: for each task axis (language, parity, XOR) compute its coding vector and measure
how many principal components of the rate code are needed to capture it (cumulative squared cosine), het vs
hom. Turns "higher-dimensional" into "higher-dimensional in the task-relevant directions" (Wakhloo 2026:
task-latent PCs map onto neural-activity PCs). Existing rate caches only.

Run once:  python _build_B3_task_subspace.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# B3 — Task-subspace alignment: which dimensions carry which task variable

het's extra PCs are *functional* (PC-reconstruction crossover). This notebook asks **where each task axis
lives** in the code. For each network we build the three balanced-dichotomy **coding vectors** in the 64-D
rate space:
- **language** = mean(Ger) − mean(Eng), **parity** = mean(odd) − mean(even), **XOR** = mean{0,3} − mean{1,2}.

We PCA the rate code and measure the **cumulative squared cosine** of each coding vector with the top-*m* PC
subspace — i.e. *how many PCs you need to reconstruct that task direction*. Following Wakhloo et al. 2026
(the PCs of the task latents map onto the PCs of neural activity), a code that packs an axis into few PCs is
compressed along it; a code that spreads it needs more.

**Prediction:** het spreads the *hard* axis (parity, and the XOR conjunction) across **more** PCs — its
functional dimensionality is spent on the task-relevant directions — while keeping the classes separable
(cross-ref the manifold/CCGP results). Inputs: cached rate codes only.
""")

code(r'''
from pathlib import Path
import numpy as np

def pca_fit(z, k):
    """Return unit principal components V [k,64] and explained variances lam [k] via numpy SVD."""
    zc = z - z.mean(0)
    U, S, Vt = np.linalg.svd(zc, full_matrices=False)
    lam = (S ** 2) / (z.shape[0] - 1)
    return Vt[:k], lam[:k]

DEFAULT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
ROOT = DEFAULT if DEFAULT.exists() else Path.cwd()
REPR = ROOT / "Project Files" / "Representation Analysis" / "outputs"
FIGS = ROOT / "Project Files" / "Explanation files" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
HERE = Path.cwd()
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]

def coding_vectors(rate, y):
    z = (rate - rate.mean(0)) / (rate.std(0) + 1e-9)     # standardise
    lang = (y >= 2); parity = (y % 2 == 1); xor = np.isin(y, [0, 3])
    v = {
        "language": z[lang].mean(0) - z[~lang].mean(0),
        "parity":   z[parity].mean(0) - z[~parity].mean(0),
        "XOR":      z[xor].mean(0) - z[~xor].mean(0),
    }
    return z, {k: (vv / (np.linalg.norm(vv) + 1e-12)) for k, vv in v.items()}
''')

code(r'''
import json
MS = list(range(1, 33))
curves = {a: {ax: [] for ax in ("language","parity","XOR")} for a in ("het","hom")}
pr_dim = {a: [] for a in ("het","hom")}
for arch in ("het","hom"):
    for seed in SEEDS:
        f = REPR / f"repr_seed{seed}_{arch}.npz"
        if not f.exists(): print("MISSING", f); continue
        d = np.load(f); rate = d["rate"]; y = d["y"].astype(int)
        z, cv = coding_vectors(rate, y)
        V, lam = pca_fit(z, min(64, z.shape[1]))               # [k,64] unit PCs, explained variances
        pr_dim[arch].append(float((lam.sum()**2) / (lam**2).sum()))    # participation ratio
        for ax in ("language","parity","XOR"):
            proj = V @ cv[ax]                                  # cos with each PC
            cum = np.cumsum(proj**2)                           # cumulative squared cosine (-> 1)
            curves[arch][ax].append([cum[min(m, cum.size)-1] for m in MS])
    print(f"{arch}: PR={np.mean(pr_dim[arch]):.2f}", flush=True)
# PCs needed to capture 80% of each axis
print("\nPCs to reach 80% of coding-vector norm (mean over seeds):")
for ax in ("language","parity","XOR"):
    for arch in ("het","hom"):
        C = np.array(curves[arch][ax]).mean(0)
        n80 = next((MS[i] for i in range(len(MS)) if C[i] >= 0.8), MS[-1])
        print(f"  {ax:>9} {arch}: n80={n80}", end="   ")
    print()
json.dump({"MS": MS, "curves": {a: {ax: np.array(curves[a][ax]).mean(0).tolist() for ax in curves[a]} for a in curves},
           "PR": pr_dim}, open(HERE / "B3_task_subspace.json", "w"), indent=1)
''')

code(r'''
import matplotlib.pyplot as plt
COL = {"het": "#d1495b", "hom": "#30638e"}
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for j, ax_name in enumerate(("language","parity","XOR")):
    for arch in ("het","hom"):
        C = np.array(curves[arch][ax_name]); m = C.mean(0); sd = C.std(0)
        axes[j].plot(MS, m, color=COL[arch], lw=2.2, marker="o", ms=3, label=arch)
        axes[j].fill_between(MS, m-sd, m+sd, color=COL[arch], alpha=0.15)
    axes[j].axhline(0.8, color="k", ls=":", lw=1)
    axes[j].set_xlabel("# principal components (m)"); axes[j].set_title(f"{ax_name} axis", fontweight="bold")
    axes[j].grid(alpha=0.3)
axes[0].set_ylabel("cumulative alignment  (Σ cos² with top-m PCs)"); axes[0].legend()
fig.suptitle("Where each task axis lives in the code (het spreads the hard axes over more PCs)", fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS / "fig_B3_task_subspace.png", dpi=150, bbox_inches="tight")
print("SAVED", FIGS / "fig_B3_task_subspace.png"); plt.show()
''')

md(r"""## Reading the output

- A curve that **rises later** (needs more PCs to reach 0.8) means that task axis is **spread across more
  dimensions**. If het spreads **parity/XOR** over more PCs than hom while still decoding them better
  (CCGP/manifold), that pins het's functional dimensionality to the *task-relevant* directions — the direct
  "where" for the dimensionality advantage.
- Cross-check against `PR` (participation ratio) printed per arch: het's slightly higher PR should be
  concentrated in the axes that matter, not spread as noise.
- Connects to Wakhloo 2026 (task-latent ↔ neural PCs) and to the PC-reconstruction crossover (het keeps
  gaining accuracy as PCs 16→32 are added — those are the PCs this notebook attributes to task axes).
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "B3 — Task-Subspace Alignment.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("WROTE", out)
