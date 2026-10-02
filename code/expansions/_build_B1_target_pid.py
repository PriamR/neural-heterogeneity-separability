"""Builder: emits 'B1 — Target-Aware PID (unique-redundant-synergy).ipynb'.

Avenue B1: upgrade co-information (one number) to a full 2-source Williams-Beer PID of neuron PAIRS about the
class -> redundant / unique / synergistic bits. Prediction from the project's story: het's edge is in UNIQUE
information carried by MORE neurons (breadth / distributed coding), with synergy shared. Self-contained
plug-in estimator (no external PID library). Existing rate caches only.

Run once:  python _build_B1_target_pid.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# B1 — Target-aware PID: redundant / unique / synergistic information about the class

Co-information gives a single redundancy−synergy number. This notebook computes the full **2-source
Partial Information Decomposition** (Williams & Beer 2010, `I_min` redundancy) for pairs of neurons `(X₁,X₂)`
about the class `Y` (language×parity), splitting `I(X₁,X₂;Y)` into four non-negative atoms:

- **R** redundant (both neurons carry the same class info),
- **U₁, U₂** unique (info only one neuron has),
- **S** synergistic (info only in the joint pattern).

**Prediction from the project's story:** het's advantage is *breadth/distributed* coding, so het should show
**more total unique information across pairs** (each of its ~56 active neurons adds its own slice), with
redundancy/synergy comparable — the target-aware confirmation of the "pairwise/dimensional, not synergy"
picture. Self-contained plug-in estimator; inputs are the cached rate codes only.
""")

code(r'''
from pathlib import Path
import numpy as np

DEFAULT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
ROOT = DEFAULT if DEFAULT.exists() else Path.cwd()
REPR = ROOT / "Project Files" / "Representation Analysis" / "outputs"
FIGS = ROOT / "Project Files" / "Explanation files" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
HERE = Path.cwd()
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
rng = np.random.default_rng(0)
LOG2 = np.log(2)
''')

code(r'''
# ---- Williams-Beer I_min PID for binary X1,X2 and 4-valued Y (plug-in) ----
def _mi(pxy):
    px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    m = pxy > 0
    return float((pxy[m] * np.log(pxy[m] / (px @ py)[m])).sum() / LOG2)

def specific_info(joint_xy):
    """I(X ; Y=y) 'specific information' for each y, for a 2-D joint p(x,y). Returns array over y."""
    py = joint_xy.sum(0)                       # p(y)
    out = np.zeros(joint_xy.shape[1])
    for yi in range(joint_xy.shape[1]):
        if py[yi] <= 0: continue
        pxgy = joint_xy[:, yi] / py[yi]        # p(x|y)
        px = joint_xy.sum(1)                   # p(x)
        s = 0.0
        for xi in range(joint_xy.shape[0]):
            if pxgy[xi] > 0 and px[xi] > 0:
                pygx = joint_xy[xi, yi] / px[xi]
                s += pxgy[xi] * (np.log(1/py[yi]) - np.log(1/pygx)) / LOG2
        out[yi] = s
    return out

def pid_pair(x1, x2, y, ny=4):
    """binary x1,x2; y in 0..ny-1. Returns dict R,U1,U2,S (bits)."""
    n = y.size
    j1 = np.zeros((2, ny)); j2 = np.zeros((2, ny))
    for a in (0,1):
        for c in range(ny):
            j1[a,c] = np.mean((x1==a) & (y==c)); j2[a,c] = np.mean((x2==a) & (y==c))
    I1 = _mi(j1); I2 = _mi(j2)
    # joint source (x1,x2) -> 4 states
    xs = x1*2 + x2
    jj = np.zeros((4, ny))
    for a in range(4):
        for c in range(ny):
            jj[a,c] = np.mean((xs==a) & (y==c))
    I12 = _mi(jj)
    py = j1.sum(0)
    R = float(np.sum(py * np.minimum(specific_info(j1), specific_info(j2))))   # I_min redundancy
    U1 = max(I1 - R, 0.0); U2 = max(I2 - R, 0.0); S = max(I12 - I1 - I2 + R, 0.0)
    return {"R": R, "U1": U1, "U2": U2, "S": S, "I12": I12}
''')

code(r'''
import json
NPAIR = 400
agg = {a: {"R": [], "U": [], "S": [], "I12": []} for a in ("het", "hom")}
for arch in ("het", "hom"):
    for seed in SEEDS:
        f = REPR / f"repr_seed{seed}_{arch}.npz"
        if not f.exists(): print("MISSING", f); continue
        d = np.load(f); rate = d["rate"]; y = d["y"].astype(int)
        active = np.where(rate.mean(0) > 0)[0]
        B = (rate > np.median(rate, 0, keepdims=True)).astype(int)     # median-binarise per neuron
        Rs = Us = Ss = Is = 0.0; cnt = 0
        for _ in range(NPAIR):
            i, j = rng.choice(active, 2, replace=False)
            if B[:,i].min()==B[:,i].max() or B[:,j].min()==B[:,j].max(): continue
            p = pid_pair(B[:,i], B[:,j], y)
            Rs += p["R"]; Us += p["U1"]+p["U2"]; Ss += p["S"]; Is += p["I12"]; cnt += 1
        agg[arch]["R"].append(Rs/cnt); agg[arch]["U"].append(Us/cnt)
        agg[arch]["S"].append(Ss/cnt); agg[arch]["I12"].append(Is/cnt)
    print(f"{arch}: R={np.mean(agg[arch]['R']):.4f} U={np.mean(agg[arch]['U']):.4f} "
          f"S={np.mean(agg[arch]['S']):.4f} I12={np.mean(agg[arch]['I12']):.4f}", flush=True)
json.dump(agg, open(HERE / "B1_target_pid.json", "w"), indent=1)
''')

code(r'''
from scipy import stats
print("=== paired het vs hom (10 seeds), mean atom per pair (bits) ===")
for k, nm in [("R","redundant"),("U","unique (U1+U2)"),("S","synergistic"),("I12","total I(pair;Y)")]:
    h = np.array(agg["het"][k]); o = np.array(agg["hom"][k]); t = stats.ttest_rel(h, o)
    print(f"{nm:>18}: het={h.mean():.4f}  hom={o.mean():.4f}  Δ={h.mean()-o.mean():+.4f}  p={t.pvalue:.3f}")

import matplotlib.pyplot as plt
atoms = ["R","U","S"]; labels = ["redundant","unique","synergistic"]
h = [np.mean(agg["het"][k]) for k in atoms]; o = [np.mean(agg["hom"][k]) for k in atoms]
hs = [np.std(agg["het"][k]) for k in atoms]; os_ = [np.std(agg["hom"][k]) for k in atoms]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.bar(x-w/2, h, w, yerr=hs, label="het", color="#d1495b", capsize=3)
ax.bar(x+w/2, o, w, yerr=os_, label="hom", color="#30638e", capsize=3)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("bits per pair (about the class)")
ax.set_title("Target-aware PID of neuron pairs — where is het's class information?", fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(FIGS / "fig_B1_target_pid.png", dpi=150, bbox_inches="tight")
print("SAVED", FIGS / "fig_B1_target_pid.png"); plt.show()
''')

md(r"""## Reading the output

- **het higher on the *unique* atom, comparable on redundant/synergistic** → the target-aware confirmation
  of the project's headline: het's class information is **distributed** (each neuron adds its own slice),
  not more synergistic. This is the PID-level version of "pairwise/dimensional, not higher-order."
- If instead **synergy** separates the architectures, that would *revise* the story — worth knowing either
  way, and consistent-checkable against the co-information result (which found synergy shared).
- **Caveat:** `I_min` is the classic (and slightly redundancy-liberal) redundancy measure; the *direction*
  of the het−hom difference is the robust readout, not the absolute bits. Pairwise keeps the sign
  interpretable (the same dimensionality lesson as W/M).
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "B1 — Target-Aware PID (unique-redundant-synergy).ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("WROTE", out)
