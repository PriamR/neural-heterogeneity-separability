"""Representational neuron ablation (fast): progressively drop hidden neurons and re-decode class from the
remaining rate code. het is ablated longest-tau-first and shortest-tau-first; hom in random order (its taus are
equal). Measures a neuron's contribution to the linearly decodable class code, NOT to the network's dynamics
(that is the dynamical-lesion version). Reads the cached clean rate codes; no forward passes.

Run: python code/analysis_suite/run_ablation_repr.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
from scipy import stats as st
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
REPR = REPO / "data_cache" / "representation_clean"
CACHE = REPO / "data_cache" / "analysis_suite"
FIGS = REPO / "docs" / "analysis_suite" / "figs" / "presentation"
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
CONFOUNDED = [202, 210, 340, 440, 550, 660, 710, 820, 930]
STEP = 2                                   # ablate in steps of 2 neurons
LEVELS = list(range(0, 64, STEP))          # neurons ablated: 0,2,...,62


def hom_name(s):
    return (f"4class_seed{s}_hom_geommean_nowd_lognormal" if s in CONFOUNDED
            else f"4class_seed{s}_hom_geommean_lognormal")


def decode(X, y, folds=5, seed=0):
    """5-fold linear decode (least squares to one-hot); numpy only. Returns accuracy on remaining columns."""
    if X.shape[1] == 0:
        return 0.25
    rng = np.random.default_rng(seed)
    n = len(y); order = rng.permutation(n); oh = np.eye(4)[y]
    accs = []
    for f in range(folds):
        te = order[f::folds]; tr = np.setdiff1d(order, te)
        A = np.c_[X[tr], np.ones(len(tr))]; B = np.c_[X[te], np.ones(len(te))]
        W = np.linalg.lstsq(A, oh[tr], rcond=None)[0]
        accs.append(float(((B @ W).argmax(1) == y[te]).mean()))
    return float(np.mean(accs))


def ablate_curve(X, y, order):
    """Accuracy vs neurons ablated, removing columns in `order` (first = ablated first)."""
    return [decode(X[:, order[k:]], y) for k in LEVELS]


het_long, het_short, hom_rand = [], [], []
for s in SEEDS:
    dh = np.load(REPR / f"repr_seed{s}_het.npz"); Xh = dh["rate"].astype(float); yh = dh["y"].astype(int)
    tau = np.array(json.load(open(CACHE / f"4class_seed{s}_het_lognormal.json"))["tau_all_ms"])
    long_first = np.argsort(-tau)          # longest tau ablated first
    short_first = np.argsort(tau)          # shortest tau ablated first
    het_long.append(ablate_curve(Xh, yh, long_first))
    het_short.append(ablate_curve(Xh, yh, short_first))
    do = np.load(REPR / f"repr_seed{s}_hom.npz"); Xo = do["rate"].astype(float); yo = do["y"].astype(int)
    # hom: average over 3 random ablation orders
    perms = [np.random.default_rng(r).permutation(64) for r in range(3)]
    hom_rand.append(np.mean([ablate_curve(Xo, yo, p) for p in perms], axis=0))
    print(f"seed {s}: done", flush=True)

het_long = np.array(het_long); het_short = np.array(het_short); hom_rand = np.array(hom_rand)


def band(ax, y, col, label, ls="-", norm=False):
    if norm:
        y = y / y[:, [0]]
    m = y.mean(0); e = st.t.ppf(0.975, y.shape[0] - 1) * y.std(0, ddof=1) / np.sqrt(y.shape[0])
    ax.plot(LEVELS, m, color=col, lw=2, ls=ls, marker="o", ms=3, label=label)
    ax.fill_between(LEVELS, m - e, m + e, color=col, alpha=0.15, lw=0)


# Absolute decodability and, beside it, decodability as a fraction of the intact code. het decodes ~9 points
# higher intact, so only the normalised panel answers "does it degrade more gracefully".
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, norm in [(axes[0], False), (axes[1], True)]:
    band(ax, het_long, S.HET, "het, longest-tau first", ls="-", norm=norm)
    band(ax, het_short, "#e08a97", "het, shortest-tau first", ls="--", norm=norm)
    band(ax, hom_rand, S.HOM, "hom, random order", ls="-", norm=norm)
    ax.set_xlabel("hidden neurons ablated")
S.chance_line(axes[0])
axes[0].set_ylabel("linear decodability of class (5-fold)")
axes[0].set_title("Absolute decodability")
axes[0].legend(loc="lower left", fontsize=9)
axes[1].axhline(1.0, color=S.CHANCE, ls=":", lw=1.2)
axes[1].set_ylabel("decodability as a fraction of the intact code")
axes[1].set_title("Normalised to each network's intact decodability")
axes[1].legend(loc="lower left", fontsize=9)
fig.suptitle("Representational ablation: het long-first vs short-first, hom random", fontweight="bold", y=1.01)
S.save(fig, FIGS / "G8a_ablation_repr.png")

def curve(A):
    return {"mean": A.mean(0).tolist(), "sd": A.std(0, ddof=1).tolist(),
            "frac_mean": (A / A[:, [0]]).mean(0).tolist(),
            "frac_sd": (A / A[:, [0]]).std(0, ddof=1).tolist(), "per_seed": A.tolist()}


# Paired per-level tests, on a coarse subset of levels so the table stays readable.
TEST_LEVELS = [8, 16, 24, 32, 40, 48, 56, 62]


def level_rows(A, B, label, norm):
    if norm:
        A, B = A / A[:, [0]], B / B[:, [0]]
    rows = []
    for lv in TEST_LEVELS:
        j = LEVELS.index(lv)
        r = paired_triple(A[:, j], B[:, j], f"{label} @ {lv}")
        rows.append({"level": lv, "comparison": f"{label}, {lv} ablated", "a_mean": r["het_mean"],
                     "b_mean": r["hom_mean"], "delta": r["delta"], "dz": r["dz"], "t_p": r["t_p"],
                     "w_p": r["w_p"], "perm_p": r["perm_p"], "perm_floor": r["perm_floor"],
                     "w_floor": r["w_floor"], "wins": r["wins"], "n": r["n"]})
    return rows


out = {"levels": LEVELS, "method": "representational (re-decode remaining rate code, 5-fold lstsq)",
       "seeds": SEEDS,
       "het_long_first": curve(het_long),
       "het_short_first": curve(het_short),
       "hom_random": curve(hom_rand),
       "tests": {
           "abs_het_long_vs_hom": level_rows(het_long, hom_rand, "het long-first vs hom", False),
           "frac_het_long_vs_hom": level_rows(het_long, hom_rand, "het long-first vs hom (fraction retained)", True),
           "frac_het_long_vs_short": level_rows(het_long, het_short, "het long-first vs short-first (fraction retained)", True)},
       "n_seeds": len(SEEDS)}
json.dump(out, open(REPO / "docs" / "analysis_suite" / "ablation_repr.json", "w"), indent=1)
print("wrote G8a_ablation_repr.png and ablation_repr.json")
