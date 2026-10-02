"""VC (Cover) shattering capacity at the clean n = 20, het vs hom.

This is the in-sample counterpart to the Bernardi shattering of run_shattering_n20.py, and it asks a different
question. Bernardi shattering asks whether a decoder GENERALISES an arbitrary split of conditions to held-out
trials, so its labels must attach to conditions. VC shattering asks whether a linear readout can REALISE an
arbitrary labelling of a set of points at all, so its labels attach to individual trials and there is no
held-out set: realisability is the whole question.

A fixed small load cannot separate the two networks, because any 64-neuron code shatters a handful of points.
The informative quantity is the load at which shattering breaks down. By Cover's function-counting theorem, P
points in general position in N dimensions remain linearly separable under random labels until roughly P = 2N,
where the separable fraction crosses one half. Real population activity is not in general position: correlations
collapse it onto fewer effective dimensions, so the breakdown arrives earlier. The crossing point is therefore a
direct read of effective dimension, and the prediction is that the heterogeneous code, having more independent
directions, stays shatterable to a higher load.

We report alpha_VC = P* / N, the load at the fifty percent crossing in units of the neuron count. Cover's value
for points in general position is 2.0; anything below that measures how far the code falls short of it.

Note on the separability test: the existing helper in run_geometry_n20.py short-circuits with
"if n_points <= n_features: return True", which assumes general position. That assumption is exactly what this
analysis measures, so every case is tested here rather than assumed.

Run: python code/analysis_suite/run_vc_shattering_n20.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
N_FEAT = 64
P_GRID = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
N_REPS = 60          # random (sample, labelling) draws per load
N_JOBS = -1

# One shared set of draws for every network: the same trial indices and the same label vectors, so a difference
# between two networks is a difference between representations and not between random draws.
_rng = np.random.default_rng(0)
N_TRIALS = 2264
DRAWS = {P: [(_rng.choice(N_TRIALS, size=P, replace=False), _rng.integers(0, 2, size=P) * 2 - 1)
             for _ in range(N_REPS)] for P in P_GRID}


def separable(X, lab):
    """Is this labelling linearly realisable? Decided exactly, by linear programming.

    A margin-SVM was tried first and rejected: liblinear reports convergence failures at these loads, and a
    solver that stops early scores a separable labelling as inseparable, which would bias the whole curve
    downward and possibly by different amounts for the two architectures.

    Separability of {(x_i, y_i)} is the feasibility of  y_i (w . x_i + b) >= 1, a linear programme in
    z = [w, b]. Rescaling w and b freely means the margin of 1 costs no generality, so feasibility here is
    exactly linear separability. No general-position shortcut: where general position fails is the measurement.
    """
    n, d = X.shape
    A_ub = -lab[:, None] * np.hstack([X, np.ones((n, 1))])      # -y_i [x_i, 1]
    b_ub = -np.ones(n)
    res = linprog(c=np.zeros(d + 1), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * (d + 1), method="highs")
    return bool(res.status == 0)


def net_curve(seed, arch):
    X = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")["rate"].astype(float)
    Z = StandardScaler().fit_transform(X)          # affine, so separability is unchanged; helps the solver
    frac = []
    for P in P_GRID:
        ok = sum(separable(Z[idx], lab) for idx, lab in DRAWS[P])
        frac.append(ok / N_REPS)
    return seed, arch, np.asarray(frac, float)


def crossing(frac, grid=P_GRID, level=0.5):
    """Load at which the separable fraction falls through `level`, linearly interpolated."""
    f = np.asarray(frac, float)
    P = np.asarray(grid, float)
    for i in range(len(f) - 1):
        if f[i] >= level >= f[i + 1]:
            if f[i] == f[i + 1]:
                return float(P[i])
            t = (f[i] - level) / (f[i] - f[i + 1])
            return float(P[i] + t * (P[i + 1] - P[i]))
    return float(P[-1]) if f[-1] > level else float(P[0])       # censored


def main():
    print(f"VC shattering: loads {P_GRID}, {N_REPS} random labellings each, {N_FEAT} features\n", flush=True)
    jobs = [(s, a) for s in SEEDS for a in ("het", "hom")]
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(net_curve)(s, a) for s, a in jobs)

    curves = {"het": {}, "hom": {}}
    for seed, arch, f in res:
        curves[arch][seed] = f

    per = {a: {s: {"P_star": crossing(curves[a][s]),
                   "alpha_vc": crossing(curves[a][s]) / N_FEAT} for s in SEEDS} for a in ("het", "hom")}
    V = lambda a, k: np.array([per[a][s][k] for s in SEEDS], float)

    print(f"\n{'load P':>7} | {'het frac':>9} {'hom frac':>9}")
    Mh = np.stack([curves["het"][s] for s in SEEDS])
    Mo = np.stack([curves["hom"][s] for s in SEEDS])
    for i, P in enumerate(P_GRID):
        print(f"{P:>7} | {Mh[:, i].mean():>9.3f} {Mo[:, i].mean():>9.3f}")

    print("\n--- paired het vs hom (n = 20) ---")
    triples = {}
    for k, lbl in [("P_star", "shattering load P* (points)"), ("alpha_vc", "alpha_VC = P*/N")]:
        r = paired_triple(V("het", k), V("hom", k), lbl)
        triples[k] = r
        print(f"{lbl:>30}: het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  "
              f"dz {r['dz']:+.2f}  t {S.fmt_p(r['t_p'])}  W {S.fmt_p(r['w_p'], r['w_floor'])}  "
              f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}", flush=True)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    for arch, col, M in [("het", S.HET, Mh), ("hom", S.HOM, Mo)]:
        S.band(ax, P_GRID, M.mean(0), M.std(0), col, label=S.LABEL[arch], marker="o", ms=3.5)
    ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="50% crossing")
    ax.axvline(2 * N_FEAT, color=S.ACCENT, ls="--", lw=1.0, alpha=0.6)
    ax.annotate("Cover bound\nfor general position\n$P = 2N$", xy=(2 * N_FEAT, 0.72),
                xytext=(2 * N_FEAT * 0.62, 0.80), fontsize=8, color=S.ACCENT,
                arrowprops=dict(arrowstyle="->", color=S.ACCENT, lw=0.9))
    ax.set_xlabel("number of points $P$ given random labels")
    ax.set_ylabel("fraction linearly separable")
    ax.set_title("Shattering breaks down as load rises", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    r = triples["alpha_vc"]
    S.paired_slopes(ax, V("het", "alpha_vc"), V("hom", "alpha_vc"))
    ax.axhline(2.0, color=S.ACCENT, ls="--", lw=1.0, alpha=0.6)
    ax.text(0.5, 2.0, " Cover bound (general position)", fontsize=7.5, color=S.ACCENT, va="bottom", ha="center")
    ax.set_ylabel(r"$\alpha_{VC} = P^{*}/N$")
    ax.set_title("Shattering capacity per seed", fontsize=10)
    ax.text(0.5, 0.03, f"$\\Delta$ {r['delta']:+.3f}   $d_z$ {r['dz']:+.2f}\n"
                       f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}   wins {r['wins']}/{r['n']}",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("VC (Cover) shattering capacity, het vs hom (n = 20)", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "F4_26_vc_shattering.png")

    summary = {
        "analysis": "vc_cover_shattering_capacity", "n_pairs": len(SEEDS),
        "n_features": N_FEAT, "loads": P_GRID, "n_labellings_per_load": N_REPS,
        "cover_bound_general_position": 2.0,
        "note": "in-sample realisability of random per-trial labellings; no general-position shortcut",
        "by_arch": {a: {"frac_curve_mean": (Mh if a == "het" else Mo).mean(0).tolist(),
                        "P_star_mean": float(V(a, "P_star").mean()),
                        "alpha_vc_mean": float(V(a, "alpha_vc").mean()),
                        "alpha_vc_sd": float(V(a, "alpha_vc").std(ddof=1))} for a in ("het", "hom")},
        "paired": triples,
        "per_network": {a: {int(s): per[a][s] for s in SEEDS} for a in ("het", "hom")},
    }
    json.dump(summary, open(OUT / "vc_shattering_n20_summary.json", "w"), indent=1)
    print(f"\nwrote {OUT/'vc_shattering_n20_summary.json'} and F4_26_vc_shattering.png")


if __name__ == "__main__":
    main()
