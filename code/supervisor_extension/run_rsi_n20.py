"""Pairwise redundancy, synergy and RSI on the class information, across all twenty seeds.

This is the cross-seed replication of RSI_test.ipynb. On seed 1001 the supervisor found that heterogeneous
pairs are both more redundant (0.104 against 0.051 bits) and more synergistic (0.187 against 0.097), with a
higher net RSI (0.082 against 0.046), every comparison at p < 1e-4 by Mann-Whitney over pairs; but that when
each quantity is divided by the pair's joint MI, the redundancy and synergy fractions survive while the net
RSI fraction does not. The reading was that the heterogeneous network carries more class information through
graded distributed rates, and that the redundancy-synergy balance is a scale effect rather than a
restructuring.

That reading is worth testing properly, because the Mann-Whitney over pairs is the wrong unit. Pairs within a
network are not independent draws: they share the same 47 units, the same training run and the same random
seed, so a test over pairs has an effective n far below its nominal one and its p-value is not interpretable
as evidence about architectures. Here each network contributes one number, the mean over its sampled pairs,
and the twenty within-seed differences are the unit of analysis. The distribution over pairs is still
reported, as a description of spread, but no claim rests on it.

Per trial each unit contributes one integer spike count, which is the feature the network's own readout
integrates. With the class label as target, the MMI decomposition for two sources is

  redundancy  R = min(I_i, I_j)
  synergy     S = I_joint - max(I_i, I_j)
  RSI         I_joint - I_i - I_j = S - R          (interaction information, sign-meaningful at any k)

Two corrections are load-bearing rather than cosmetic. Counts are integers with heavy ties, so the kNN
distances collapse and the estimator returns NaN unless the counts are dithered. And subtracting two
separately estimated kNN differential entropies is biased, with a different bias in the two-dimensional
joint than in the one-dimensional marginals, so without a label-permutation null the joint MI comes out
negative and the synergy term is meaningless. The null is taken per term, per network.

Writes data_cache/supervisor_extension/rsi_n20.json.

Run: python code/supervisor_extension/run_rsi_n20.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
from stats_core import paired_triple

K_NN = 5           # neighbours for the continuous estimator
N_PAIRS = 300      # random unit pairs per network
N_DITHER = 2       # dither realisations averaged, to break count ties
N_PERM = 5         # label shuffles for the per-term bias floor
DITHER = 0.5
IJ_MIN = 0.05      # bits; pairs below this carry too little information to normalise by
N_JOBS = 14
LN2 = np.log(2)


def cond_H(Xc, Yd, k=K_NN):
    """H(continuous X | discrete Y) in bits, via the syntropy mixed kNN estimator.

    syntropy.mixed.mutual_information is Gaussian-only in this build, so the mixed kNN MI is assembled from
    I(X;Y) = H(X) - H(X|Y) instead; H(X) cancels against the permutation null below, leaving only conditional
    entropies to estimate.
    """
    import syntropy.mixed as SM
    _, h = SM.conditional_entropy(Yd.astype(int), Xc.astype(float),
                                  conditional="discrete", continuous_estimator="knn", k=k)
    return h / LN2


def mixed_mi_knn(Xc, Yd, rng, n_perm=N_PERM):
    """Null-corrected mixed kNN MI I(X;Y) in bits: mean_perm H(X | shuffled Y) - H(X | Y)."""
    n = Yd.shape[1]
    h_true = cond_H(Xc, Yd)
    h_null = np.mean([cond_H(Xc, Yd[:, rng.permutation(n)]) for _ in range(n_perm)])
    return float(h_null - h_true)


def _gate_check():
    """The estimator has to give the right answer on cases where the answer is known: a redundant pair
    (R > 0, S about 0), an XOR pair (S > 0, R about 0), and independent noise (all about 0)."""
    rng = np.random.default_rng(0)
    N = 1500

    def mmi(X1, X2, Y):
        r = np.random.default_rng(0)
        I1 = mixed_mi_knn(X1, Y, r)
        I2 = mixed_mi_knn(X2, Y, r)
        Ij = mixed_mi_knn(np.vstack([X1, X2]), Y, r)
        return min(I1, I2), Ij - max(I1, I2), Ij - I1 - I2

    Yr = rng.integers(0, 2, size=(1, N))
    b = Yr[0] * 3.0
    R, S, _ = mmi((b + rng.standard_normal(N))[None], (b + rng.standard_normal(N))[None], Yr)
    print(f"  redundant gate : R {R:+.3f}  S {S:+.3f}   (want R > 0, S about 0)")
    b1, b2 = rng.integers(0, 2, N), rng.integers(0, 2, N)
    R, S, _ = mmi((b1 + .15 * rng.standard_normal(N))[None], (b2 + .15 * rng.standard_normal(N))[None],
                  (b1 ^ b2)[None])
    print(f"  XOR gate       : R {R:+.3f}  S {S:+.3f}   (want R about 0, S > 0)")
    R, S, _ = mmi(rng.standard_normal(N)[None], rng.standard_normal(N)[None],
                  rng.integers(0, 2, size=(1, N)))
    print(f"  independent    : R {R:+.3f}  S {S:+.3f}   (want both about 0)")


def analyse(seed, arch):
    t0 = time.time()
    C, _ = SC.load_counts(seed, arch)
    n = C.shape[0]
    Yd = SC.targets()["4-class"][None, :].astype(int)
    rng = np.random.default_rng(seed)

    allpairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pick = rng.choice(len(allpairs), min(N_PAIRS, len(allpairs)), replace=False)
    pairs = [allpairs[p] for p in pick]

    rows = []
    for (i, j) in pairs:
        aI1 = aI2 = aIj = 0.0
        for _ in range(N_DITHER):
            X1 = C[[i]] + rng.uniform(0, DITHER, size=(1, C.shape[1]))
            X2 = C[[j]] + rng.uniform(0, DITHER, size=(1, C.shape[1]))
            aI1 += mixed_mi_knn(X1, Yd, rng)
            aI2 += mixed_mi_knn(X2, Yd, rng)
            aIj += mixed_mi_knn(np.vstack([X1, X2]), Yd, rng)
        I1, I2, Ij = aI1 / N_DITHER, aI2 / N_DITHER, aIj / N_DITHER
        rows.append((I1, I2, Ij, min(I1, I2), Ij - max(I1, I2), Ij - I1 - I2))

    A = np.array(rows, float)                     # columns: I1 I2 Ijoint R S RSI
    keep = A[:, 2] > IJ_MIN                       # normalising by a near-zero joint MI is meaningless

    out = {"seed": seed, "arch": arch, "n_active": int(n), "n_pairs": len(pairs),
           "n_kept_for_norm": int(keep.sum()),
           "I_marginal": float(np.mean(A[:, :2])), "I_joint": float(A[:, 2].mean()),
           "redundancy": float(A[:, 3].mean()), "synergy": float(A[:, 4].mean()),
           "RSI": float(A[:, 5].mean()),
           "frac_synergistic_pairs": float((A[:, 5] > 0).mean())}
    if keep.sum() >= 10:
        Ij = A[keep, 2]
        out.update(redundancy_n=float((A[keep, 3] / Ij).mean()),
                   synergy_n=float((A[keep, 4] / Ij).mean()),
                   RSI_n=float((A[keep, 5] / Ij).mean()))
    else:
        out.update(redundancy_n=np.nan, synergy_n=np.nan, RSI_n=np.nan)
    out["pairs_raw"] = A.round(5).tolist()
    out["seconds"] = round(time.time() - t0, 1)
    print(f"  seed {seed:>4} {arch:<3}  n={n:<3} Ij {out['I_joint']:.3f}  R {out['redundancy']:.3f}  "
          f"S {out['synergy']:.3f}  RSI {out['RSI']:+.3f}  RSI/Ij {out['RSI_n']:+.3f}  "
          f"({out['seconds']:.0f}s)", flush=True)
    return out


def main():
    print("estimator gate check on synthetic data:", flush=True)
    _gate_check()
    jobs = [(s, a) for s in SC.SEEDS for a in ("het", "hom")]
    print(f"\nRSI across {len(SC.SEEDS)} seeds, {len(jobs)} networks, {N_PAIRS} pairs each\n", flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(analyse)(s, a) for s, a in jobs)

    by = {(r["seed"], r["arch"]): r for r in res}
    V = lambda arch, k: np.array([by[(s, arch)][k] for s in SC.SEEDS], float)

    METRICS = [
        ("I_joint", "Pair joint MI I(X_i,X_j;Y) (bits)"),
        ("I_marginal", "Single-unit MI I(X_i;Y) (bits)"),
        ("redundancy", "Redundancy min(I_i,I_j) (bits)"),
        ("synergy", "Synergy I_joint - max(I_i,I_j) (bits)"),
        ("RSI", "RSI = synergy - redundancy (bits)"),
        ("redundancy_n", "Redundancy / I_joint"),
        ("synergy_n", "Synergy / I_joint"),
        ("RSI_n", "RSI / I_joint"),
        ("frac_synergistic_pairs", "Fraction of pairs with RSI > 0"),
    ]
    paired = {}
    import style as S
    print("\n--- paired het vs hom (n = 20, one mean per network) ---")
    for key, lbl in METRICS:
        r = paired_triple(V("het", key), V("hom", key), lbl)
        paired[key] = r
        print(f"{lbl:>36}: het {r['het_mean']:>+8.4f}  hom {r['hom_mean']:>+8.4f}  d {r['delta']:>+8.4f}  "
              f"dz {r['dz']:>+6.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor']):>14}  "
              f"wins {r['wins']}/{r['n']}", flush=True)

    summary = {
        "analysis": "pairwise_rsi_class_information", "n_pairs_design": len(SC.SEEDS), "seeds": SC.SEEDS,
        "n_unit_pairs_per_net": N_PAIRS, "k_nn": K_NN, "n_dither": N_DITHER, "n_perm": N_PERM,
        "ij_min_for_normalisation": IJ_MIN,
        "note": "unit of analysis is the network mean; the pair distribution is descriptive only",
        "paired": paired,
        "per_network": {f"{r['seed']}_{r['arch']}": {k: v for k, v in r.items() if k != "pairs_raw"}
                        for r in res},
        "pair_level": {f"{r['seed']}_{r['arch']}": r["pairs_raw"] for r in res},
    }
    p = SC.DATA / "rsi_n20.json"
    json.dump(summary, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
