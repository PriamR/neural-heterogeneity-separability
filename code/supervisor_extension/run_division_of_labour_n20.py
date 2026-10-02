"""Do language and parity live in different modules, across all twenty seeds.

This is the module-level companion to the modularity result, and the supervisor's package calls it the
keystone: it is what ties the topological finding (homogeneous modular, heterogeneous diffuse) to the
functional one (homogeneous sequential, heterogeneous concurrent). On seed 1001 module identity predicted a
unit's factor preference in the homogeneous network, eta^2 = 0.27 with a permutation p of 0.007, and did not
in the heterogeneous one, 0.09 at p = 0.26.

That pair of numbers is exactly the shape of result that a single seed cannot establish. One arm clears an
0.05 threshold and the other does not, which is not the same as the two arms differing: with n = 1 there is
no test of the difference at all, and a per-network permutation p answers only whether that one network's
units are arranged non-randomly across its own modules. What follows keeps the per-network permutation, since
it is the right test of "is this network organised", and adds the test the claim actually needs, which is
whether the twenty paired differences in eta^2 lean one way.

The measure. Each unit gets a language MI and a parity MI, from quantile-binned per-trial counts with a
label-shuffle null, and a selectivity index SI = (L - P)/(|L| + |P|) that is scale-free by construction, so a
unit that carries more of everything does not read as a specialist. Division of labour then means module
identity explains SI: eta^2(SI ~ module).

One caution that applies to the comparison and not to either arm alone. eta^2 rises with the number of
groups for a fixed number of units, so if the two arms find different numbers of communities the comparison
is not clean. Both the community count and an eta^2 adjusted for it are recorded, and the modules are taken
from the same full-graph Louvain partition used in the modularity analysis so the two results describe the
same decomposition.

Writes data_cache/supervisor_extension/division_of_labour_n20.json.

Run: python code/supervisor_extension/run_division_of_labour_n20.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
from stats_core import paired_triple
from run_modularity_n20 import mi_matrix, circ_shift, full_graph

L_BINS = 5
N_SURR = 100
N_NULL_MI = 20
N_PERM_ETA = 2000
N_JOBS = 14


def qbin(c, L=L_BINS):
    return np.digitize(c, np.quantile(c, np.linspace(0, 1, L + 1)[1:-1]))


def single_mi(B, Yv, K, rng, n_null=N_NULL_MI):
    """Per-unit discrete MI I(binned count; Y) in bits, label-shuffle corrected.

    The shuffle floor matters because plug-in MI on an L-by-K table is biased upward and the bias depends on
    how many of the L levels a unit actually occupies, which differs between arms.
    """
    n, N = B.shape
    oh = np.stack([(B == l).astype(float) for l in range(L_BINS)])

    def mi(Y2):
        Pc = np.array([(Y2 == c).mean() for c in range(K)])
        Plc = np.stack([oh[:, :, Y2 == c].sum(2) / N for c in range(K)])
        Pl = Plc.sum(0)
        m = np.zeros(n)
        for c in range(K):
            p = Plc[c]
            with np.errstate(divide="ignore", invalid="ignore"):
                tm = p * (np.log2(p) - np.log2(Pl * Pc[c]))
            m += np.where(p > 1e-15, tm, 0).sum(0)
        return m

    return mi(Yv) - np.mean([mi(Yv[rng.permutation(N)]) for _ in range(n_null)], 0)


def analyse(seed, arch):
    t0 = time.time()
    R, _ = SC.load_spikes(seed, arch)
    n = R.shape[0]
    CNT = R.sum(2)
    T = SC.targets()
    lang, par = T["language"], T["parity"]

    # the same modules the modularity analysis uses: full-graph Louvain on the surrogate-corrected weights
    rng = np.random.default_rng(seed)
    MI_obs = mi_matrix(R.reshape(n, -1).astype(np.float32))
    null = np.zeros((n, n), np.float64)
    for _ in range(N_SURR):
        null += mi_matrix(circ_shift(R, rng))
    W = np.clip(MI_obs - null / N_SURR, 0, None)
    np.fill_diagonal(W, 0)
    G = full_graph(W)
    part = sorted(nx.community.louvain_communities(G, weight="weight", seed=seed), key=len, reverse=True)
    modid = np.zeros(n, int)
    for ci, cm in enumerate(part):
        for v in cm:
            modid[v] = ci

    B = np.stack([qbin(CNT[i]) for i in range(n)])
    r2 = np.random.default_rng(seed)
    Lm = single_mi(B, lang, 2, r2)
    Pm = single_mi(B, par, 2, r2)
    SI = (Lm - Pm) / (np.abs(Lm) + np.abs(Pm) + 1e-9)

    K = len(part)
    e2 = SC.eta2(SI, modid, K)
    rp = np.random.default_rng(seed + 7)
    nulls = np.array([SC.eta2(SI, modid[rp.permutation(n)], K) for _ in range(N_PERM_ETA)])
    perm_p = float((1 + (nulls >= e2).sum()) / (1 + N_PERM_ETA))
    # eta^2 grows with the number of groups even under the null, so the excess over the null mean is the
    # figure that can be compared between arms with different community counts
    e2_excess = float(e2 - nulls.mean())

    k = max(3, n // 4)
    topL, topP = set(np.argsort(Lm)[-k:]), set(np.argsort(Pm)[-k:])
    jac = len(topL & topP) / len(topL | topP)

    out = {"seed": seed, "arch": arch, "n_active": int(n), "n_modules": K,
           "eta2_SI_module": float(e2), "eta2_null_mean": float(nulls.mean()),
           "eta2_excess": e2_excess, "perm_p": perm_p,
           "corr_L_P": float(np.corrcoef(Lm, Pm)[0, 1]),
           "jaccard_top": float(jac),
           "SI_mean": float(SI.mean()), "SI_sd": float(SI.std()),
           "mi_lang_mean": float(Lm.mean()), "mi_par_mean": float(Pm.mean()),
           "seconds": round(time.time() - t0, 1)}
    print(f"  seed {seed:>4} {arch:<3}  {K} modules  eta2 {e2:.3f} (null {nulls.mean():.3f}, "
           f"excess {e2_excess:+.3f}, p {perm_p:.3f})  corr(L,P) {out['corr_L_P']:+.2f}  "
           f"({out['seconds']:.0f}s)", flush=True)
    return out


def main():
    jobs = [(s, a) for s in SC.SEEDS for a in ("het", "hom")]
    print(f"division of labour across {len(SC.SEEDS)} seeds, {len(jobs)} networks\n", flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(analyse)(s, a) for s, a in jobs)

    by = {(r["seed"], r["arch"]): r for r in res}
    V = lambda arch, k: np.array([by[(s, arch)][k] for s in SC.SEEDS], float)

    import style as S
    METRICS = [
        ("eta2_SI_module", "eta2(selectivity ~ module)"),
        ("eta2_excess", "eta2 excess over permutation null"),
        ("n_modules", "Number of modules"),
        ("corr_L_P", "corr(language MI, parity MI) over units"),
        ("jaccard_top", "Jaccard overlap of top language/parity units"),
        ("mi_lang_mean", "Mean per-unit language MI (bits)"),
        ("mi_par_mean", "Mean per-unit parity MI (bits)"),
    ]
    paired = {}
    print("\n--- paired het vs hom (n = 20) ---")
    for key, lbl in METRICS:
        r = paired_triple(V("het", key), V("hom", key), lbl)
        paired[key] = r
        print(f"{lbl:>44}: het {r['het_mean']:>+8.3f}  hom {r['hom_mean']:>+8.3f}  d {r['delta']:>+8.3f}  "
              f"dz {r['dz']:>+6.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor']):>14}  "
              f"wins {r['wins']}/{r['n']}", flush=True)

    sig = {a: int(sum(by[(s, a)]["perm_p"] < 0.05 for s in SC.SEEDS)) for a in ("het", "hom")}
    print(f"\n  networks with a significant within-network module alignment (p < 0.05): {sig} of 20 each")

    summary = {"analysis": "division_of_labour", "n_pairs": len(SC.SEEDS), "seeds": SC.SEEDS,
               "n_perm_eta": N_PERM_ETA, "quantile_levels": L_BINS,
               "paired": paired, "n_significant_within_network": sig,
               "per_network": {f"{r['seed']}_{r['arch']}": r for r in res}}
    p = SC.DATA / "division_of_labour_n20.json"
    json.dump(summary, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
