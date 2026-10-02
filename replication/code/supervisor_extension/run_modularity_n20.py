"""Functional-network modularity, heterogeneous against homogeneous, across all twenty seeds.

This is the cross-seed replication of network_analysis.ipynb sections 1-11 from the supervisor's package,
whose headline claim rests on one seed: the heterogeneous network's co-activation graph is near-random
(Q/Q_rand about 0.62 on the full graph) while the homogeneous one is modular and small-world (about 1.39).
That claim is the topological form of the whole distributed-versus-modular story, so it is the one that most
needs an n.

The pipeline is the supervisor's, unchanged in substance:

  edge statistic   zero-lag binary mutual information between units over the pooled 2048 x 250 samples
  null             circular time shift, an independent offset per unit and trial, which preserves each
                   unit's rate and its within-trial autocorrelation and destroys only cross-unit timing
  weight           the excess over the null mean, clipped at zero
  modularity       weighted Louvain, normalised against degree-preserving double-edge-swap randomisations
  small-worldness  sigma = (C/C_rand)/(L/L_rand) on the largest component

Two graphs are built per network because they fail differently. The binned graph reads within-trial
co-activity and could in principle be defeated by the heterogeneous network's diverse time constants
smearing coincidences. The cumulative-count graph reads trial-integrated rate covariation through a
continuous kNN estimator and has no time bins at all, so it cannot suffer that failure. Agreement between
them is what rules out a time-binning artefact.

Density matching matters more here than anywhere else in the suite. The heterogeneous network fires more, so
its raw graph is denser and heavier; comparing raw Q would measure firing rate. Every reported comparison is
either at matched density or normalised against a degree-preserving null, and usually both.

Writes data_cache/supervisor_extension/modularity_n20.json.

Run: python code/supervisor_extension/run_modularity_n20.py
"""
import itertools
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

N_SURR = 100                                     # circular-shift surrogates for the edge null
N_NULL = 10                                      # degree-preserving randomisations for Q_rand
DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
K_KNN, N_DITHER, N_KNULL = 5, 2, 2                # cumulative-count KSG settings
N_JOBS = 14


# ---------------------------------------------------------------- edge statistics
def mi_matrix(S):
    """Zero-lag pairwise binary MI in bits for every unit pair at once.

    Each of the pooled samples is a joint state (s_i, s_j) in {0,1}^2, so the whole 2x2 table follows from
    the marginals and the single co-occurrence product S S^T / N. That is the entire matrix in one matmul.
    """
    n, N = S.shape
    p1 = S.mean(1)
    P11 = (S @ S.T) / N
    pi, pj = p1[:, None], p1[None, :]

    def term(pxy, px, py):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = pxy * (np.log2(pxy) - np.log2(px * py))
        return np.where(pxy > 1e-15, t, 0.0)

    MI = (term(P11, pi, pj) + term(pi - P11, pi, 1 - pj)
          + term(pj - P11, 1 - pi, pj) + term(1 - pi - pj + P11, 1 - pi, 1 - pj))
    MI = np.clip(MI, 0, None)
    np.fill_diagonal(MI, 0.0)
    return MI


def circ_shift(Rt, rng):
    """Independent circular shift per unit and trial. Rate and within-trial autocorrelation survive; only the
    alignment between units is destroyed, which is exactly the quantity the edge is supposed to measure."""
    n, TR, T = Rt.shape
    off = rng.integers(0, T, size=(n, TR)).astype(np.int32)
    idx = (np.arange(T, dtype=np.int32)[None, None, :] - off[:, :, None]) % T
    return np.take_along_axis(Rt, idx, axis=2).reshape(n, TR * T).astype(np.float32)


def knn_mi_matrix(CNT, rng):
    """Pairwise KSG kNN MI in bits across trials on per-trial counts, trial-shuffle corrected.

    Integer counts give zero k-th-neighbour distances and the estimator returns inf, so the counts are
    dithered; the shuffle null removes the estimator's bias floor, which is not the same in the two arms
    because their count distributions differ.
    """
    import syntropy.knn as KNN
    n, N = CNT.shape
    pairs = list(itertools.combinations(range(n), 2))

    def pw(D):
        M = np.zeros((n, n))
        for (i, j) in pairs:
            _, mi = KNN.mutual_information((i,), (j,), K_KNN, D)
            M[i, j] = M[j, i] = mi / np.log(2)
        return M

    true = np.mean([pw(CNT + rng.uniform(0, 0.5, CNT.shape)) for _ in range(N_DITHER)], 0)

    def null_draw():
        Dn = CNT.astype(float).copy()
        for r in range(n):
            Dn[r] = Dn[r][rng.permutation(N)]
        return pw(Dn + rng.uniform(0, 0.5, Dn.shape))

    null = np.mean([null_draw() for _ in range(N_KNULL)], 0)
    W = np.clip(true - null, 0, None)
    np.fill_diagonal(W, 0)
    return true, null, W


# ---------------------------------------------------------------- graph metrics
def full_graph(W):
    """Every edge with positive excess weight, no density threshold. This keeps all units in one component,
    so the community count is real rather than inflated by singletons."""
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in np.where(W[iu] > 0)[0]:
        G.add_edge(int(iu[0][e]), int(iu[1][e]), weight=float(W[iu][e]))
    return G


def graph_at_density(W, d, weighted):
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    w = W[iu]
    K = max(1, int(round(d * len(w))))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in np.argsort(w)[::-1][:K]:
        if w[e] <= 0:
            continue
        G.add_edge(int(iu[0][e]), int(iu[1][e]), weight=float(w[e]) if weighted else 1.0)
    return G


def randomize(G, rng, weighted):
    """Degree-preserving double-edge-swap, weights reshuffled onto the result. Preserving degree is what
    makes the normalisation isolate topology from the rate differences that drive degree."""
    ws = [d["weight"] for *_, d in G.edges(data=True)] if weighted else None
    H = G.copy()
    m = H.number_of_edges()
    if m > 1 and H.number_of_nodes() > 3:
        try:
            H = nx.double_edge_swap(H, nswap=10 * m, max_tries=100 * m,
                                    seed=int(rng.integers(1_000_000_000)))
        except Exception:
            pass
    if weighted:
        rng.shuffle(ws)
        for (u, v), val in zip(list(H.edges()), ws):
            H[u][v]["weight"] = val
    return H


def clus_path(G, weighted):
    if G.number_of_edges() == 0:
        return 0.0, np.nan, 0.0
    C = nx.average_clustering(G, weight="weight" if weighted else None)
    lcc = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc)
    frac = len(lcc) / G.number_of_nodes()
    if H.number_of_nodes() < 2:
        return C, np.nan, frac
    if weighted:
        Hd = nx.Graph()
        Hd.add_nodes_from(H)
        for u, v, d in H.edges(data=True):
            Hd.add_edge(u, v, dist=1.0 / max(d["weight"], 1e-9))
        L = nx.average_shortest_path_length(Hd, weight="dist")
    else:
        L = nx.average_shortest_path_length(H)
    return C, L, frac


def mod_sw(G, seed, weighted=True, n_null=N_NULL):
    """Modularity and small-worldness with their degree-preserving normalisations."""
    wkey = "weight" if weighted else None
    part = sorted(nx.community.louvain_communities(G, weight=wkey, seed=seed), key=len, reverse=True)
    Q = nx.community.modularity(G, part, weight=wkey)
    C, L, _ = clus_path(G, weighted)
    rng = np.random.default_rng(seed)
    Qn, Cn, Ln = [], [], []
    for _ in range(n_null):
        Gr = randomize(G, rng, weighted)
        pr = nx.community.louvain_communities(Gr, weight=wkey, seed=seed)
        Qn.append(nx.community.modularity(Gr, pr, weight=wkey))
        cc, ll, _ = clus_path(Gr, weighted)
        Cn.append(cc)
        Ln.append(ll)
    Qr, Cr, Lr = np.mean(Qn), np.nanmean(Cn), np.nanmean(Ln)
    sigma = ((C / Cr) / (L / Lr)) if (Cr > 0 and np.isfinite(L) and np.isfinite(Lr) and L > 0) else np.nan
    n = G.number_of_nodes()
    return dict(Q=float(Q), Qrand=float(Qr), Qnorm=float(Q / Qr) if Qr else np.nan,
                ncomm=len(part), sigma=float(sigma), C=float(C), L=float(L),
                density=float(2 * G.number_of_edges() / (n * (n - 1))) if n > 1 else np.nan), part


# ---------------------------------------------------------------- per-network job
def analyse(seed, arch):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, active = SC.load_spikes(seed, arch)
    n = R.shape[0]
    S = R.reshape(n, -1).astype(np.float32)

    MI_obs = mi_matrix(S)
    stack = np.empty((N_SURR, n, n), np.float32)
    for s in range(N_SURR):
        stack[s] = mi_matrix(circ_shift(R, rng))
    null_mean = stack.mean(0)
    W = np.clip(MI_obs - null_mean, 0, None)
    np.fill_diagonal(W, 0)

    out = {"seed": seed, "arch": arch, "n_active": int(n)}
    iu = np.triu_indices(n, 1)
    out["edge_mi_mean"] = float(MI_obs[iu].mean())
    out["null_floor_frac"] = float(null_mean[iu].mean() / MI_obs[iu].mean())

    # the headline: the full positive-excess graph, no density threshold
    G = full_graph(W)
    m, part = mod_sw(G, seed, weighted=True)
    out["full"] = m
    out["comm_sizes"] = sorted([len(c) for c in part], reverse=True)

    # matched-density sweep, weighted and binary, as the robustness check
    sweep = []
    for weighted in (True, False):
        for d in DENSITIES:
            Gd = graph_at_density(W, d, weighted)
            md, _ = mod_sw(Gd, seed, weighted=weighted)
            md.update(weighting="weighted" if weighted else "binary", target_density=d)
            sweep.append(md)
    out["sweep"] = sweep

    # cumulative-count graph: no time bins at all, so it cannot be a binning artefact
    CNT = R.sum(2).astype(float)
    _, _, cW = knn_mi_matrix(CNT, np.random.default_rng(seed))
    Gc = full_graph(cW)
    mc, partc = mod_sw(Gc, seed, weighted=True)
    out["cumulative"] = mc

    # are the label-blind communities class tuned, and on which factor?
    T = SC.targets()
    Y, lang, par = T["4-class"], T["language"], T["parity"]
    cnt_i = R.sum(2)
    for key, pp in (("tuning_binned", part), ("tuning_cumulative", partc)):
        e4, el, ep = [], [], []
        for com in pp:
            a = cnt_i[sorted(com)].mean(0).astype(float)
            e4.append(SC.eta2(a, Y, 4))
            el.append(SC.eta2(a, lang, 2))
            ep.append(SC.eta2(a, par, 2))
        out[key] = {"eta2_4c": float(np.mean(e4)), "eta2_lang": float(np.mean(el)),
                    "eta2_par": float(np.mean(ep)), "eta2_4c_max": float(np.max(e4)),
                    "n_comm": len(pp)}

    out["seconds"] = round(time.time() - t0, 1)
    print(f"  seed {seed:>4} {arch:<3}  n={n:<3} full Q/Qrand {m['Qnorm']:.3f}  sigma {m['sigma']:.2f}  "
          f"cum Q/Qrand {mc['Qnorm']:.3f}  ({out['seconds']:.0f}s)", flush=True)
    return out


def main():
    jobs = [(s, a) for s in SC.SEEDS for a in ("het", "hom")]
    print(f"modularity across {len(SC.SEEDS)} seeds, {len(jobs)} networks, {N_SURR} surrogates each\n", flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(analyse)(s, a) for s, a in jobs)

    by = {(r["seed"], r["arch"]): r for r in res}
    V = lambda arch, fn: np.array([fn(by[(s, arch)]) for s in SC.SEEDS], float)

    METRICS = [
        ("full_Qnorm", "Full-graph Q/Q_rand (binned)", lambda r: r["full"]["Qnorm"]),
        ("full_Q", "Full-graph modularity Q (binned)", lambda r: r["full"]["Q"]),
        ("full_sigma", "Small-worldness sigma (binned)", lambda r: r["full"]["sigma"]),
        ("full_ncomm", "Communities (binned full graph)", lambda r: r["full"]["ncomm"]),
        ("cum_Qnorm", "Cumulative-count Q/Q_rand", lambda r: r["cumulative"]["Qnorm"]),
        ("cum_sigma", "Cumulative-count sigma", lambda r: r["cumulative"]["sigma"]),
        ("eta2_lang", "Community eta2 language (binned)", lambda r: r["tuning_binned"]["eta2_lang"]),
        ("eta2_par", "Community eta2 parity (binned)", lambda r: r["tuning_binned"]["eta2_par"]),
        ("eta2_4c_cum", "Community eta2 4-class (cumulative)", lambda r: r["tuning_cumulative"]["eta2_4c"]),
        ("n_active", "Active units", lambda r: r["n_active"]),
        ("edge_mi_mean", "Mean edge MI (bits)", lambda r: r["edge_mi_mean"]),
    ]
    paired = {}
    print("\n--- paired het vs hom (n = 20) ---")
    import style as S
    for key, lbl, fn in METRICS:
        r = paired_triple(V("het", fn), V("hom", fn), lbl)
        paired[key] = r
        print(f"{lbl:>36}: het {r['het_mean']:>8.3f}  hom {r['hom_mean']:>8.3f}  d {r['delta']:>+8.3f}  "
              f"dz {r['dz']:>+6.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor']):>14}  "
              f"wins {r['wins']}/{r['n']}", flush=True)

    # the density sweep, averaged over seeds, for the robustness figure
    sweep = {}
    for weighting in ("weighted", "binary"):
        for arch in ("het", "hom"):
            M = np.array([[next(x["Qnorm"] for x in by[(s, arch)]["sweep"]
                                if x["weighting"] == weighting and x["target_density"] == d)
                           for d in DENSITIES] for s in SC.SEEDS], float)
            Sg = np.array([[next(x["sigma"] for x in by[(s, arch)]["sweep"]
                                 if x["weighting"] == weighting and x["target_density"] == d)
                            for d in DENSITIES] for s in SC.SEEDS], float)
            sweep[f"{weighting}_{arch}"] = {"Qnorm": M.tolist(), "sigma": Sg.tolist()}

    summary = {
        "analysis": "functional_network_modularity", "n_pairs": len(SC.SEEDS), "seeds": SC.SEEDS,
        "n_surrogates": N_SURR, "n_degree_nulls": N_NULL, "densities": DENSITIES,
        "paired": paired, "densities_sweep": sweep,
        "per_network": {f"{r['seed']}_{r['arch']}": r for r in res},
    }
    p = SC.DATA / "modularity_n20.json"
    json.dump(summary, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
