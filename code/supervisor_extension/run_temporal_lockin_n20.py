"""When does each factor of the task resolve, and in what order, across all twenty seeds.

This is the cross-seed replication of temporal_information.ipynb, and it exists to settle one specific
claim. On seed 1001 the homogeneous network locked in language at 232 ms and parity at 408 ms, a clean
176 ms gap, while the heterogeneous network locked both in late and close together. That was read as the
homogeneous network being a sequential processor that resolves language first and parity second, and the
heterogeneous one being concurrent. One seed cannot distinguish a mechanism from a coincidence, and a
176 ms gap on a 1000 ms trial is well within the range a single draw could produce by chance.

Two clocks are measured, because they answer different questions and can disagree.

  Instantaneous. At each of the 250 samples, the joint MI that every unit pair carries about the target,
  I({s_i(t), s_j(t)}; Y), averaged over pairs and corrected by a label-shuffle null. This is when
  information is momentarily present in the population state. Summarised by centre of mass rather than
  argmax, because a single-sample peak on a noisy profile is not a stable statistic.

  Accumulated. The running spike count C_i(<=t), quantile-binned per unit and per time point, then the same
  discrete joint MI. This is the decision-variable view and it is the one that matches what the network's
  readout actually integrates. Its summary is the lock-in time: the first crossing of ninety per cent of the
  asymptote, linearly interpolated between grid points rather than snapped to one, so the estimate is not
  quantised by the grid.

The order question is then asked three ways, so that a single arbitrary threshold does not decide it: the
signed gap parity minus language per network, the per-network ordering label under a tolerance band, and a
paired test of the gap between the arms. A gap that is real should survive all three.

Writes data_cache/supervisor_extension/temporal_lockin_n20.json.

Run: python code/supervisor_extension/run_temporal_lockin_n20.py
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
from stats_core import paired_triple, signflip_p

T_STEPS = SC.T_STEPS
DT_MS = SC.DT_MS
N_NULL_JMI = 20                                     # label shuffles for the instantaneous profile
N_NULL_ACC = 10                                     # label shuffles for the accumulation profile
L_ACC = 5                                           # quantile levels for the cumulative counts
ACC_GRID = np.unique(np.linspace(1, T_STEPS - 1, 40).astype(int))
ACC_MS = ACC_GRID * DT_MS
TOL_MS = 20.0                                       # gap below this reads as concurrent, not ordered
N_JOBS = 14


def jmi_matrix(St, Yv, K):
    """I({s_i, s_j}; Y) for every pair at one time point, from the per-class co-occurrence products.

    At a single sample the pair state is one of four, so the 4-by-K table follows from the class-conditional
    marginals and one matmul per class. This equals the syntropy discrete estimator and is fast enough to run
    for every time point, every null and every network.

    Must be float64. In float32 the co-occurrence products round enough that a state probability such as
    1 - p_i - p_j + p_ij can come out slightly negative; one class's small negative then cancels another's
    small positive in the marginal, so the marginal reaches zero or below while the class-conditional term is
    still above the guard, and the logarithm of a non-positive number returns NaN through the mask. The
    resulting NaN propagates silently into the profile and destroys its centre of mass.
    """
    n, N = St.shape
    St = np.asarray(St, dtype=np.float64)
    pI = np.zeros((K, n))
    PIJ = np.zeros((K, n, n))
    Pc = np.zeros(K)
    for c in range(K):
        m = (Yv == c)
        Nc = m.sum()
        Pc[c] = Nc / N
        Sc = St[:, m]
        pI[c] = Sc.mean(1)
        PIJ[c] = (Sc @ Sc.T) / max(Nc, 1)
    p_i, p_j = pI[:, :, None], pI[:, None, :]
    MI = np.zeros((n, n))
    Pc_ = Pc[:, None, None]
    for Ps in (1 - p_i - p_j + PIJ, p_j - PIJ, p_i - PIJ, PIJ):
        Psc = Pc_ * np.clip(Ps, 0.0, None)            # a probability cannot be negative
        Pst = Psc.sum(0)
        ok = (Psc > 1e-15) & (Pst[None] * Pc_ > 1e-15)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = Psc * (np.log2(Psc) - np.log2(Pst[None] * Pc_))
        MI += np.where(ok, term, 0.0).sum(0)
    assert np.isfinite(MI).all(), "non-finite pairwise JMI"
    np.fill_diagonal(MI, 0.0)
    return MI


def jmi_profile(R, Yv, K, rng, n_null=N_NULL_JMI):
    """Mean pairwise JMI at every time point, minus the mean of n_null label-shuffled profiles.

    The shuffle floor is not a formality. Plug-in MI on a 4-by-K table from 2048 trials is biased upward, and
    the bias depends on the firing density, which differs between the arms. Subtracting a per-time-point,
    per-network floor is what makes the two profiles comparable at all.
    """
    n = R.shape[0]
    iu = np.triu_indices(n, 1)
    S = R.astype(np.float64)
    true = np.array([jmi_matrix(S[:, :, t], Yv, K)[iu].mean() for t in range(T_STEPS)])
    null = np.zeros(T_STEPS)
    for _ in range(n_null):
        Yp = Yv[rng.permutation(len(Yv))]
        null += np.array([jmi_matrix(S[:, :, t], Yp, K)[iu].mean() for t in range(T_STEPS)])
    p = np.clip(true - null / n_null, 0, None)
    assert np.isfinite(p).all(), "non-finite instantaneous profile"
    return p


def qbin(C, L):
    """Per-unit quantile bins of the cumulative counts at one time point. MI is invariant under a monotone
    transform, so rank binning loses nothing and keeps the state space small enough to estimate."""
    out = np.zeros_like(C, dtype=np.int64)
    for i in range(C.shape[0]):
        out[i] = np.digitize(C[i], np.quantile(C[i], np.linspace(0, 1, L + 1)[1:-1]))
    return out


def onehot(Bt, L=L_ACC):
    """(L, n, N) indicator stack for L-level binned data. Built once per time point and reused across every
    target and every permutation, since the binning depends on the network and the time point alone."""
    return np.stack([(Bt == l) for l in range(L)]).astype(np.float64)


def disc_jmi_from_oh(oh, Yv, K, N):
    """Discrete plug-in JMI matrix for all pairs at once, from a prebuilt indicator stack.

    The class-conditional joint P(a,b,i,j | c) is one Gram matrix per class: flattening the level and unit
    axes into a single row index turns the four-index contraction into a single matrix product, which is
    where the speed comes from. Written as an einsum it is the same arithmetic but roughly an order of
    magnitude slower, which matters at the roughly 1300 evaluations each network needs.
    """
    L, n, _ = oh.shape
    JMI = np.zeros((n, n))
    Pab = np.zeros((L, L, n, n))
    Jc, Pc = [], np.empty(K)
    for c in range(K):
        m = (Yv == c)
        Pc[c] = m.mean()
        Ac = np.ascontiguousarray(oh[:, :, m]).reshape(L * n, -1)
        jc = (Ac @ Ac.T).reshape(L, n, L, n).transpose(0, 2, 1, 3) / N
        Jc.append(jc)
        Pab += jc
    for c in range(K):
        jc = Jc[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = jc * (np.log2(jc) - np.log2(Pab * Pc[c]))
        JMI += np.where(jc > 1e-15, term, 0).sum((0, 1))
    np.fill_diagonal(JMI, 0.0)
    return JMI


def acc_profiles(bins, targets, rng, n_null=N_NULL_ACC):
    """Accumulation profiles for every target at once, null-corrected.

    The loop runs time-outer and target-inner so the indicator stack is built once per time point rather than
    once per (time point, target, permutation). The same permutations are reused at every time point, which
    pairs the null across time and removes one source of jitter from the lock-in estimate.
    """
    n, N = bins[0].shape
    iu = np.triu_indices(n, 1)
    G = len(bins)
    names = list(targets)
    perms = {nm: [rng.permutation(N) for _ in range(n_null)] for nm in names}
    true = {nm: np.zeros(G) for nm in names}
    null = {nm: np.zeros(G) for nm in names}
    for gi, b in enumerate(bins):
        oh = onehot(b)
        for nm in names:
            Yv, K = targets[nm], SC.NK[nm]
            true[nm][gi] = disc_jmi_from_oh(oh, Yv, K, N)[iu].mean()
            acc = 0.0
            for p in perms[nm]:
                acc += disc_jmi_from_oh(oh, Yv[p], K, N)[iu].mean()
            null[nm][gi] = acc / n_null
    return {nm: np.clip(true[nm] - null[nm], 0, None) for nm in names}


def lockin_ms(p, grid_ms=ACC_MS, frac=0.9):
    """First time the accumulated information reaches `frac` of its asymptote, linearly interpolated.

    Snapping to the nearest grid point would quantise every lock-in time to the 25 ms grid spacing and could
    manufacture or erase a gap of the size under test. The asymptote is the mean of the last three points.
    """
    p = np.asarray(p, float)
    asy = p[-3:].mean()
    if not np.isfinite(asy) or asy <= 0:
        return float("nan")
    thr = frac * asy
    idx = np.where(p >= thr)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(grid_ms[0])
    p0, p1 = p[i - 1], p[i]
    if p1 == p0:
        return float(grid_ms[i])
    t = (thr - p0) / (p1 - p0)
    return float(grid_ms[i - 1] + t * (grid_ms[i] - grid_ms[i - 1]))


def analyse(seed, arch):
    t0 = time.time()
    R, _ = SC.load_spikes(seed, arch)
    T = SC.targets()
    out = {"seed": seed, "arch": arch, "n_active": int(R.shape[0])}

    inst = {}
    for name, Yv in T.items():
        p = jmi_profile(R, Yv, SC.NK[name], np.random.default_rng(seed))
        inst[name] = p
        out[f"inst_{name}_com_ms"] = SC.com_ms(p)
        out[f"inst_{name}_spread_ms"] = SC.spread_ms(p)
        out[f"inst_{name}_peak_ms"] = float(p.argmax() * DT_MS)
        out[f"inst_{name}_max_bits"] = float(p.max())
    out["inst_profiles"] = {k: v.round(7).tolist() for k, v in inst.items()}
    out["inst_gap_ms"] = out["inst_parity_com_ms"] - out["inst_language_com_ms"]

    Cc = np.cumsum(R, 2)
    bins = [qbin(Cc[:, :, g], L_ACC) for g in ACC_GRID]     # bins depend on network and time only, so reuse
    acc = acc_profiles(bins, T, np.random.default_rng(seed))
    for name, p in acc.items():
        out[f"acc_{name}_lockin_ms"] = lockin_ms(p)
        out[f"acc_{name}_asymptote"] = float(p[-3:].mean())
    out["acc_profiles"] = {k: v.round(7).tolist() for k, v in acc.items()}
    out["acc_gap_ms"] = out["acc_parity_lockin_ms"] - out["acc_language_lockin_ms"]

    g = out["acc_gap_ms"]
    out["order"] = ("concurrent" if not np.isfinite(g) or abs(g) <= TOL_MS
                    else "language-first" if g > 0 else "parity-first")

    out["seconds"] = round(time.time() - t0, 1)
    print(f"  seed {seed:>4} {arch:<3}  lock-in lang {out['acc_language_lockin_ms']:>6.0f}  "
          f"par {out['acc_parity_lockin_ms']:>6.0f}  gap {g:>+7.0f} ms  {out['order']:<14} "
          f"({out['seconds']:.0f}s)", flush=True)
    return out


def main():
    jobs = [(s, a) for s in SC.SEEDS for a in ("het", "hom")]
    print(f"temporal lock-in across {len(SC.SEEDS)} seeds, {len(jobs)} networks\n", flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(analyse)(s, a) for s, a in jobs)

    by = {(r["seed"], r["arch"]): r for r in res}
    V = lambda arch, k: np.array([by[(s, arch)][k] for s in SC.SEEDS], float)

    import style as S
    METRICS = [
        ("inst_4-class_com_ms", "Instantaneous 4-class COM (ms)"),
        ("inst_language_com_ms", "Instantaneous language COM (ms)"),
        ("inst_parity_com_ms", "Instantaneous parity COM (ms)"),
        ("inst_4-class_spread_ms", "Instantaneous 4-class spread (ms)"),
        ("inst_gap_ms", "Instantaneous COM gap, parity - language (ms)"),
        ("acc_4-class_lockin_ms", "Accumulated 4-class lock-in (ms)"),
        ("acc_language_lockin_ms", "Accumulated language lock-in (ms)"),
        ("acc_parity_lockin_ms", "Accumulated parity lock-in (ms)"),
        ("acc_gap_ms", "Lock-in gap, parity - language (ms)"),
        ("acc_4-class_asymptote", "Accumulated 4-class asymptote (bits)"),
    ]
    paired = {}
    print("\n--- paired het vs hom (n = 20) ---")
    for key, lbl in METRICS:
        r = paired_triple(V("het", key), V("hom", key), lbl)
        paired[key] = r
        print(f"{lbl:>44}: het {r['het_mean']:>8.1f}  hom {r['hom_mean']:>8.1f}  d {r['delta']:>+8.1f}  "
              f"dz {r['dz']:>+6.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor']):>14}  "
              f"wins {r['wins']}/{r['n']}", flush=True)

    # Is the gap non-zero within an arm? The paired test above only says whether the two arms differ, which
    # is a different question from whether either one actually sequences its two factors.
    print("\n--- is the language-then-parity gap real within each arm? (one-sample sign-flip on the gap) ---")
    within = {}
    for arch in ("het", "hom"):
        for key, lbl in (("acc_gap_ms", "lock-in gap"), ("inst_gap_ms", "instantaneous COM gap")):
            g = V(arch, key)
            g = g[np.isfinite(g)]                  # the count and the test must use the same networks
            p, floor = signflip_p(g)
            n_pos = int((g > 0).sum())
            within[f"{arch}_{key}"] = {"mean_ms": float(np.mean(g)), "sd_ms": float(np.std(g, ddof=1)),
                                       "median_ms": float(np.median(g)), "p": float(p), "floor": float(floor),
                                       "n_language_first": n_pos, "n": int(g.size)}
            print(f"  {arch} {lbl:<24}: mean {np.mean(g):>+7.1f} ms  median {np.median(g):>+7.1f}  "
                  f"language-first on {n_pos}/{g.size} seeds  sign-flip p {S.fmt_p(p, floor)}")

    counts = {arch: {} for arch in ("het", "hom")}
    for arch in ("het", "hom"):
        for s in SC.SEEDS:
            o = by[(s, arch)]["order"]
            counts[arch][o] = counts[arch].get(o, 0) + 1
    print(f"\n  ordering tally (tolerance {TOL_MS:.0f} ms): {counts}")

    summary = {
        "analysis": "temporal_information_and_lockin_order", "n_pairs": len(SC.SEEDS), "seeds": SC.SEEDS,
        "n_null_instantaneous": N_NULL_JMI, "n_null_accumulated": N_NULL_ACC,
        "acc_grid_ms": ACC_MS.tolist(), "tolerance_ms": TOL_MS, "quantile_levels": L_ACC,
        "paired": paired, "within_arm_gap": within, "order_counts": counts,
        "per_network": {f"{r['seed']}_{r['arch']}": {k: v for k, v in r.items()
                                                     if k not in ("inst_profiles", "acc_profiles")}
                        for r in res},
        "profiles": {f"{r['seed']}_{r['arch']}": {"inst": r["inst_profiles"], "acc": r["acc_profiles"]}
                     for r in res},
    }
    p = SC.DATA / "temporal_lockin_n20.json"
    json.dump(summary, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
