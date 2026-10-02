"""Shattering dimensionality against retained principal components on the TWENTY-condition set, het vs hom, n = 20.

Companion to run_shattering_vs_rank_n20.py, which runs the same analysis on the eight-condition set. The method
is identical in every respect except the condition definition and the dichotomy list:

  eight-condition set : language x parity x magnitude, all 35 balanced 4-vs-4 dichotomies (exhaustive)
  twenty-condition set: digit x language, 200 sampled balanced 10-vs-10 dichotomies

The condition definition, the trial subsample RNG (seed 0), the dichotomy sampling RNG (seed 1) and the sample
size (200) are taken from run_shattering_n20.py so that this analysis sits on exactly the same footing as the
headline twenty-condition shattering result.

Within each cross-validation fold the scaler and the PCA are fitted on the training trials only and applied to
the held-out trials, so no test information leaks into the projection. The PCA is unsupervised and therefore
label-free, which means one fit per fold is reused for every dichotomy and every rank. For each rank m the
decoder sees the first m principal component scores, which for a linear readout is equivalent to the rank-m
reconstruction in the original space.

Writes shattering_vs_rank_20cond_n20_summary.json. Nothing produced by the eight-condition run is overwritten.

Run: python code/analysis_suite/run_shattering_vs_rank_20cond_n20.py [--probe SEED]
"""
import json, sys, time, argparse
from pathlib import Path
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stats_core import paired_triple

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
OUT = REPO / "docs" / "analysis_suite"

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
RANKS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
CAP_PER_COND = 100
N_FOLDS = 5
N_JOBS = -1

N_COND = 20             # digit x language
N_SAMPLE = 200          # matches N_SAMPLE_B in run_shattering_n20.py
SEED_TRIALS = 0
SEED_DICH = 1

with h5py.File(SHD_TEST, "r") as _f:
    RAW = np.asarray(_f["labels"][:]).astype(int)
assert RAW.min() == 0 and RAW.max() == 19, f"unexpected SHD label range {RAW.min()}..{RAW.max()}"
COND = RAW.copy()       # 0..19, the twenty digit-by-language conditions


def balanced_index(cond, n_cond, cap, rng):
    k = min(cap, min(int((cond == c).sum()) for c in range(n_cond)))
    idx = []
    for c in range(n_cond):
        ci = np.where(cond == c)[0]
        rng.shuffle(ci)
        idx.append(ci[:k])
    return np.sort(np.concatenate(idx)), k


def sampled_balanced_dichotomies(n, n_sample, rng):
    """Canonicalised to the side containing condition 0, deduped; matches run_shattering_n20.py."""
    half = n // 2
    seen, out = set(), []
    guard = 0
    while len(out) < n_sample and guard < n_sample * 50:
        guard += 1
        plus = set(rng.choice(n, size=half, replace=False).tolist())
        if 0 not in plus:
            plus = set(range(n)) - plus
        key = frozenset(plus)
        if key not in seen:
            seen.add(key); out.append(key)
    return out


IDX, K_PER = balanced_index(COND, N_COND, CAP_PER_COND, np.random.default_rng(SEED_TRIALS))
COND_SUB = COND[IDX]
DICH = sampled_balanced_dichotomies(N_COND, N_SAMPLE, np.random.default_rng(SEED_DICH))
LABELS = [np.where(np.isin(COND_SUB, list(p)), 1, -1) for p in DICH]


def net_curve(seed, arch):
    """Shattering dimensionality at each rank for one network."""
    X = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")["rate"].astype(float)[IDX]
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    acc = np.zeros((len(RANKS), len(DICH)))
    for tr, te in cv.split(X, COND_SUB):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        p = PCA(n_components=max(RANKS), svd_solver="full").fit(Xtr)
        Ztr, Zte = p.transform(Xtr), p.transform(Xte)
        for di, lab in enumerate(LABELS):
            ytr, yte = lab[tr], lab[te]
            for ri, m in enumerate(RANKS):
                clf = LogisticRegression(max_iter=1000, C=1.0).fit(Ztr[:, :m], ytr)
                acc[ri, di] += clf.score(Zte[:, :m], yte)
    acc /= N_FOLDS
    return seed, arch, acc.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=None, help="time a single seed and exit")
    args = ap.parse_args()

    print(f"{N_COND} conditions, {K_PER} trials/cond ({IDX.size} trials), {len(DICH)} dichotomies, "
          f"{len(RANKS)} ranks", flush=True)

    if args.probe is not None:
        t0 = time.time()
        _, _, c = net_curve(args.probe, "het")
        dt = time.time() - t0
        print(f"one network took {dt:.1f}s -> 40 networks ~= {dt*40/60:.1f} core-min", flush=True)
        print("curve:", np.round(c, 4).tolist(), flush=True)
        return

    t0 = time.time()
    jobs = [(s, a) for s in SEEDS for a in ("het", "hom")]
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(net_curve)(s, a) for s, a in jobs)

    curves = {"het": {}, "hom": {}}
    for seed, arch, c in res:
        curves[arch][seed] = c
    M = {a: np.stack([curves[a][s] for s in SEEDS]) for a in ("het", "hom")}

    print(f"\n{'rank':>5} | {'het':>7} {'hom':>7} {'delta':>7} {'dz':>6} {'t p':>10} {'wins':>6}")
    per_rank = {}
    for ri, m in enumerate(RANKS):
        r = paired_triple(M["het"][:, ri], M["hom"][:, ri], f"shatter@m={m}")
        per_rank[m] = r
        print(f"{m:>5} | {r['het_mean']:>7.4f} {r['hom_mean']:>7.4f} {r['delta']:>+7.4f} "
              f"{r['dz']:>+6.2f} {r['t_p']:>10.3g} {r['wins']:>3}/{r['n']}", flush=True)

    sig = [m for m in RANKS if per_rank[m]["perm_p"] < 0.05 and per_rank[m]["delta"] > 0]
    onset = None
    for m in RANKS:
        if m in sig and all((mm in sig) for mm in RANKS if mm >= m):
            onset = m
            break

    d = (M["het"] - M["hom"]).mean(0)
    thr = 0.9 * d.max()
    sat = next(m for m, v in zip(RANKS, d) if v >= thr)
    print(f"\nadvantage significant and stays so from m = {onset}")
    print(f"difference at m=1 {d[0]:+.4f}; max {d.max():+.4f} at m={RANKS[int(np.argmax(d))]}; "
          f"90% of max first reached at m = {sat}")

    summary = {
        "analysis": "shattering_vs_retained_rank", "n_pairs": len(SEEDS),
        "condition_set": {"n_conditions": N_COND, "trials_per_condition": K_PER,
                          "n_dichotomies": len(DICH), "exhaustive": False,
                          "dichotomy_seed": SEED_DICH, "trial_seed": SEED_TRIALS},
        "ranks": RANKS, "onset_rank": onset, "saturation_rank_90pct": sat,
        "by_arch": {a: {"mean": M[a].mean(0).tolist(), "sd": M[a].std(0, ddof=1).tolist()}
                    for a in ("het", "hom")},
        "paired_by_rank": {str(m): per_rank[m] for m in RANKS},
        "per_network": {a: {int(s): curves[a][s].tolist() for s in SEEDS} for a in ("het", "hom")},
    }
    dest = OUT / "shattering_vs_rank_20cond_n20_summary.json"
    json.dump(summary, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
