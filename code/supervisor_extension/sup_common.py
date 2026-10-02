"""Shared loading and helpers for the cross-seed extension of the supervisor's information analyses.

The supervisor's package (O-Info Review Package) runs every analysis on seed 1001 alone and says so: n = 1
per arm, illustrative until replicated. This module is the base the replication scripts sit on. It resolves
which checkpoint is the homogeneous partner for a given seed, loads the 4 ms spike cache, and supplies the
labels and the active-unit filter, so every script in this directory sees exactly the same data.

Three facts about the cache that the analyses depend on:

  Shape. Each file is [64, 512000] uint8, reshaping to (64 neurons, 2048 trials, 250 steps). The 2048 trials
  are the first 2048 of shd_test.h5 in file order, which is what makes the label join valid.

  Sub-sampling, not binning. run_oinfo.build_cache takes spk[:, ::4, :], every fourth 1 ms step. It does not
  OR four steps into a 4 ms bin. So a cached "count" is roughly a quarter of the true spike count and the
  cache is a 250-sample subsample of the trial, not a coarsening of it. Nothing here is invalidated by that,
  because both arms are treated identically and every measure is either null-corrected or paired, but it does
  mean absolute rates in this pipeline are not the network's firing rates.

  The homogeneous partner differs by seed. Seeds 101 and 1001-1010 use the plain geometric-mean homogeneous
  network; seeds 202-930 use the no-weight-decay retrain, which is the one the clean n=20 design pairs. That
  is why arch "hom" maps to a different filename for those nine seeds.
"""
import sys
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RROOT = REPO.parent
CACHE = REPO / "data_cache" / "analysis_suite" / "spk_cache_4ms"
SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"

OUT = REPO / "docs" / "supervisor_extension"
FIGS = OUT / "figs"
DATA = REPO / "data_cache" / "supervisor_extension"
for _p in (OUT, FIGS, DATA):
    _p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "code" / "analysis_suite"))

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
NOWD_SEEDS = {202, 210, 340, 440, 550, 660, 710, 820, 930}

N_NEURONS, N_TRIALS, T_STEPS = 64, 2048, 250
DT_MS = 4.0                       # nominal ms per cached sample (stride 4 over a 1 ms, 1000 ms trial)
ACTIVE_MIN = 0.005                # keep units firing in at least 0.5% of samples, as the supervisor did

CLASSES = ["Eng-even", "Eng-odd", "Ger-even", "Ger-odd"]


def cache_path(seed, arch):
    """The 4 ms cache for one network. 'hom' resolves to the no-weight-decay retrain where that is the
    clean design's homogeneous partner."""
    if arch == "hom" and seed in NOWD_SEEDS:
        return CACHE / f"4class_seed{seed}_hom_nowd_spk.npy"
    return CACHE / f"4class_seed{seed}_{arch}_spk.npy"


def load_spikes(seed, arch, reshape=True):
    """(n_active, N_TRIALS, T_STEPS) uint8 plus the active-unit indices into the full 64."""
    a = np.load(cache_path(seed, arch))
    assert a.shape == (N_NEURONS, N_TRIALS * T_STEPS), f"{cache_path(seed, arch)} has shape {a.shape}"
    active = np.where(a.mean(1) >= ACTIVE_MIN)[0]
    a = a[active]
    return (a.reshape(len(active), N_TRIALS, T_STEPS) if reshape else a), active


def load_counts(seed, arch):
    """(n_active, N_TRIALS) int per-trial spike counts, the trial-integrated representation."""
    R, active = load_spikes(seed, arch)
    return R.sum(2).astype(int), active


_TARGETS = None


def targets():
    """The three decision variables, aligned to the first N_TRIALS rows of shd_test.h5.

    4-class is the trained task; language and parity are its two factors. Splitting them is what makes the
    lock-in-order question askable at all: the claim under test is about which factor resolves first.
    """
    global _TARGETS
    if _TARGETS is None:
        with h5py.File(SHD_TEST, "r") as f:
            y20 = f["labels"][:N_TRIALS].astype(int)
        _TARGETS = {
            "4-class": (2 * (y20 >= 10).astype(int) + (y20 % 10) % 2).astype(int),
            "language": (y20 >= 10).astype(int),
            "parity": ((y20 % 10) % 2).astype(int),
        }
    return _TARGETS


NK = {"4-class": 4, "language": 2, "parity": 2}


def check_alignment(seed, arch, verbose=False):
    """Cache-to-label alignment, verified rather than assumed.

    The cache ships no label vector, so a silent trial-order mismatch would produce a clean-looking but
    meaningless analysis. A linear decode of the 4-class label from per-trial counts is the arbiter: aligned
    caches land near the published 0.79 / 0.70, a misaligned one sits at the 0.25 chance line.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    C, _ = load_counts(seed, arch)
    Y = targets()["4-class"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    acc = float(cross_val_score(clf, C.T, Y, cv=5).mean())
    if verbose:
        print(f"  seed {seed:>4} {arch:<3}: 4-class decode {acc:.3f} "
              f"({'aligned' if acc > 0.5 else 'CHANCE - MISALIGNED'})")
    return acc


def com_ms(p, dt=DT_MS):
    """Centre of mass of a non-negative time profile, in ms. Used instead of argmax because a single-sample
    peak on a noisy profile moves around; the centre of mass does not."""
    p = np.clip(np.asarray(p, float), 0, None)
    if p.sum() <= 0:
        return float("nan")
    return float((np.arange(len(p)) * p).sum() / p.sum() * dt)


def spread_ms(p, dt=DT_MS):
    """Temporal SD of the same profile, in ms: how sustained the information is."""
    p = np.clip(np.asarray(p, float), 0, None)
    if p.sum() <= 0:
        return float("nan")
    c = com_ms(p, dt) / dt
    return float(np.sqrt((p * (np.arange(len(p)) - c) ** 2).sum() / p.sum()) * dt)


def smooth(p, w=7):
    return np.convolve(np.asarray(p, float), np.ones(w) / w, mode="same")


def ci95(M, axis=0):
    """Mean and half-width of a 95% normal-approximation CI on the seed mean.

    The bands in these figures are CIs on the across-seed mean, not SD bands over seeds. They answer 'where
    is the arm's mean', which is the quantity the paired tests are about. An SD band would be wider and would
    invite the wrong comparison.
    """
    M = np.asarray(M, float)
    n = M.shape[axis]
    m = np.nanmean(M, axis=axis)
    sd = np.nanstd(M, axis=axis, ddof=1)
    return m, 1.96 * sd / np.sqrt(n)


def eta2(a, g, K):
    """Fraction of variance in per-trial activity a explained by grouping g. Chance is about (K-1)/(N-1)."""
    a = np.asarray(a, float)
    grand = a.mean()
    ssb = sum((g == k).sum() * (a[g == k].mean() - grand) ** 2 for k in range(K))
    sst = ((a - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else 0.0


if __name__ == "__main__":
    print(f"cache dir {CACHE}")
    missing = [(s, a) for s in SEEDS for a in ("het", "hom") if not cache_path(s, a).exists()]
    print(f"missing caches: {missing if missing else 'none, all 40 present'}")
    T = targets()
    print("class counts:", {CLASSES[k]: int((T['4-class'] == k).sum()) for k in range(4)})
    if not missing:
        print("\nalignment check (5-fold linear decode of 4-class from per-trial counts, chance 0.25):")
        for s in SEEDS:
            for a in ("het", "hom"):
                check_alignment(s, a, verbose=True)
