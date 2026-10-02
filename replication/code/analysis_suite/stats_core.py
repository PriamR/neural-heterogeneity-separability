"""The paired test triple used for every het-vs-hom comparison in the paper.

Every comparison is paired within seed: the pair shares initial weights, data order and training schedule,
so the only free difference is the hidden membrane time constant. An unpaired test would throw that away.

Three tests, always reported together, so no single test carries a claim:
  paired t          parametric, fragile at n = 10
  Wilcoxon          distribution-free but coarse
  sign-flip perm    exact by full enumeration over 2^n sign assignments (the primary at n = 20)

Both non-parametric tests have a hard floor: the smallest two-sided p obtainable is 2 / 2^n, because only the
all-positive and all-negative flips are as extreme as a unanimous observation. At n = 10 that floor is 0.00195
and at n = 20 it is 1.9e-6. A p sitting at the floor is a maximally significant result, not a weak one, and
`fmt_p` in style.py marks it so a reader cannot mistake the ceiling for a limit of the effect.
"""
import numpy as np
from scipy import stats

MAX_EXACT_N = 22          # 2^22 = 4.2M sign vectors, still cheap in chunks


def signflip_p(d, chunk=1 << 16):
    """Two-sided sign-flip permutation p on paired differences d.

    Exact by full enumeration when n <= MAX_EXACT_N, else Monte Carlo with a fixed seed.
    Statistic is |mean(d)|; under the null each d_i is equally likely to carry either sign.
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0 or np.allclose(d, 0):
        return float("nan"), float("nan")
    obs = abs(d.mean())

    if n <= MAX_EXACT_N:
        total = 1 << n
        ge = 0
        bit = (1 << np.arange(n, dtype=np.uint64))
        for start in range(0, total, chunk):
            k = np.arange(start, min(start + chunk, total), dtype=np.uint64)
            # rows = sign assignments, +1 where the bit is set, -1 where it is not
            signs = np.where((k[:, None] & bit[None, :]) > 0, 1.0, -1.0)
            # signs @ d is a SUM, so compare against the observed sum, not the observed mean
            ge += int((np.abs(signs @ d) >= obs * n * (1 - 1e-12)).sum())
        return ge / total, 2.0 / total          # p, floor

    rng = np.random.default_rng(0)
    n_mc = 200_000
    ge = 0
    for start in range(0, n_mc, chunk):
        b = min(chunk, n_mc - start)
        signs = rng.choice([-1.0, 1.0], size=(b, n))
        ge += int((np.abs(signs @ d) >= obs * n * (1 - 1e-12)).sum())
    return (ge + 1) / (n_mc + 1), 1.0 / (n_mc + 1)


def wilcoxon_floor(n):
    """Smallest attainable two-sided Wilcoxon p at this n (all differences same sign)."""
    return 2.0 / (1 << n) if n > 0 else float("nan")


def paired_triple(het, hom, label="", higher_is_better=True):
    """Full row for one paired metric: means, delta, dz, all three p-values, wins/n."""
    het = np.asarray(het, dtype=np.float64)
    hom = np.asarray(hom, dtype=np.float64)
    ok = np.isfinite(het) & np.isfinite(hom)
    het, hom = het[ok], hom[ok]
    n = het.size
    d = het - hom

    row = {
        "metric": label,
        "n": int(n),
        "het_mean": float(het.mean()) if n else float("nan"),
        "het_sd": float(het.std(ddof=1)) if n > 1 else float("nan"),
        "hom_mean": float(hom.mean()) if n else float("nan"),
        "hom_sd": float(hom.std(ddof=1)) if n > 1 else float("nan"),
        "delta": float(d.mean()) if n else float("nan"),
        "wins": int((d > 0).sum()) if higher_is_better else int((d < 0).sum()),
    }
    sd = d.std(ddof=1) if n > 1 else 0.0
    row["dz"] = float(d.mean() / sd) if sd > 0 else float("nan")

    try:
        row["t_p"] = float(stats.ttest_rel(het, hom).pvalue)
    except Exception:
        row["t_p"] = float("nan")
    try:
        row["w_p"] = float(stats.wilcoxon(het, hom).pvalue)
    except Exception:
        row["w_p"] = float("nan")
    p, floor = signflip_p(d)
    row["perm_p"] = float(p)
    row["perm_floor"] = float(floor)
    row["w_floor"] = float(wilcoxon_floor(n))
    return row


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved. Used to report the scoreboard's family-wise view
    alongside raw p, since the scoreboard runs many metrics."""
    p = np.asarray(pvals, dtype=np.float64)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    m = idx.size
    out = np.full(p.shape, np.nan)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for i, j in enumerate(order):
        adj = (m - i) * p[j]
        running = max(running, adj)          # enforce monotonicity
        out[j] = min(1.0, running)
    return out


def unpaired_triple(a, b, label=""):
    """For comparisons that are genuinely unpaired (a fixed-tau rung against the learned-hom baseline is
    paired; but rung-vs-rung across different seeds sets is not). Welch t plus Mann-Whitney."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    row = {"metric": label, "n_a": int(a.size), "n_b": int(b.size),
           "a_mean": float(a.mean()), "b_mean": float(b.mean()), "delta": float(a.mean() - b.mean())}
    try:
        row["welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        row["welch_p"] = float("nan")
    try:
        row["mw_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        row["mw_p"] = float("nan")
    return row


if __name__ == "__main__":
    # self-check: a unanimous effect must sit exactly on the floor
    rng = np.random.default_rng(0)
    het = rng.normal(0.70, 0.02, 20)
    hom = het - rng.uniform(0.05, 0.15, 20)          # het wins 20/20 by construction
    r = paired_triple(het, hom, "self-check")
    print(r)
    assert r["wins"] == 20
    assert abs(r["perm_p"] - 2 / 2**20) < 1e-12, "unanimous n=20 must hit the 1.9e-6 floor"
    d = np.array([1.0, -1.0])                        # perfectly symmetric -> p = 1
    p, fl = signflip_p(d)
    assert abs(p - 1.0) < 1e-9, p
    print("stats_core self-check passed")
