import numpy as np
from scipy import stats

MAX_EXACT_N = 22


def signflip_p(d, chunk=1 << 16):
    """Two-sided sign-flip permutation p on paired differences. Exact for n <= MAX_EXACT_N."""
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
            signs = np.where((k[:, None] & bit[None, :]) > 0, 1.0, -1.0)
            # signs @ d is a sum, so compare against the observed sum
            ge += int((np.abs(signs @ d) >= obs * n * (1 - 1e-12)).sum())
        return ge / total, 2.0 / total

    rng = np.random.default_rng(0)
    n_mc = 200_000
    ge = 0
    for start in range(0, n_mc, chunk):
        b = min(chunk, n_mc - start)
        signs = rng.choice([-1.0, 1.0], size=(b, n))
        ge += int((np.abs(signs @ d) >= obs * n * (1 - 1e-12)).sum())
    return (ge + 1) / (n_mc + 1), 1.0 / (n_mc + 1)


def wilcoxon_floor(n):
    """Smallest attainable two-sided Wilcoxon p at this n."""
    return 2.0 / (1 << n) if n > 0 else float("nan")


def paired_triple(het, hom, label="", higher_is_better=True):
    """Paired t, Wilcoxon and exact sign-flip test for one metric, plus means, delta, dz, wins."""
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
    """Holm-Bonferroni adjusted p-values, order preserved."""
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
        running = max(running, adj)
        out[j] = min(1.0, running)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    het = rng.normal(0.70, 0.02, 20)
    hom = het - rng.uniform(0.05, 0.15, 20)
    r = paired_triple(het, hom, "check")
    assert r["wins"] == 20
    assert abs(r["perm_p"] - 2 / 2**20) < 1e-12
    p, fl = signflip_p(np.array([1.0, -1.0]))
    assert abs(p - 1.0) < 1e-9
    print("ok")
