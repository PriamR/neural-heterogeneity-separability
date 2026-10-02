"""Two more dimensionality measures at the clean n = 20, het vs hom: nonlinear intrinsic dimension and the
eigenspectrum power-law exponent.

Both target the same claim from a different angle than PR / n90 / capacity:

  TwoNN intrinsic dimension (Facco 2017): a nearest-neighbour ID estimator that captures curved manifolds a
  linear variance measure undercounts. Computed on the 64-D rate code (the representation used by the shattering
  and capacity analyses). Higher = the trial cloud occupies more intrinsic directions. Expected higher for het.

  Eigenspectrum power-law exponent (Stringer 2019): the covariance eigenvalues of the time-resolved state decay
  as lambda_n ~ n^-alpha; a smaller alpha is a heavier tail and a higher effective dimension. Computed on the
  same time-resolved state matrix (rows = trial x time bin, cols = 64 neurons) that the paper's scree and n90
  use, so alpha is a one-number summary of that figure. Expected lower for het.

Run: python code/analysis_suite/run_dimension_measures_n20.py
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors

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
PL_LO, PL_HI = 2, 50    # rank band for the power-law fit (avoid the very top mode and the 64-dim floor)


def twonn(X, discard_frac=0.10):
    """TwoNN intrinsic dimension (Facco 2017). ID is the through-origin slope of -log(1-F) vs log(mu),
    mu = r2/r1 the ratio of the two nearest-neighbour distances; the top discard_frac of mu is dropped."""
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)                     # d[:,0] = 0 (self), d[:,1] = r1, d[:,2] = r2
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = np.sort(mu[mu > 1 + 1e-9])
    N = mu.size
    F = np.arange(1, N + 1) / (N + 1)
    keep = int((1 - discard_frac) * N)
    x = np.log(mu[:keep])
    y = -np.log(1 - F[:keep])
    return float(np.sum(x * y) / np.sum(x * x))


def powerlaw_alpha(states, lo=PL_LO, hi=PL_HI):
    """Exponent alpha of lambda_n ~ n^-alpha, fit by least squares in log-log over ranks [lo, hi]."""
    C = np.cov(states.T)
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    eig = eig[eig > 0]
    n = np.arange(1, eig.size + 1)
    hi = min(hi, eig.size)
    sel = slice(lo - 1, hi)
    slope = np.polyfit(np.log(n[sel]), np.log(eig[sel]), 1)[0]
    return float(-slope), n, eig


def main():
    per = {"het": {}, "hom": {}}
    spectra = {"het": [], "hom": []}
    print(f"{'seed':>5} {'arch':>4} | {'TwoNN_ID':>8} {'PL_alpha':>8}")
    for seed in SEEDS:
        for arch in ("het", "hom"):
            d = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")
            rate = d["rate"].astype(float)                          # (N, 64)
            hidden = d["hidden"].astype(float)                      # (N, 50, 64)
            states = hidden.reshape(-1, hidden.shape[-1])           # (N*50, 64) time-resolved
            idim = twonn(rate)
            alpha, n_rank, eig = powerlaw_alpha(states)
            per[arch][seed] = {"twonn_id": idim, "pl_alpha": alpha}
            sp = np.zeros(hidden.shape[-1])                    # fixed length so seeds stack
            e = eig / eig.sum()
            sp[: e.size] = e
            spectra[arch].append(sp)
            print(f"{seed:>5} {arch:>4} | {idim:>8.3f} {alpha:>8.3f}", flush=True)

    V = lambda arch, k: np.array([per[arch][s][k] for s in SEEDS], float)
    METRICS = [("twonn_id", "TwoNN intrinsic dim", True), ("pl_alpha", "power-law exponent alpha", False)]
    triples = {}
    print("\n--- paired het vs hom (n = 20) ---")
    for k, lbl, hib in METRICS:
        r = paired_triple(V("het", k), V("hom", k), lbl, higher_is_better=hib)
        triples[k] = r
        arrow = "het>hom" if hib else "het<hom"
        print(f"{lbl:>26} ({arrow}): het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  "
              f"dz {r['dz']:+.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}",
              flush=True)

    # ================================================================= F4.17 two measures + spectrum
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (k, lbl, hib) in zip(axes[:2], METRICS):
        r = triples[k]
        ax.bar([0, 1], [r["het_mean"], r["hom_mean"]], 0.6, yerr=[r["het_sd"], r["hom_sd"]],
               color=[S.HET, S.HOM], capsize=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels([S.LABEL["het"], S.LABEL["hom"]])
        good = "higher = more dimensions" if hib else "lower = heavier tail"
        ax.set_ylabel(lbl); ax.set_title(f"{lbl}\n({good})", fontsize=10)
        ax.text(0.5, 0.97, f"$\\Delta$ {r['delta']:+.3f}  $d_z$ {r['dz']:+.2f}\n"
                           f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  {r['wins']}/{r['n']}",
                ha="center", va="top", transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    ax = axes[2]
    for arch, col in [("het", S.HET), ("hom", S.HOM)]:
        M = np.stack(spectra[arch]); m = M.mean(0)
        n_rank = np.arange(1, m.size + 1)
        ax.loglog(n_rank, m, color=col, label=S.LABEL[arch], lw=2)
    ax.axvspan(PL_LO, PL_HI, color=S.CHANCE, alpha=0.10, label=f"fit band [{PL_LO},{PL_HI}]")
    ax.set_xlabel("PC rank"); ax.set_ylabel("normalised variance")
    ax.set_title("Eigenspectrum (time-resolved state)", fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("Intrinsic dimension and spectral decay, het vs hom (n = 20)", fontweight="bold", y=1.03)
    S.save(fig, FIGS / "F4_17_intrinsic_dim_spectrum.png")

    summary = {
        "analysis": "intrinsic_dimension_and_powerlaw", "n_pairs": len(SEEDS),
        "twonn_on": "rate code (N,64)", "powerlaw_on": "time-resolved state (N*50,64)",
        "powerlaw_fit_band": [PL_LO, PL_HI],
        "by_arch": {arch: {k: {"mean": float(V(arch, k).mean()), "sd": float(V(arch, k).std(ddof=1))}
                           for k, _, _ in METRICS} for arch in ("het", "hom")},
        "paired": triples,
        "per_network": {arch: {int(s): per[arch][s] for s in SEEDS} for arch in ("het", "hom")},
    }
    json.dump(summary, open(OUT / "dimension_measures_n20_summary.json", "w"), indent=1)
    print(f"\nwrote {OUT/'dimension_measures_n20_summary.json'} and F4_17_intrinsic_dim_spectrum.png")


if __name__ == "__main__":
    main()
