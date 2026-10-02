"""Faithful Cohen 2020 / Chung 2018 manifold capacity, using the reference per-manifold anchor code.

The reference package (schung039/neural_manifolds_replicaMFT, `mftma`) ships only the correlation variant,
whose top-level pymanopt import breaks on pymanopt 2.x (solvers -> optimizers). But the capacity, radius and
dimension themselves are computed by `each_manifold_analysis_D1`, which is pymanopt-free (cvxopt QP only). We
shim the one broken import so the module loads, then call that reference routine directly. The only piece we do
NOT run is the low-rank center-correlation SUBTRACTION (fun_FA/CGmanopt, the pymanopt part); we instead report
the raw center-correlation coefficient (res_coeff0), which the reference also computes without pymanopt. So this
is the reference's exact alpha_c / R_M / D_M with correlations reported rather than projected out.

manifold_capacity(XtotT, kappa=0, n_t=200) mirrors manifold_analysis_corr's flow with the center-correlation
projection set to identity (V11 = 0).
"""
import sys as _sys, types as _types
import numpy as np
from scipy.linalg import qr

# ---- shim the broken import, then pull the faithful pymanopt-free anchor routine from the reference package
if "pymanopt.solvers" not in _sys.modules:
    _shim = _types.ModuleType("pymanopt.solvers")
    _shim.ConjugateGradient = object          # never called on our path
    _sys.modules["pymanopt.solvers"] = _shim
from mftma.manifold_analysis_correlation import each_manifold_analysis_D1  # noqa: E402


def center_correlation(centers):
    """Raw mean off-diagonal absolute center correlation (res_coeff0 in the reference), pymanopt-free."""
    N, P = centers.shape
    Xb = centers.T - centers.T.mean(0, keepdims=True)
    q, _ = qr(Xb.T, mode="economic")
    X = Xb @ q[:, : P - 1]
    xnorm = np.sqrt(np.square(X).sum(1, keepdims=True))
    C0 = (X @ X.T) / (xnorm @ xnorm.T)
    return float((np.sum(np.abs(C0)) - P) / (P * (P - 1)))


def manifold_capacity(XtotT, kappa=0, n_t=200, seed=0):
    """Reference alpha_c / R_M / D_M per manifold plus the raw center correlation.

    XtotT: list of P arrays, each (N, m_i) = a manifold's points in feature space.
    Returns dict with alpha_M (overall = 1/mean(1/a_i)), a_vec, R_M, D_M (means over manifolds), center_corr.
    """
    np.random.seed(seed)                       # each_manifold_analysis_D1 samples t-vecs via np.random
    P = len(XtotT)
    Xori = np.concatenate(XtotT, axis=1)
    X_origin = Xori.mean(1, keepdims=True)
    Xtot0 = [XtotT[i] - X_origin for i in range(P)]
    centers = np.stack([XtotT[i].mean(1) for i in range(P)], axis=1)   # (N, P)
    ccorr = center_correlation(centers)

    a_vec, R_vec, D_vec = np.zeros(P), np.zeros(P), np.zeros(P)
    for i in range(P):
        Xr = Xtot0[i]                          # (N, m), center kept relative to global mean
        c = Xr.mean(1)                         # this manifold's center offset
        cn = np.linalg.norm(c) + 1e-12
        S = (Xr - c.reshape(-1, 1)) / cn       # center-normalised points
        D, m = S.shape
        if D > m:
            Q, _ = qr(S, mode="economic")
            S = Q.T @ S
            D, m = S.shape
        sD1 = np.concatenate([S, np.ones((1, m))], axis=0)
        a_vec[i], R_vec[i], D_vec[i] = each_manifold_analysis_D1(sD1, kappa, n_t)

    a_ok = a_vec[np.isfinite(a_vec) & (a_vec > 0)]
    alpha_M = float(1.0 / np.mean(1.0 / a_ok)) if a_ok.size else float("nan")
    return dict(alpha_M=alpha_M, a_mean=float(np.nanmean(a_vec)),
                R_M=float(np.nanmean(R_vec)), D_M=float(np.nanmean(D_vec)),
                center_corr=ccorr, a_vec=a_vec.tolist())


if __name__ == "__main__":
    # validation: capacity must fall as manifolds grow in radius or intrinsic dimension; R_M/D_M must track them
    rng = np.random.default_rng(0)
    N, P, m = 64, 20, 50

    def bank(radius, dim):
        out = []
        for _ in range(P):
            c = rng.standard_normal((N, 1)) * 5.0                 # well-separated centers
            B = rng.standard_normal((N, dim))                     # dim intrinsic axes
            coeff = rng.standard_normal((dim, m)) * radius
            out.append(c + B @ coeff)
        return out

    for rad, dim in [(0.3, 3), (1.0, 3), (1.0, 10), (2.0, 10)]:
        r = manifold_capacity(bank(rad, dim), kappa=0, n_t=150)
        print(f"radius {rad}  intrinsic-dim {dim:2d}  ->  alpha_M {r['alpha_M']:.3f}  "
              f"R_M {r['R_M']:.3f}  D_M {r['D_M']:.2f}  center_corr {r['center_corr']:.3f}")
