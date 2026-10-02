"""Correlation-corrected Cohen 2020 manifold capacity: the faithful headline alpha_c.

This runs the reference manifold_analysis_corr in full, INCLUDING the low-rank center-correlation subtraction
(fun_FA/CGmanopt) that mftma_capacity.py omits. That step broke on pymanopt 2.x (the reference targets the old
0.2 API: pymanopt.solvers, solver.solve, Problem(cost=...)). We monkeypatch the module's CGmanopt with a
pymanopt-2.x version that uses the reference's own analytic gradient (numpy backend, not autograd, so the
optimisation is the reference's exact objective and gradient). CG iterations are capped for tractability; the
Stiefel problems are tiny (P-1 x K, P <= 20) and converge quickly.

manifold_capacity_corr(XtotT, kappa=0, n_t=200, n_reps=3) returns the corrected per-manifold capacity plus the
same geometry (R_M, D_M) and the residual center correlation the reference reports.
"""
import sys as _sys, types as _types
import numpy as np

# ---- shim the broken top-level import so the reference module loads
if "pymanopt.solvers" not in _sys.modules:
    _shim = _types.ModuleType("pymanopt.solvers")
    _shim.ConjugateGradient = object
    _sys.modules["pymanopt.solvers"] = _shim

import pymanopt                                             # noqa: E402
from pymanopt.manifolds import Stiefel                      # noqa: E402
from pymanopt.optimizers import ConjugateGradient           # noqa: E402
import mftma.manifold_analysis_correlation as _mac          # noqa: E402

CG_MAX_ITERS = 300      # reference passes 20000; the tiny Stiefel problems converge far sooner


def _CGmanopt_v2(X, objective_function, A, **kwargs):
    """pymanopt-2.x replacement for the reference CGmanopt, using the reference's own analytic gradient."""
    n, k = X.shape
    manifold = Stiefel(n, k)

    @pymanopt.function.numpy(manifold)
    def cost(V):
        c, _ = objective_function(V, A, grad=False)
        return float(c)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(V):
        _, g = _mac.square_corrcoeff_full_cost(V, A, grad=True)
        return np.asarray(g, dtype=float)

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    optimizer = ConjugateGradient(max_iterations=CG_MAX_ITERS, min_gradient_norm=1e-6,
                                  min_step_size=1e-10, verbosity=0)
    res = optimizer.run(problem, initial_point=X)
    return res.point, None


# swap the reference's CGmanopt (fun_FA calls the module global) for the 2.x version
_mac.CGmanopt = _CGmanopt_v2


def manifold_capacity_corr(XtotT, kappa=0, n_t=200, n_reps=3, seed=0):
    """Full correlation-corrected capacity via the reference manifold_analysis_corr.

    Returns dict: alpha_M (overall = 1/mean(1/a_i)), a_mean, R_M, D_M (means over manifolds),
    center_corr (res_coeff0, raw), KK (rank of removed low-rank center structure).
    """
    np.random.seed(seed)
    a_vec, R_vec, D_vec, res_coeff0, KK = _mac.manifold_analysis_corr(XtotT, kappa, n_t, n_reps=n_reps)
    a_ok = a_vec[np.isfinite(a_vec) & (a_vec > 0)]
    alpha_M = float(1.0 / np.mean(1.0 / a_ok)) if a_ok.size else float("nan")
    return dict(alpha_M=alpha_M, a_mean=float(np.nanmean(a_vec)),
                R_M=float(np.nanmean(R_vec)), D_M=float(np.nanmean(D_vec)),
                center_corr=float(res_coeff0), KK=int(KK), a_vec=a_vec.tolist())


if __name__ == "__main__":
    # validation: with radius/dimension held fixed, correlated centers must give LOWER corrected capacity
    # than decorrelated centers. This is the effect the omitted step folds back in.
    rng = np.random.default_rng(0)
    N, P, m, dim, rad = 64, 20, 50, 4, 0.8

    def bank(corr):
        # shared direction injected into every center with weight `corr` -> center correlation
        shared = rng.standard_normal((N, 1))
        out = []
        for _ in range(P):
            c = (1 - corr) * rng.standard_normal((N, 1)) * 4 + corr * shared * 4
            B = rng.standard_normal((N, dim))
            out.append(c + B @ (rng.standard_normal((dim, m)) * rad))
        return out

    for corr in (0.0, 0.5, 0.85):
        r = manifold_capacity_corr(bank(corr), kappa=0, n_t=150, n_reps=3)
        print(f"injected-corr {corr:.2f} -> alpha_M {r['alpha_M']:.3f}  R_M {r['R_M']:.3f}  "
              f"D_M {r['D_M']:.2f}  center_corr {r['center_corr']:.3f}  KK {r['KK']}")
