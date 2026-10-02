# replica mean-field manifold capacity (Chung 2018 PRX; Cohen 2020), correlation-corrected.
# port of the reference implementation (schung039/neural_manifolds_replicaMFT) with the
# Stiefel conjugate-gradient step moved to the pymanopt 2.x API; objective, analytic
# gradient and rng call order unchanged.
from functools import partial

import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import qr

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient

solvers.options["show_progress"] = False
solvers.options["maxiters"] = 1000000
solvers.options["abstol"] = 1e-12
solvers.options["reltol"] = 1e-12
solvers.options["feastol"] = 1e-12

CG_MAX_ITERS = 300   # reference passes 20000; the (P-1) x K Stiefel problems converge far sooner


def square_corrcoeff_full_cost(V, X, grad=True):
    "squared difference in center correlation coefficients after projecting out V"
    P, N = X.shape
    N_v, K = V.shape
    assert N_v == N
    C = X @ X.T
    c = X @ V
    c0 = np.diagonal(C).reshape(P, 1) - np.sum(np.square(c), axis=1, keepdims=True)
    Fmn = np.square(C - c @ c.T) / (c0 @ c0.T)
    cost = np.sum(Fmn) / 2
    if not grad:
        return cost, None
    X1 = X.reshape(1, P, N, 1)
    X2 = X.reshape(P, 1, N, 1)
    C1 = c.reshape(P, 1, 1, K)
    C2 = c.reshape(1, P, 1, K)
    PF1 = ((C - c @ c.T) / (c0 @ c0.T)).reshape(P, P, 1, 1)
    PF2 = (np.square(C - c @ c.T) / np.square(c0 @ c0.T)).reshape(P, P, 1, 1)
    Gmni = -PF1 * C1 * X1
    Gmni += -PF1 * C2 * X2
    Gmni += PF2 * c0.reshape(P, 1, 1, 1) * C2 * X1
    Gmni += PF2 * c0.T.reshape(1, P, 1, 1) * C1 * X2
    return cost, np.sum(Gmni, axis=(0, 1))


def CGmanopt(X, objective_function, A):
    "minimise the objective over the Stiefel manifold, reference analytic gradient"
    manifold = Stiefel(X.shape[0], X.shape[1])

    @pymanopt.function.numpy(manifold)
    def cost(V):
        c, _ = objective_function(V, A)
        return float(c)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(V):
        _, g = square_corrcoeff_full_cost(V, A, grad=True)
        return np.asarray(g, dtype=float)

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    optimizer = ConjugateGradient(max_iterations=CG_MAX_ITERS, min_gradient_norm=1e-6,
                                  min_step_size=1e-10, verbosity=0)
    return optimizer.run(problem, initial_point=X).point, None


def fun_FA(centers, maxK, n_repeats):
    "low-rank center structure: optimal V per rank K, residual correlation per K"
    N, P = centers.shape
    Xb = centers.T - np.mean(centers.T, axis=0, keepdims=True)
    xbnorm = np.sqrt(np.square(Xb).sum(axis=1, keepdims=True))
    q, r = qr(Xb.T, mode="economic")
    X = Xb @ q[:, 0:P - 1]
    xnorm = np.sqrt(np.square(X).sum(axis=1, keepdims=True))
    C0 = (X @ X.T) / (xnorm @ xnorm.T)
    res_coeff0 = (np.sum(np.abs(C0)) - P) / (P * (P - 1))

    V1_mat, norm_coeff, norm_coeff_vec, res_coeff = [], [], [], []
    V1 = None
    for i in range(1, maxK + 1):
        best_stability = 0
        for j in range(n_repeats):
            s = np.random.randn(P, 1)
            sX = s.T @ X
            V0 = sX if V1 is None else np.concatenate([sX, V1.T], axis=0)
            V0, _ = qr(V0.T, mode="economic")
            V1tmp, _ = CGmanopt(V0, partial(square_corrcoeff_full_cost, grad=False), X)
            assert np.linalg.norm(V1tmp.T @ V1tmp - np.identity(i), ord="fro") < 1e-10
            X0 = X - (X @ V1tmp) @ V1tmp.T
            stability = min(np.sqrt(np.sum(np.square(X0), axis=1))
                            / np.sqrt(np.sum(np.square(X), axis=1)))
            if stability > best_stability:
                best_stability = stability
                best_V1 = V1tmp
        V1 = best_V1
        X0 = X - (X @ V1) @ V1.T
        xnorm = np.sqrt(np.square(X0).sum(axis=1, keepdims=True))
        C0 = (X0 @ X0.T) / (xnorm @ xnorm.T)
        res_coeff.append((np.sum(np.abs(C0)) - P) / (P * (P - 1)))
        V1_mat.append(V1)
        norm_coeff.append((xnorm / xbnorm)[:, 0])
        norm_coeff_vec.append(np.mean(xnorm / xbnorm))
        # stop after three consecutive cost increases
        if i > 4 and res_coeff[i - 1] > res_coeff[i - 2] > res_coeff[i - 3] > res_coeff[i - 4]:
            break
    return norm_coeff, norm_coeff_vec, q[:, 0:P - 1], V1_mat, res_coeff, res_coeff0


def maxproj(t_vec, sD1, sc=1):
    "manifold point with the largest projection onto each t vector"
    D1, n_t = t_vec.shape
    D = D1 - 1
    s0 = np.zeros((D1, n_t))
    gt = np.zeros(n_t)
    for i in range(n_t):
        t = t_vec[:, i]
        max_S = np.argmax(np.dot(t[0:D], sD1[0:D]))
        s0[:, i] = np.append(sD1[0:D, max_S], [sc])
        gt[i] = np.dot(t, s0[:, i])
    return s0, gt


def minimize_vt_sq(t, sD1, kappa=0):
    "constrained minimisation of 0.5*V^2 - T*V (cvxopt qp), sec IIIa of the 2018 PRX paper"
    D1 = t.shape[0]
    m = sD1.shape[1]
    P = matrix(np.identity(D1))
    q = matrix(-t.astype(np.double))
    G = matrix(sD1.T.astype(np.double))
    h = matrix((-np.ones(m) * kappa).astype(np.double))
    output = solvers.qp(P, q, G, h)
    v_f = np.array(output["x"])
    vt_f = output["primal objective"]
    exitflag = 1 if output["status"] == "optimal" else 0
    alphar = np.array(output["z"])
    normvt2 = np.square(v_f - t).sum()
    return v_f, vt_f, exitflag, alphar, normvt2


def each_manifold_analysis_D1(sD1, kappa, n_t, eps=1e-8):
    "capacity a_Mfull, anchor radius R_M and anchor dimension D_M for one manifold"
    D1, m = sD1.shape
    D = D1 - 1
    t_vec = np.random.randn(D1, n_t)
    ss, gg = maxproj(t_vec, sD1)

    s_all = np.empty((D1, n_t))
    for i in range(n_t):
        t = t_vec[:, i].reshape(-1, 1)
        if gg[i] + kappa < 0:
            # v = t already satisfies the constraints
            s_f = ss[:, i].reshape(-1, 1)
        else:
            v_f, _, _, alpha, _ = minimize_vt_sq(t, sD1, kappa=kappa)
            if np.linalg.norm(v_f - t) < eps:
                s_f = ss[:, i].reshape(-1, 1)
            else:
                s_f = (t - v_f) / np.sum(alpha)
        s_all[:, i] = s_f[:, 0]

    # eq. 16-17 (capacity), eq. 28 (radius), eq. 29 (dimension) of the 2018 PRX paper
    max_ts = np.maximum(np.sum(t_vec * s_all, axis=0) + kappa, np.zeros(n_t))
    s_sum = np.sum(np.square(s_all), axis=0)
    lamb = np.asarray([max_ts[i] / s_sum[i] if s_sum[i] > 0 else 0 for i in range(n_t)])
    a_Mfull = 1 / np.mean(np.square(lamb) * s_sum)

    ds0 = s_all - s_all.mean(axis=1, keepdims=True)
    ds = ds0[0:-1, :] / s_all[-1, :]
    R_M = np.sqrt(np.mean(np.sum(np.square(ds), axis=0)))

    t_hat = t_vec[0:D, :] / np.sqrt(np.sum(np.square(t_vec[0:D, :]), axis=0, keepdims=True))
    s_hat = s_all[0:D, :] / np.sqrt(np.sum(np.square(s_all[0:D, :]), axis=0, keepdims=True) + 1e-12)
    D_M = D * np.square(np.mean(np.sum(t_hat * s_hat, axis=0)))
    return a_Mfull, R_M, D_M


def manifold_analysis_corr(XtotT, kappa, n_t, n_reps=10):
    "per-manifold capacity/radius/dimension after removing the low-rank center structure"
    num = len(XtotT)
    Xori = np.concatenate(XtotT, axis=1)
    X_origin = np.mean(Xori, axis=1, keepdims=True)
    Xtot0 = [XtotT[i] - X_origin for i in range(num)]
    centers = np.stack([np.mean(XtotT[i], axis=1) for i in range(num)], axis=1)
    center_mean = np.mean(centers, axis=1, keepdims=True)

    UU, SS, VV = np.linalg.svd(centers - center_mean)
    total = np.cumsum(np.square(SS) / np.sum(np.square(SS)))
    maxK = np.argmax([t if t < 0.95 else 0 for t in total]) + 11

    norm_coeff, norm_coeff_vec, Proj, V1_mat, res_coeff, res_coeff0 = fun_FA(centers, maxK, n_reps)
    KK = np.argmin(res_coeff) + 1
    V11 = Proj @ V1_mat[KK - 1]

    a_vec = np.zeros(num)
    R_vec = np.zeros(num)
    D_vec = np.zeros(num)
    for i in range(num):
        Xr = Xtot0[i]
        Xr_ns = Xr - V11 @ (V11.T @ Xr)          # project out the center subspace
        Xr0_ns = np.mean(Xr_ns, axis=1)
        S_r = (Xr_ns - Xr0_ns.reshape(-1, 1)) / np.linalg.norm(Xr0_ns)
        D, m = S_r.shape
        if D > m:
            Q, R = qr(S_r, mode="economic")
            S_r = Q.T @ S_r
            D, m = S_r.shape
        sD1 = np.concatenate([S_r, np.ones((1, m))], axis=0)
        a_vec[i], R_vec[i], D_vec[i] = each_manifold_analysis_D1(sD1, kappa, n_t)
    return a_vec, R_vec, D_vec, res_coeff0, KK


def manifold_capacity_corr(XtotT, kappa=0, n_t=200, n_reps=3, seed=0):
    "corrected capacity alpha_M = 1/mean(1/a_i), mean geometry, residual center correlation"
    np.random.seed(seed)
    a_vec, R_vec, D_vec, res_coeff0, KK = manifold_analysis_corr(XtotT, kappa, n_t, n_reps=n_reps)
    a_ok = a_vec[np.isfinite(a_vec) & (a_vec > 0)]
    alpha_M = float(1.0 / np.mean(1.0 / a_ok)) if a_ok.size else float("nan")
    return dict(alpha_M=alpha_M, a_mean=float(np.nanmean(a_vec)),
                R_M=float(np.nanmean(R_vec)), D_M=float(np.nanmean(D_vec)),
                center_corr=float(res_coeff0), KK=int(KK), a_vec=a_vec.tolist())


if __name__ == "__main__":
    # corrected capacity must fall as injected center correlation rises,
    # with radius and intrinsic dimension held fixed
    rng = np.random.default_rng(0)
    N, P, m, dim, rad = 64, 20, 50, 4, 0.8

    def bank(corr):
        shared = rng.standard_normal((N, 1))
        out = []
        for _ in range(P):
            c = (1 - corr) * rng.standard_normal((N, 1)) * 4 + corr * shared * 4
            B = rng.standard_normal((N, dim))
            out.append(c + B @ (rng.standard_normal((dim, m)) * rad))
        return out

    for corr in (0.0, 0.5, 0.85):
        r = manifold_capacity_corr(bank(corr), kappa=0, n_t=150, n_reps=3)
        print(f"corr {corr:.2f} -> alpha_M {r['alpha_M']:.3f}  R_M {r['R_M']:.3f}  "
              f"D_M {r['D_M']:.2f}  center_corr {r['center_corr']:.3f}")
