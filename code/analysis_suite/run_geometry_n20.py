"""Recompute the geometry analyses that were still at n = 10 (CCGP / abstraction, manifold capacity, memory)
at the clean n = 20, on the no-WD homs, with the paired test triple.

Consumes the clean representation cache from extract_geometry.py (data_cache/representation_clean), which was
built on the CLEAN homs for all twenty seeds. Writes docs/analysis_suite/geometry_n20_summary.json and the
figures F4_6_ccgp.png and F4_14_memory.png. Pure numpy/sklearn (no torch in-process).

Run: python code/analysis_suite/run_geometry_n20.py
"""
import json, sys, itertools
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
rng = np.random.default_rng(0)

# ---- CCGP dichotomies over the 4 factorial conditions {0,1,2,3} = 2*is_german + is_odd
DICH = {"Language": {0: +1, 1: +1, 2: -1, 3: -1},
        "Parity":   {0: +1, 1: -1, 2: +1, 3: -1},
        "XOR":      {0: +1, 1: -1, 2: -1, 3: +1}}
CONTEXTS = {"Language": [(0, 2), (1, 3)], "Parity": [(0, 1), (2, 3)]}
BIN_MS, MAXLAG = 20, 25


def load(seed, arch):
    return np.load(CACHE / f"repr_seed{seed}_{arch}.npz")


def clf():
    return LogisticRegression(max_iter=1000, C=1.0)


def decodability(X, lab):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    accs = []
    for tr, te in cv.split(X, lab):
        sc = StandardScaler().fit(X[tr])
        m = clf().fit(sc.transform(X[tr]), lab[tr])
        accs.append(m.score(sc.transform(X[te]), lab[te]))
    return float(np.mean(accs))


def ccgp(X, y, dich):
    plus = [c for c, s in dich.items() if s > 0]
    minus = [c for c, s in dich.items() if s < 0]
    accs = []
    for tp, tm in itertools.product(plus, minus):
        te_plus = [c for c in plus if c != tp]
        te_minus = [c for c in minus if c != tm]
        tr_mask = np.isin(y, [tp, tm]); te_mask = np.isin(y, te_plus + te_minus)
        ytr = np.array([dich[c] for c in y[tr_mask]]); yte = np.array([dich[c] for c in y[te_mask]])
        sc = StandardScaler().fit(X[tr_mask])
        m = clf().fit(sc.transform(X[tr_mask]), ytr)
        accs.append(m.score(sc.transform(X[te_mask]), yte))
    return float(np.mean(accs))


def parallelism(X, y, name):
    cents = {c: X[y == c].mean(0) for c in range(4)}
    if name == "Language":
        v1 = cents[2] - cents[0]; v2 = cents[3] - cents[1]
    else:
        v1 = cents[1] - cents[0]; v2 = cents[3] - cents[2]
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))


def pr(eigs):
    e = np.asarray(eigs, float); e = e[e > 0]
    return float((e.sum() ** 2) / np.square(e).sum()) if e.size else 0.0


def manifold_geometry(rate, y):
    R, D = [], []
    for c in np.unique(y):
        Xc = rate[y == c]
        if Xc.shape[0] < 3:
            continue
        x0 = Xc.mean(0)
        eigs = np.linalg.eigvalsh(np.cov((Xc - x0).T)); eigs = eigs[eigs > 1e-12]
        R.append(np.sqrt(eigs.sum()) / (np.linalg.norm(x0) + 1e-9))
        D.append(pr(eigs))
    return float(np.mean(R)), float(np.mean(D))


def separable(X, lab):
    # Points in general position with fewer samples than feature dimensions are always linearly separable,
    # so skip the solver for that trivial case (it dominates the low-P rungs).
    if X.shape[0] <= X.shape[1]:
        return True
    c = LinearSVC(C=1e3, max_iter=4000, tol=1e-3, dual=True, random_state=0)
    try:
        c.fit(X, lab)
        return c.score(X, lab) > 0.9999
    except Exception:
        return False


def empirical_capacity(binned, n_pts=8, P_grid=(6, 9, 12, 16, 20, 24, 32, 40, 52, 64), reps=15):
    """Trial-as-manifold capacity: manifold = n_pts time-bin points of one trial in 64-D. Sweep P manifolds,
    random +/- labels, measure fraction linearly separable; alpha_c = P/N at frac = 0.5.
    Reports the full frac curve so the grid floor (smallest P/N = min(P_grid)/N) is diagnosable."""
    n_trials, NB, N = binned.shape
    Ps = [P for P in P_grid if P <= n_trials]
    fracs = []
    for P in Ps:
        ok = 0
        for _ in range(reps):
            mi = rng.choice(n_trials, size=P, replace=False)
            bins_sel = rng.choice(NB, size=n_pts, replace=False)
            pts = binned[mi][:, bins_sel, :].reshape(P * n_pts, N)
            lab = np.repeat(rng.integers(0, 2, size=P) * 2 - 1, n_pts)
            pts = StandardScaler().fit_transform(pts)
            ok += separable(pts, lab)
        fracs.append(ok / reps)
    Ps = np.array(Ps, float); fracs = np.array(fracs); alphas = Ps / N
    ac = np.nan
    for i in range(len(fracs) - 1):
        if fracs[i] >= 0.5 >= fracs[i + 1]:
            f0, f1, a0, a1 = fracs[i], fracs[i + 1], alphas[i], alphas[i + 1]
            ac = a0 + (a1 - a0) * (f0 - 0.5) / (f0 - f1 + 1e-9)
            break
    floored = False
    if np.isnan(ac):
        floored = True
        ac = alphas[-1] if fracs[-1] > 0.5 else alphas[0]
    return float(ac), alphas.tolist(), fracs.tolist(), bool(floored)


def memory_function(hidden, u, maxlag=MAXLAG):
    n, NB, D = hidden.shape
    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    r2 = []
    for tau in range(maxlag + 1):
        Xtr = hidden[tr, tau:, :].reshape(-1, D); ytr = u[tr, :NB - tau].reshape(-1)
        Xte = hidden[te, tau:, :].reshape(-1, D); yte = u[te, :NB - tau].reshape(-1)
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
        pred = m.predict(sc.transform(Xte))
        ss_res = np.sum((yte - pred) ** 2); ss_tot = np.sum((yte - yte.mean()) ** 2) + 1e-12
        r2.append(max(0.0, 1 - ss_res / ss_tot))
    return np.array(r2)


# ================================================================= per-network pass
per = {"het": {}, "hom": {}}
mc_curves = {"het": [], "hom": []}
print(f"{'seed':>5} {'arch':>4} | {'shatter':>7} {'ccgp':>6} {'ccgp0':>6} | {'R_M':>5} {'D_M':>5} "
      f"{'a_c':>5} {'flr':>3} | {'MC':>5}")
for seed in SEEDS:
    for arch in ("het", "hom"):
        d = load(seed, arch)
        X = d["rate"].astype(float); y = d["y"].astype(int)
        binned = d["binned"].astype(float); hidden = d["hidden"].astype(float); u = d["u"].astype(float)
        dec = {k: decodability(X, np.array([DICH[k][c] for c in y])) for k in DICH}
        cc = {k: ccgp(X, y, DICH[k]) for k in DICH}
        yshuf = y.copy(); rng.shuffle(yshuf)
        cc_null = float(np.mean([ccgp(X, yshuf, DICH[k]) for k in DICH]))
        ps = {k: parallelism(X, y, k) for k in CONTEXTS}
        Xs = StandardScaler().fit_transform(X)
        lin = float(cross_val_score(clf(), Xs, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())
        R_M, D_M = manifold_geometry(X, y)
        ac, alphas, fracs, floored = empirical_capacity(binned)
        mc = memory_function(hidden, u)
        shatter = float(np.mean([dec["Language"], dec["Parity"], dec["XOR"]]))
        per[arch][seed] = dict(
            dec_Language=dec["Language"], dec_Parity=dec["Parity"], dec_XOR=dec["XOR"], shatter=shatter,
            ccgp_Language=cc["Language"], ccgp_Parity=cc["Parity"], ccgp_XOR=cc["XOR"], ccgp_mean=float(np.mean(list(cc.values()))),
            ccgp_null=cc_null, PS_Language=ps["Language"], PS_Parity=ps["Parity"],
            lin_decode=lin, R_M=R_M, D_M=D_M, alpha_c=ac, alpha_floored=floored,
            MC=float(mc.sum()), temporal_dim=None)
        mc_curves[arch].append(mc)
        print(f"{seed:>5} {arch:>4} | {shatter:>7.3f} {np.mean(list(cc.values())):>6.3f} {cc_null:>6.3f} | "
              f"{R_M:>5.2f} {D_M:>5.2f} {ac:>5.3f} {'Y' if floored else 'n':>3} | {mc.sum():>5.2f}", flush=True)

V = lambda arch, k: np.array([per[arch][s][k] for s in SEEDS], float)

# ================================================================= paired triples
print("\n--- paired het vs hom (n=20) ---")
METRICS = [("Shattering (mean decodability)", "shatter"), ("CCGP (mean)", "ccgp_mean"),
           ("CCGP null (shuffle)", "ccgp_null"), ("XOR decodability", "dec_XOR"),
           ("Parallelism (Language)", "PS_Language"), ("Manifold radius R_M", "R_M"),
           ("Manifold dimension D_M", "D_M"), ("Manifold capacity alpha_c", "alpha_c"),
           ("Memory capacity MC", "MC")]
triples = {}
for label, k in METRICS:
    r = paired_triple(V("het", k), V("hom", k), label)
    triples[k] = r
    print(f"{label:>34}: het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  "
          f"dz {r['dz']:+.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}")

n_floored_het = int(V("het", "alpha_floored").sum())
n_floored_hom = int(V("hom", "alpha_floored").sum())
print(f"\nalpha_c grid floor (P/N min = 6/64 = 0.094): het floored {n_floored_het}/20, hom floored {n_floored_hom}/20")

# ================================================================= F4.6 CCGP / abstraction
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
cats = ["Language", "Parity", "XOR"]
xp = np.arange(3)
for off, arch, col in [(-0.19, "het", S.HET), (0.19, "hom", S.HOM)]:
    m = [V(arch, f"dec_{c}").mean() for c in cats]
    sd = [V(arch, f"dec_{c}").std(ddof=1) for c in cats]
    ax.bar(xp + off, m, 0.36, yerr=sd, color=col, label=S.LABEL[arch], capsize=3)
ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="chance (binary)")
ax.set_xticks(xp); ax.set_xticklabels(cats)
ax.set_ylabel("cross-validated decodability")
ax.set_title("Shattering: cross-validated decodability by dichotomy")
ax.set_ylim(0.45, 1.0); ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
ccats = ["Language", "Parity", "XOR"]
xpc = np.arange(3)
for off, arch, col in [(-0.19, "het", S.HET), (0.19, "hom", S.HOM)]:
    m = [V(arch, f"ccgp_{c}").mean() for c in ccats]
    sd = [V(arch, f"ccgp_{c}").std(ddof=1) for c in ccats]
    ax.bar(xpc + off, m, 0.36, yerr=sd, color=col, label=S.LABEL[arch], capsize=3)
ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="shuffle null / chance")
ax.set_xticks(xpc); ax.set_xticklabels(ccats)
ax.set_ylabel("cross-condition generalisation (CCGP)")
ax.set_title("Cross-condition generalisation (CCGP) by dichotomy")
ax.set_ylim(0.30, 0.75); ax.legend(loc="upper right", fontsize=8)
fig.suptitle("CCGP and shattering, het vs hom (n = 20)", fontweight="bold", y=1.02)
S.save(fig, FIGS / "F4_6_ccgp.png")

# ================================================================= F4.14 memory
lags = np.arange(MAXLAG + 1) * BIN_MS
fig, ax = plt.subplots(figsize=(7, 4.8))
for arch, col in [("het", S.HET), ("hom", S.HOM)]:
    M = np.stack(mc_curves[arch]); m = M.mean(0); sd = M.std(0)
    S.band(ax, lags, m, sd, col, label=f"{S.LABEL[arch]} (MC = {V(arch, 'MC').mean():.1f})", marker="o", ms=3)
ax.set_xlabel("lag into the past (ms)")
ax.set_ylabel("R^2 reconstructing past input drive")
ax.set_title("Network memory function")
ax.legend(loc="upper right")
S.save(fig, FIGS / "F4_14_memory.png")

# ================================================================= summary json
summary = {
    "n_pairs": len(SEEDS),
    "hom_mode": "clean (no-WD retrains for 202..930)",
    "by_arch": {arch: {k: {"mean": float(V(arch, k).mean()), "std": float(V(arch, k).std(ddof=1))}
                       for k in ["shatter", "dec_Language", "dec_Parity", "dec_XOR", "ccgp_mean", "ccgp_null",
                                 "PS_Language", "PS_Parity", "lin_decode", "R_M", "D_M", "alpha_c", "MC"]}
                for arch in ("het", "hom")},
    "paired": {k: triples[k] for k in triples},
    "alpha_c_floored": {"het": n_floored_het, "hom": n_floored_hom, "grid_floor": 6 / 64},
    "per_network": {arch: {int(s): per[arch][s] for s in SEEDS} for arch in ("het", "hom")},
}
json.dump(summary, open(OUT / "geometry_n20_summary.json", "w"), indent=1)
print(f"\nwrote {OUT/'geometry_n20_summary.json'} and figures F4_6_ccgp.png, F4_14_memory.png")
