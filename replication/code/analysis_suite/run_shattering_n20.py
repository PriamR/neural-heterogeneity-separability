"""Shattering dimensionality (Bernardi 2020) at the clean n = 20, het vs hom.

The existing shatter metric in run_geometry_n20.py averages decodability over just the three factorial
dichotomies of the 4-class label. This generalises it to the Bernardi shattering dimensionality: the
cross-validated decodability of a linear readout averaged over ALL balanced dichotomies of a condition set.
A representation that decodes more random dichotomies is more expressive (higher-dimensional). The headline is
one scalar per network -> two bars, het vs hom, with the paired test triple across the twenty seeds.

We run it over two condition sets, exactly as discussed:
  A  8 conditions  = language x parity x magnitude (digit<5 vs >=5)   -> all 35 balanced 4-vs-4 dichotomies
  B  20 conditions = the original digit x language classes            -> N_SAMPLE_B sampled 10-vs-10 dichotomies

Labelling follows Bernardi: the +/- label attaches to a CONDITION, every trial inherits its condition's label,
and decoding is cross-validated across trials. Conditions are balanced to equal trial counts so every dichotomy
has chance exactly 0.5. The identical dichotomy list and the identical shared trial subsample are applied to all
forty networks, so the het-vs-hom comparison stays paired within seed.

The fine (digit, language) label per trial is recovered from shd_test.h5 read in file order, which matches the
cache row order because extract_geometry.py builds the cache with shuffle=False, drop_last=False, class_list =
range(20). The recovery is asserted against the cached 4-class label: 2*(label//10) + (label%2) must equal y.

Decoder and CV are the house decodability() from run_geometry_n20.py (5-fold logistic), so the new SD sits on
the same footing as the existing 3-dichotomy shatter. Bernardi used a max-margin SVM; swapping clf() to
LinearSVC is a one-line change and does not move the paired comparison.

Run: python code/analysis_suite/run_shattering_n20.py
"""
import json, sys, itertools
from pathlib import Path
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))

CAP_PER_COND = 100      # trials kept per condition after balancing (shared across all nets)
N_SAMPLE_B = 200        # sampled balanced dichotomies for the 20-condition set (all 35 used for the 8-set)
N_JOBS = -1
SEED_TRIALS = 0         # rng for the shared trial subsample
SEED_DICH = 1           # rng for sampling set-B dichotomies


# ------------------------------------------------------------------ fine labels (digit, language) per trial
def load_fine_labels():
    with h5py.File(SHD_TEST, "r") as f:
        lab = np.asarray(f["labels"][:]).astype(int)     # 0..19, file order
    assert lab.min() == 0 and lab.max() == 19, f"unexpected SHD label range {lab.min()}..{lab.max()}"
    digit = lab % 10                                     # 0..9
    lang = lab // 10                                     # 0 english, 1 german
    return lab, digit, lang


RAW_LABEL, DIGIT, LANG = load_fine_labels()


def condition_sets():
    """Return {name: (cond_array, n_conditions)} for the two condition definitions, per trial, cache-aligned."""
    parity = DIGIT % 2
    magnitude = (DIGIT >= 5).astype(int)
    condA = 4 * LANG + 2 * parity + magnitude            # 0..7  (language x parity x magnitude)
    condB = RAW_LABEL.copy()                             # 0..19 (digit x language)
    return {"A_8cond": (condA, 8), "B_20cond": (condB, 20)}


# ------------------------------------------------------------------ balanced dichotomies over a condition set
def all_balanced_dichotomies(n):
    """All 4-vs-4 (or n/2-vs-n/2) balanced dichotomies, deduped by complement via 'must contain condition 0'."""
    half = n // 2
    out = []
    for plus in itertools.combinations(range(n), half):
        if 0 in plus:                                    # exactly one of {plus, complement} contains 0
            out.append(frozenset(plus))
    return out


def sampled_balanced_dichotomies(n, n_sample, rng):
    half = n // 2
    seen, out = set(), []
    guard = 0
    while len(out) < n_sample and guard < n_sample * 50:
        guard += 1
        plus = set(rng.choice(n, size=half, replace=False).tolist())
        if 0 not in plus:                                # canonicalise to the side containing condition 0
            plus = set(range(n)) - plus
        key = frozenset(plus)
        if key not in seen:
            seen.add(key); out.append(key)
    return out


# ------------------------------------------------------------------ shared balanced trial subsample (all nets)
def balanced_index(cond, n_cond, cap, rng):
    """One shared trial-index set: up to `cap` trials per condition, equal across conditions so chance = 0.5."""
    counts = [int((cond == c).sum()) for c in range(n_cond)]
    k = min(cap, min(counts))
    idx = []
    for c in range(n_cond):
        ci = np.where(cond == c)[0]
        rng.shuffle(ci)
        idx.append(ci[:k])
    return np.sort(np.concatenate(idx)), k


# ------------------------------------------------------------------ decoder (house convention)
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


def sd_for_net(X_sub, cond_sub, dichotomies):
    """Shattering dimensionality: mean cross-validated decodability over the dichotomy list, plus the vector."""
    def one(plus):
        lab = np.where(np.isin(cond_sub, list(plus)), 1, -1)
        return decodability(X_sub, lab)
    decs = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(one)(p) for p in dichotomies)
    decs = np.asarray(decs, float)
    return float(decs.mean()), decs


def load_rate_and_check(seed, arch):
    d = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")
    X = d["rate"].astype(float)
    y = d["y"].astype(int)
    assert X.shape[0] == RAW_LABEL.shape[0], f"{seed}/{arch}: cache n={X.shape[0]} vs SHD n={RAW_LABEL.shape[0]}"
    recon = 2 * (RAW_LABEL // 10) + (RAW_LABEL % 2)
    assert np.array_equal(recon, y), f"{seed}/{arch}: fine-label join misaligned with cached 4-class y"
    return X


# ================================================================= run
def main():
    sets = condition_sets()
    rng_t = np.random.default_rng(SEED_TRIALS)
    rng_d = np.random.default_rng(SEED_DICH)

    plans = {}
    for name, (cond, n_cond) in sets.items():
        idx, k = balanced_index(cond, n_cond, CAP_PER_COND, rng_t)
        if n_cond <= 8:
            dich = all_balanced_dichotomies(n_cond)
        else:
            dich = sampled_balanced_dichotomies(n_cond, N_SAMPLE_B, rng_d)
        plans[name] = dict(cond=cond, n_cond=n_cond, idx=idx, k=k, dich=dich)
        print(f"[{name}] {n_cond} conditions, {k} trials/cond ({idx.size} trials), {len(dich)} dichotomies",
              flush=True)

    per = {name: {"het": {}, "hom": {}} for name in plans}
    print(f"\n{'seed':>5} {'arch':>4} | " + " | ".join(f"SD_{n:<8}" for n in plans))
    for seed in SEEDS:
        for arch in ("het", "hom"):
            X = load_rate_and_check(seed, arch)
            row = []
            for name, pl in plans.items():
                Xs = X[pl["idx"]]
                cond_sub = pl["cond"][pl["idx"]]
                sd, decs = sd_for_net(Xs, cond_sub, pl["dich"])
                per[name][arch][seed] = {"SD": sd, "dec_min": float(decs.min()),
                                         "dec_max": float(decs.max()), "dec_sd": float(decs.std(ddof=1))}
                row.append(sd)
            print(f"{seed:>5} {arch:>4} | " + " | ".join(f"{v:>10.4f}" for v in row), flush=True)

    # ---- paired triples (het vs hom, n = 20) per condition set
    V = lambda name, arch: np.array([per[name][arch][s]["SD"] for s in SEEDS], float)
    triples = {}
    print("\n--- paired het vs hom shattering dimensionality (n = 20) ---")
    for name in plans:
        r = paired_triple(V(name, "het"), V(name, "hom"), f"SD {name}")
        triples[name] = r
        print(f"{name:>10}: het {r['het_mean']:.4f}  hom {r['hom_mean']:.4f}  d {r['delta']:+.4f}  "
              f"dz {r['dz']:+.2f}  t {S.fmt_p(r['t_p'])}  W {S.fmt_p(r['w_p'], r['w_floor'])}  "
              f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}", flush=True)

    # ================================================================= F4.7 two-bar headline
    fig, axes = plt.subplots(1, len(plans), figsize=(5.4 * len(plans), 4.8))
    if len(plans) == 1:
        axes = [axes]
    titles = {"A_8cond": "8 conditions (language x parity x magnitude)\n35 balanced dichotomies",
              "B_20cond": f"20 conditions (digit x language)\n{N_SAMPLE_B} sampled dichotomies"}
    for ax, name in zip(axes, plans):
        r = triples[name]
        means = [r["het_mean"], r["hom_mean"]]
        sds = [r["het_sd"], r["hom_sd"]]
        ax.bar([0, 1], means, 0.6, yerr=sds, color=[S.HET, S.HOM], capsize=4)
        ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="chance (balanced)")
        ax.set_xticks([0, 1]); ax.set_xticklabels([S.LABEL["het"], S.LABEL["hom"]])
        ax.set_ylabel("shattering dimensionality\n(mean CV decodability)")
        ax.set_title(titles.get(name, name))
        lo = min(means) - max(sds) - 0.02
        ax.set_ylim(max(0.45, lo), 1.0)
        ax.text(0.5, 0.97, f"$\\Delta$ = {r['delta']:+.4f}   $d_z$ = {r['dz']:+.2f}\n"
                           f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}   wins {r['wins']}/{r['n']}",
                ha="center", va="top", transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Shattering dimensionality, het vs hom (n = 20)", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "F4_11_shattering_dimension.png")

    # ================================================================= F4.7b per-seed paired slopes
    fig, axes = plt.subplots(1, len(plans), figsize=(4.6 * len(plans), 4.8))
    if len(plans) == 1:
        axes = [axes]
    for ax, name in zip(axes, plans):
        S.paired_slopes(ax, V(name, "het"), V(name, "hom"))
        ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.0)
        ax.set_ylabel("shattering dimensionality")
        ax.set_title(titles.get(name, name).split("\n")[0])
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Per-seed shattering dimensionality (paired)", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "F4_11b_shattering_paired.png")

    # ================================================================= summary json
    summary = {
        "analysis": "shattering_dimensionality_bernardi2020",
        "n_pairs": len(SEEDS),
        "decoder": "logistic (house decodability, 5-fold CV), max-margin SVM one line away",
        "cap_per_condition": CAP_PER_COND,
        "condition_sets": {
            name: {"n_conditions": pl["n_cond"], "trials_per_condition": pl["k"],
                   "n_trials": int(pl["idx"].size), "n_dichotomies": len(pl["dich"]),
                   "dichotomies_exhaustive": pl["n_cond"] <= 8}
            for name, pl in plans.items()},
        "by_arch": {name: {arch: {"SD_mean": float(V(name, arch).mean()),
                                  "SD_sd": float(V(name, arch).std(ddof=1))}
                           for arch in ("het", "hom")} for name in plans},
        "paired": triples,
        "per_network": {name: {arch: {int(s): per[name][arch][s] for s in SEEDS}
                               for arch in ("het", "hom")} for name in plans},
    }
    (OUT).mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "shattering_n20_summary.json", "w"), indent=1)
    print(f"\nwrote {OUT/'shattering_n20_summary.json'} and F4_11_shattering_dimension.png, "
          f"F4_11b_shattering_paired.png")


if __name__ == "__main__":
    main()
