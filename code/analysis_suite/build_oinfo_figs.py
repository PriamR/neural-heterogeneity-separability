"""Min/max O-information analysis at n=20, in the analysis-suite house style.

Combines the legacy per-subset records (10 original seeds) with the new n=20 records (10 new seeds, plus the
9 no-WD hom retrains when present). Produces, for each subset size k, the most redundant (max O) and most
synergistic (min O) subset per network, and tests het vs hom on the paired seeds.

Two hom sets are selectable:
  --hom asis    the homogeneous nets as trained (original batch carries weight decay). Uses legacy hom records
                for 202..930 and new records for 1001..1010. Gives min/max at n=20 as the suite stands.
  --hom clean   replaces the 9 confounded homs (202..930) with their no-WD retrain records (arch 'hom',
                source 'hom_nowd' in the n20 file). Requires the retrains to have been swept. The paper number.

het is always the 20 het nets (legacy 202..930 + 101, new 1001..1010).

Run: python code/analysis_suite/build_oinfo_figs.py --hom asis
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt
from stats_core import paired_triple

REPO = HERE.parent.parent
RROOT = REPO.parent
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
LEGACY = RROOT / "Project Files" / "O-S Information" / "outputs" / "os_info_records.jsonl"
N20 = REPO / "data_cache" / "analysis_suite" / "os_info_records_n20.jsonl"
S.apply()

SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_NEW = list(range(1001, 1011))
CONFOUNDED = [202, 210, 340, 440, 550, 660, 710, 820, 930]
RK = [2, 4, 6, 8, 10, 12]


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def build_records(hom_mode):
    """Return ok records with (seed, arch, k, O_info), assembling het (all 20) and hom per the mode."""
    legacy = [r for r in load(LEGACY) if r.get("status") == "ok"]
    new = [r for r in load(N20) if r.get("status") == "ok"]
    recs = []
    # het: legacy for the original 10, new for the new 10
    recs += [r for r in legacy if r["arch"] == "het"]
    recs += [r for r in new if r["arch"] == "het" and r["seed"] in SEEDS_NEW]
    # hom
    recs += [r for r in new if r["arch"] == "hom" and r["seed"] in SEEDS_NEW and r.get("source") == "hom"]
    recs += [r for r in legacy if r["arch"] == "hom" and r["seed"] == 101]
    if hom_mode == "asis":
        recs += [r for r in legacy if r["arch"] == "hom" and r["seed"] in CONFOUNDED]
    else:  # clean: use the no-WD retrains
        recs += [r for r in new if r["arch"] == "hom" and r.get("source") == "hom_nowd"]
    return recs


def minmax(recs):
    """Per (arch, seed, k): min and max O over that net's subsets.

    Pairing is resolved *per k*, not once globally. Sweep coverage is not uniform across k (the no-WD hom
    retrains were swept to k=8 first), and taking whichever seeds happen to have data at each k would plot a
    20-network het mean against a 12-network hom mean on the same axes while the paired test silently skipped
    that k. Every k here uses the seeds that have both arms, and the count is carried through so the figure and
    the tables can state it.
    """
    seeds = {"het": set(), "hom": set()}
    store = {}
    for r in recs:
        key = (r["arch"], r["seed"], r["k"])
        store.setdefault(key, []).append(r["O_info"])
        seeds[r["arch"]].add(r["seed"])
    paired = sorted(seeds["het"] & seeds["hom"])
    agg = {a: {"min": {}, "max": {}, "spread": {}} for a in ("het", "hom")}
    seeds_by_k = {}
    for k in RK:
        pk = [s for s in paired if store.get(("het", s, k)) and store.get(("hom", s, k))]
        seeds_by_k[k] = pk
        for a in ("het", "hom"):
            mn = np.array([min(store[(a, s, k)]) for s in pk])
            mx = np.array([max(store[(a, s, k)]) for s in pk])
            agg[a]["min"][k] = mn; agg[a]["max"][k] = mx; agg[a]["spread"][k] = mx - mn
    return agg, paired, seeds_by_k


def main(hom_mode):
    recs = build_records(hom_mode)
    agg, paired, seeds_by_k = minmax(recs)
    n_by_k = {k: len(seeds_by_k[k]) for k in RK}
    n = len(paired)
    tag = "clean" if hom_mode == "clean" else "asis"
    print(f"hom mode: {hom_mode} | paired seeds: n={n} {paired}")
    print("paired seeds per k: " + ", ".join(f"k={k}: {n_by_k[k]}" for k in RK))

    # ---- figure: min/max O vs k with bands
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for a in ("het", "hom"):
        mx = np.array([agg[a]["max"][k].mean() for k in RK])
        mn = np.array([agg[a]["min"][k].mean() for k in RK])
        hx = np.array([S.ci95_mean(agg[a]["max"][k]) for k in RK])
        hn = np.array([S.ci95_mean(agg[a]["min"][k]) for k in RK])
        S.band(ax, RK, mx, hx, S.COL[a], marker="^", ms=5,
               label=f"{S.LABEL[a]}, most redundant subset (max $O$)")
        S.band(ax, RK, mn, hn, S.COL[a], marker="v", ms=5, ls=(0, (4, 2)),
               label=f"{S.LABEL[a]}, most synergistic subset (min $O$)")
    ax.axhline(0, color=S.CHANCE, lw=1.0)
    ax.set_xlabel("Subset size, $k$")
    ax.set_ylabel("O-information (bits; positive redundant, negative synergistic)")
    _uniform = len(set(n_by_k[k] for k in RK)) == 1
    S.panel_title(ax, None, "Most redundant and most synergistic subsets against subset size")
    if not _uniform:
        for k in RK:
            ax.annotate(f"n = {n_by_k[k]}", xy=(k, ax.get_ylim()[0]), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color=S.MUTED)
    ax.set_xticks(RK)
    S.legend(ax, loc="upper left", fontsize=8)
    S.save(fig, FIGS / f"F4_5_minmax_oinfo_{tag}.png")

    # ---- figure: the range (max - min) against k (not in the paper)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for a in ("het", "hom"):
        sp = np.array([agg[a]["spread"][k].mean() for k in RK])
        h = np.array([S.ci95_mean(agg[a]["spread"][k]) for k in RK])
        S.band(ax, RK, sp, h, S.COL[a], label=S.LABEL[a], marker="o")
    ax.set_xlabel("Subset size, $k$")
    ax.set_ylabel("O-information range (max $-$ min, bits)")
    S.panel_title(ax, None, "Range of the O-information against subset size")
    ax.set_xticks(RK)
    S.legend(ax)
    S.save(fig, FIGS / f"F4_5b_oinfo_range_{tag}.png")

    # ---- stats at k=8
    rows = []
    print(f"\n--- het vs hom at each k (paired, n={n}) ---")
    print(f"{'k':>3} {'stat':>7} | {'het':>8} {'hom':>8} {'delta':>8} {'dz':>6} {'perm p':>10} {'wins/n':>7}")
    tables = {8: [], 10: []}
    for k in RK:
        for st in ("min", "max", "spread"):
            h, o = agg["het"][st][k], agg["hom"][st][k]
            if h.size != o.size or h.size == 0:
                continue
            r = paired_triple(h, o, f"O {st} k={k}", higher_is_better=True)
            print(f"{k:>3} {st:>7} | {r['het_mean']:>+8.3f} {r['hom_mean']:>+8.3f} {r['delta']:>+8.3f} "
                  f"{r['dz']:>+6.2f} {S.fmt_p(r['perm_p'], r['perm_floor']):>10} {r['wins']}/{r['n']}")
            if k in tables:
                tables[k].append([st, f"{r['het_mean']:+.3f}", f"{r['hom_mean']:+.3f}", f"{r['delta']:+.3f}",
                                  f"{r['dz']:+.2f}", S.fmt_p(r['perm_p'], r['perm_floor']), f"{r['wins']}/{r['n']}"])

    # ---- fraction O>0 (redundancy dominance), restricted to the seeds paired at that k
    def frac_pos(a, k):
        pk = set(seeds_by_k.get(k, []))
        vals = [r["O_info"] for r in recs if r["arch"] == a and r["k"] == k and r["seed"] in pk]
        return float(np.mean(np.array(vals) > 0)) if vals else float("nan")

    fp = {k: {a: frac_pos(a, k) for a in ("het", "hom")} for k in (8, 10) if k in RK}
    for k in fp:
        print(f"fraction of k={k} subsets with O>0 (redundant): "
              f"het {fp[k]['het']:.1%}, hom {fp[k]['hom']:.1%}")

    summary = {
        "hom_mode": hom_mode, "n_paired": n, "paired_seeds": paired,
        "n_by_k": {str(k): n_by_k[k] for k in RK},
        "frac_pos_k8": fp.get(8, {}), "frac_pos_k10": fp.get(10, {}),
        "k8_table": tables[8], "k10_table": tables[10],
        "k8": {st: {"het": float(agg["het"][st][8].mean()), "hom": float(agg["hom"][st][8].mean())}
               for st in ("min", "max", "spread")},
        "k10": ({st: {"het": float(agg["het"][st][10].mean()), "hom": float(agg["hom"][st][10].mean())}
                 for st in ("min", "max", "spread")} if 10 in RK and agg["het"]["min"][10].size else {}),
    }
    json.dump(summary, open(OUT / f"oinfo_summary_{tag}.json", "w"), indent=1)
    print(f"\nwrote {OUT / f'oinfo_summary_{tag}.json'} and figures F4_5_minmax_oinfo_{tag}.png, F4_5b_oinfo_range_{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hom", choices=["asis", "clean"], default="asis")
    ap.add_argument("--kmax", type=int, default=12, help="cap subset sizes shown (k>8 is bias-dominated)")
    args = ap.parse_args()
    RK = [k for k in RK if k <= args.kmax]
    main(args.hom)
