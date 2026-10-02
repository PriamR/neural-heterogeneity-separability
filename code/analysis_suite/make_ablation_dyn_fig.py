"""Build the dynamical-lesion figures from ablation_dynamical.jsonl (produced by run_ablation_dynamical.py):
  G8b  het long-first / short-first vs hom random, absolute accuracy and as a fraction of the intact network
  G8c  the same with the fixed-tau rungs overlaid
Writes ablation_dyn.json (curve means/CIs plus the paired per-level tests) for the presentation tables.

het starts ten accuracy points above hom, so the absolute curves cannot be compared at face value: a network
with more to lose falls further. The second panel divides every curve by its own intact accuracy, which is the
comparison the robustness claim actually needs.

Run after the dynamical sweep completes: python code/analysis_suite/make_ablation_dyn_fig.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs" / "presentation"
JL = OUT / "ablation_dynamical.jsonl"
S.apply()

recs = [json.loads(l) for l in JL.read_text().splitlines() if l.strip()]
LEVELS = recs[0]["levels"]
by, seeds_by = {}, {}
for r in recs:
    fam, sd = r["key"].rsplit("_", 1)            # e.g. het_long, hom_rand, fixed15_rand
    by.setdefault(fam, []).append(r["acc"])
    seeds_by.setdefault(fam, []).append(int(sd))
# sort every family by seed so the rows of different families line up pairwise
for fam in by:
    o = np.argsort(seeds_by[fam])
    by[fam] = [by[fam][i] for i in o]
    seeds_by[fam] = [seeds_by[fam][i] for i in o]
print("families:", {k: len(v) for k, v in by.items()})


def stack(fam):
    return np.array(by[fam])


def frac(fam):
    """Each seed's curve as a fraction of its own intact (0 lesioned) accuracy."""
    A = stack(fam)
    return A / A[:, [0]]


def common(f1, f2):
    """Row indices of f1 and f2 restricted to the seeds they share, in the same order."""
    sh = sorted(set(seeds_by[f1]) & set(seeds_by[f2]))
    return ([seeds_by[f1].index(s) for s in sh], [seeds_by[f2].index(s) for s in sh], sh)


def band(ax, fam, col, label, ls="-", marker="o", norm=False):
    A = frac(fam) if norm else stack(fam)
    m = A.mean(0); e = st.t.ppf(0.975, A.shape[0] - 1) * A.std(0, ddof=1) / np.sqrt(A.shape[0])
    ax.plot(LEVELS, m, color=col, lw=2, ls=ls, marker=marker, ms=3, label=label)
    ax.fill_between(LEVELS, m - e, m + e, color=col, alpha=0.15, lw=0)
    return m


# G8b het vs hom, absolute and normalised to each network's own intact accuracy
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, norm in [(axes[0], False), (axes[1], True)]:
    band(ax, "het_long", S.HET, "het, longest-tau first", ls="-", norm=norm)
    band(ax, "het_short", "#e08a97", "het, shortest-tau first", ls="--", norm=norm)
    band(ax, "hom_rand", S.HOM, "hom, random order", ls="-", norm=norm)
    ax.set_xlabel("hidden neurons lesioned")
S.chance_line(axes[0])
axes[0].set_ylabel("task accuracy through the real readout")
axes[0].set_title("Absolute accuracy")
axes[0].legend(loc="upper right", fontsize=9)
axes[1].axhline(1.0, color=S.CHANCE, ls=":", lw=1.2)
axes[1].set_ylabel("accuracy as a fraction of the intact network")
axes[1].set_title("Normalised to each network's intact accuracy")
axes[1].legend(loc="lower left", fontsize=9)
fig.suptitle("Dynamical lesion: het long-first vs short-first, hom random", fontweight="bold", y=1.01)
S.save(fig, FIGS / "G8b_ablation_dyn_hom_het.png")

# G8c + fixed rungs
fig, ax = plt.subplots(figsize=(8.5, 5))
band(ax, "het_long", S.HET, "het, longest-tau first", ls="-")
band(ax, "het_short", "#e08a97", "het, shortest-tau first", ls="--")
band(ax, "hom_rand", S.HOM, "hom, random", ls="-")
RUNG_COL = {"fixed2_rand": S.FIXED_RAMP[2.0], "fixed3.5_rand": S.FIXED_RAMP[3.5],
            "fixed15_rand": S.FIXED_RAMP[15.0], "fixed35_rand": S.FIXED_RAMP[35.0]}
RUNG_LBL = {"fixed2_rand": "fixed 2 ms*", "fixed3.5_rand": "fixed 3.5 ms",
            "fixed15_rand": "fixed 15 ms", "fixed35_rand": "fixed 35 ms"}
for fam in ["fixed2_rand", "fixed3.5_rand", "fixed15_rand", "fixed35_rand"]:
    if fam in by:
        band(ax, fam, RUNG_COL[fam], RUNG_LBL[fam], ls="-", marker="s")
S.chance_line(ax)
ax.set_xlabel("hidden neurons lesioned")
ax.set_ylabel("task accuracy through the real readout")
ax.set_title("Dynamical lesion with the fixed-tau ladder (random order)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
S.save(fig, FIGS / "G8c_ablation_dyn_all.png")

# --- paired tests at each lesion level -------------------------------------------------------------
# Two questions, each paired within seed. (1) Does het hold up better than hom once the head start is
# removed? That is the normalised comparison. (2) Does the order matter, i.e. do the long-tau and the
# short-tau units contribute equally? That is het long-first against het short-first on the same seeds.
def level_rows(f1, f2, norm, label):
    i1, i2, sh = common(f1, f2)
    A = (frac(f1) if norm else stack(f1))[i1]
    B = (frac(f2) if norm else stack(f2))[i2]
    rows = []
    for j, lv in enumerate(LEVELS):
        if lv == 0 and norm:                   # every normalised curve starts at exactly 1.0
            continue
        r = paired_triple(A[:, j], B[:, j], f"{label} @ {lv}")
        rows.append({"level": lv, "comparison": f"{label}, {lv} lesioned", "a_mean": r["het_mean"],
                     "b_mean": r["hom_mean"], "delta": r["delta"], "dz": r["dz"], "t_p": r["t_p"],
                     "w_p": r["w_p"], "perm_p": r["perm_p"], "perm_floor": r["perm_floor"],
                     "w_floor": r["w_floor"], "wins": r["wins"], "n": r["n"]})
    return rows


summary = {"levels": LEVELS, "n_per_family": {k: len(v) for k, v in by.items()},
           "families": {k: {"mean": stack(k).mean(0).tolist(), "sd": stack(k).std(0, ddof=1).tolist(),
                            "frac_mean": frac(k).mean(0).tolist(),
                            "frac_sd": frac(k).std(0, ddof=1).tolist(),
                            "seeds": seeds_by[k]} for k in by},
           "tests": {"abs_het_long_vs_hom": level_rows("het_long", "hom_rand", False, "het long-first vs hom"),
                     "frac_het_long_vs_hom": level_rows("het_long", "hom_rand", True, "het long-first vs hom (fraction retained)"),
                     "frac_het_short_vs_hom": level_rows("het_short", "hom_rand", True, "het short-first vs hom (fraction retained)"),
                     "frac_het_long_vs_short": level_rows("het_long", "het_short", True, "het long-first vs short-first (fraction retained)")}}
json.dump(summary, open(OUT / "ablation_dyn.json", "w"), indent=1)
print("wrote G8b_ablation_dyn_hom_het.png, G8c_ablation_dyn_all.png, ablation_dyn.json")
