"""Redraw the shattering-dimensionality figures from shattering_n20_summary.json (no recompute).

F4_11: one panel, two groups (the eight-condition and the twenty-condition set), heterogeneous against
homogeneous as thin bars with 95% confidence whiskers and per-seed dots, the paired t p and win count printed
under each group. The axis starts at chance (0.5) so no bar is truncated.

Run: python code/analysis_suite/make_shattering_fig_from_json.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
S.apply()

summ = json.load(open(OUT / "shattering_n20_summary.json"))
sets = [("A_8cond", "Eight conditions\n(language $\\times$ parity $\\times$ magnitude)\n35 dichotomies"),
        ("B_20cond", "Twenty conditions\n(digit $\\times$ language)\n"
                     f"{summ['condition_sets']['B_20cond']['n_dichotomies']} sampled dichotomies")]
seeds = sorted(int(s) for s in summ["per_network"][sets[0][0]]["het"])


def sd_vec(name, arch):
    return np.array([summ["per_network"][name][arch][str(s)]["SD"] for s in seeds], float)


# ---- grouped bars, one figure
fig, ax = plt.subplots(figsize=(7.0, 4.8))
V = {a: np.array([sd_vec(name, a) for name, _ in sets]) for a in ("het", "hom")}
# the statistics ride on the tick labels, under the group they belong to, so nothing sits on the title
ticklabels = []
for name, lbl in sets:
    r = summ["paired"][name]
    ticklabels.append(f"{lbl}\nPaired $t$ $p$ = {S.fmt_p(r['t_p'])}, heterogeneous higher on {r['wins']}/{r['n']}")
S.paired_bars(ax, ticklabels, V, None, width=0.30)
ax.tick_params(axis="x", labelsize=8)
S.chance_line(ax, 0.5, label="Chance")
ax.set_ylim(0.5, 0.95)
ax.set_ylabel("Shattering dimensionality\n(mean cross-validated accuracy over dichotomies)")
S.panel_title(ax, None, "Shattering dimensionality on two condition sets")
S.legend(ax, loc="upper left", ncol=3)
S.save(fig, FIGS / "F4_11_shattering_dimension.png")

# ---- per-seed paired slopes (diagnostic, not in the paper)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
for ax, (name, lbl) in zip(axes, sets):
    S.paired_slopes(ax, sd_vec(name, "het"), sd_vec(name, "hom"))
    ax.set_ylabel("Shattering dimensionality")
    S.panel_title(ax, "ab"[sets.index((name, lbl))], lbl.split("\n")[0])
S.save(fig, FIGS / "F4_11b_shattering_paired.png")
print("redrew F4_11_shattering_dimension.png, F4_11b_shattering_paired.png")
