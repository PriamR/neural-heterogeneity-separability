"""Emit clean LaTeX (booktabs) tables for the paper from the analysis-suite JSON, so the numbers in the tables
cannot drift from the figures. Writes thesis/chapters/tab_*.tex, which the chapters \\input.

The canonical thesis is Final Repo/thesis. Do not repoint this at any msc-template-v* directory: those are
retired snapshots and writing into them produces tables that disagree with the built paper.

Run: python code/analysis_suite/make_latex_tables.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "docs" / "analysis_suite"
M = json.load(open(OUT / "summary.json"))
T = json.load(open(OUT / "tables.json"))
sb = M["scoreboard"]


def pfmt(p, floor=None):
    if p != p:
        return "n/a"
    if floor is not None and p <= floor * 1.001:
        return r"$<10^{-5}$"
    if p < 1e-3:
        e = f"{p:.0e}".replace("e-0", "e-").replace("e-", r" \times 10^{-") + "}"
        base = e.split(r" \times")[0]
        return f"${base} \\times 10^{{-{int(-__import__('math').log10(p)):d}}}$" if p > 0 else "0"
    return f"{p:.3f}"


def sci(p):
    import math
    if p != p:
        return "n/a"
    if p >= 1e-3:
        return f"{p:.3f}"
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    return f"${mant:.1f}\\times10^{{{exp}}}$"


# ---------------------------------------------------------------- scoreboard
SCORE_ORDER = [
    ("Task accuracy", "Task accuracy", "{:.3f}"),
    ("Linear decoder accuracy", "Linear decodability", "{:.3f}"),
    ("Recurrent gain (spectral radius)", "Recurrent gain", "{:.2f}"),
    ("PCA n90 (1 ms)", "PCA $n_{90}$", "{:.1f}"),
    ("PCA participation ratio (1 ms)", "Participation ratio", "{:.2f}"),
    ("PCA top-5 variance", "Top-5 variance", "{:.3f}"),
    ("PC-reconstruction @ m=16", "PC-reconstruction ($m{=}16$)", "{:.3f}"),
    ("PC-reconstruction @ m=32", "PC-reconstruction ($m{=}32$)", "{:.3f}"),
    ("Active neurons", "Active units", "{:.1f}"),
    ("Mean firing rate", "Mean firing rate", "{:.3f}"),
]
rows = []
for key, lbl, fmt in SCORE_ORDER:
    r = sb[key]
    rows.append(f"{lbl} & {fmt.format(r['het_mean'])} & {fmt.format(r['hom_mean'])} & "
                f"{r['delta']:+.3f} & {r['dz']:+.2f} & {sci(r['t_p'])} & {sci(r['w_p'])} & "
                f"{sci(r['perm_p'])} & {r['wins']}/{r['n']} \\\\")

# shattering dimensionality (Bernardi) from its own summary, added to the het-leads block
SD = json.load(open(OUT / "shattering_n20_summary.json")) if (OUT / "shattering_n20_summary.json").exists() else None
sd_rows = []
if SD:
    for key, lbl in [("A_8cond", "Shattering dim.\\ (8 cond.)"), ("B_20cond", "Shattering dim.\\ (20 cond.)")]:
        r = SD["paired"][key]
        sd_rows.append(f"{lbl} & {r['het_mean']:.3f} & {r['hom_mean']:.3f} & "
                       f"{r['delta']:+.3f} & {r['dz']:+.2f} & {sci(r['t_p'])} & {sci(r['w_p'])} & "
                       f"{sci(r['perm_p'])} & {r['wins']}/{r['n']} \\\\")
scoreboard_tex = r"""\begin{table}[H]
  \centering
  \small
  \setlength{\tabcolsep}{4pt}
  \caption{The scoreboard. Every metric is a within-seed paired comparison across the twenty seeds, heterogeneous
  against homogeneous, with the mean of each, the paired difference, Cohen's $d_z$, all three paired tests, and
  the number of seeds on which the heterogeneous network is higher. The sign-flip permutation test is the primary
  and its floor at $n=20$ is $1.9\times10^{-6}$; the other two are reported alongside it, and they agree on every
  row. The block above the rule is where the heterogeneous network leads. Below it sit the two activity measures:
  recruitment, on which the two networks are level, and firing rate, which is higher in the heterogeneous network
  but does not track accuracy across the forty networks. The recurrent-gain row is the largest effect in the
  table and is taken up in Appendix~\ref{app:gain}.}
  \label{tab:scoreboard}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{lrrrrrrrr}
    \toprule
    Metric & het & hom & $\Delta$ & $d_z$ & paired $t$ $p$ & Wilcoxon $p$ & sign-flip $p$ & wins \\
    \midrule
""" + "\n    ".join(rows[:8] + sd_rows) + r"""
    \midrule
    """ + "\n    ".join(rows[8:]) + r"""
    \bottomrule
  \end{tabular}}
\end{table}
"""

# ------------------------------------------------- recruitment, single line, for S4.2
_ra = sb["Active neurons"]
recruit_tex = r"""\begin{table}[H]
  \centering
  \caption{Recruitment. The number of hidden units emitting at least one spike, as a within-seed paired
  comparison across the twenty seeds. The two networks are level, the difference runs against the heterogeneous
  network, and all three tests agree that it is not significant.}
  \label{tab:recruit}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{lrrrrrrrr}
    \toprule
    Quantity & het & hom & $\Delta$ & $d_z$ & paired $t$ $p$ & Wilcoxon $p$ & sign-flip $p$ & wins \\
    \midrule
    Active units & """ + (f"{_ra['het_mean']:.1f} & {_ra['hom_mean']:.1f} & {_ra['delta']:+.2f} & "
                          f"{_ra['dz']:+.2f} & {sci(_ra['t_p'])} & {sci(_ra['w_p'])} & "
                          f"{sci(_ra['perm_p'])} & {_ra['wins']}/{_ra['n']}") + r""" \\
    \bottomrule
  \end{tabular}}
\end{table}
"""

# ---------------------------------------------------------------- fixed-tau ladder (parse tables.json T4.8)
def parse_md(md):
    lines = [l for l in md.splitlines() if l.startswith("|")]
    body = lines[2:]
    return [[c.strip() for c in row.strip("|").split("|")] for row in body]


t48 = parse_md(T["T4.8"])
# Rungs are named by the nominal frozen value the ladder was designed around, matching the figures. The
# shortest rung is reported by the pipeline as its realised 3.26 ms, which is what the membrane-decay clip
# permits; that bound is stated once in the hyperparameter table rather than carried on every mention.
NOMINAL = {"3.26*": "2", "3.26": "2"}
rows = []
for r in t48:
    # cols: tau, accuracy(mean+/-sd), active, n90, PR, gain, delta, dz, t_p, w_p, perm_p, wins
    acc = r[1].replace("+/-", r"$\pm$")
    tau = NOMINAL.get(r[0], r[0])
    rows.append(f"{tau} & {acc} & {r[2]} & {r[3]} & {r[4]} & {r[6]} & {r[8]} & {r[9]} & {r[10]} & {r[11]} \\\\")
bowl_tex = r"""\begin{table}[H]
  \centering
  \small
  \setlength{\tabcolsep}{4pt}
  \caption{The fixed-tau ladder. Each rung freezes the hidden $\taumem$ at the stated value and retrains, ten
  seeds per rung. Accuracy, active-unit count, $n_{90}$ and participation ratio are means over the ten
  seeds; the remaining columns give the paired difference against the learned homogeneous baseline and the three
  paired tests at $n=10$ (sign-flip floor $2\times10^{-3}$). No rung reaches the learned homogeneous network, let
  alone the heterogeneous one, which reaches $0.698$ on these same ten seeds. Rungs are named by their nominal
  frozen value; the shortest is bounded below by the membrane-decay clip (Appendix~\ref{app:hyper}).}
  \label{tab:bowl}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{lrrrrrrrrr}
    \toprule
    Fixed $\taumem$ (ms) & Accuracy & Active & $n_{90}$ & PR & $\Delta$ vs hom & $t$ $p$ & Wilcoxon $p$ & sign-flip $p$ & wins \\
    \midrule
    """ + "\n    ".join(rows) + r"""
    \bottomrule
  \end{tabular}}
\end{table}
"""

# ---------------------------------------------------------------- O-information at k=8 (clean)
OI = json.load(open(OUT / "oinfo_summary_clean.json")) if (OUT / "oinfo_summary_clean.json").exists() else None
oinfo_tex = ""
if OI:
    fp8, fp10 = OI["frac_pos_k8"], OI.get("frac_pos_k10") or {}
    # k8_table / k10_table rows are [stat, het, hom, delta, dz, sign-flip p, wins/n], already formatted by
    # build_oinfo_figs.py through the same paired_triple every other comparison uses. Surfaced here so the
    # asserted null carries its paired test, rather than resting on the two arms looking similar.
    _k8 = {r[0]: r for r in OI["k8_table"]}
    _k10 = {r[0]: r for r in OI.get("k10_table", [])}

    def _pc(x):
        return f"{x:.0%}".replace("%", r"\%")

    def _orow(label, stat):
        r = _k8[stat]
        return f"    {label} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} \\\\"

    # k=10 is carried as a robustness check in the caption rather than as four more columns: it shows the same
    # null at a second subset size without widening the table past its paired stats.
    _k10note = ""
    if all(s in _k10 for s in ("max", "min", "spread")):
        _k10note = (f" At ten-neuron subsets the picture is unchanged: {_pc(fp10['het'])} against "
                    f"{_pc(fp10['hom'])} of subsets redundant, and none of the three comparisons significant "
                    f"(max $O$ $p={_k10['max'][5]}$, min $O$ $p={_k10['min'][5]}$, "
                    f"range $p={_k10['spread'][5]}$).")

    oinfo_tex = (
        "\\begin{table}[H]\n  \\centering\n  \\small\n  \\setlength{\\tabcolsep}{4pt}\n"
        "  \\caption{Higher-order information of eight-neuron subsets, the reliability ceiling of the "
        "estimator, twenty paired seeds. Both networks are redundancy-dominated: most subsets carry positive "
        f"O-information ({_pc(fp8['het'])} of heterogeneous and {_pc(fp8['hom'])} of homogeneous subsets at "
        "$k=8$), and neither the most redundant subset, the most synergistic subset nor the interaction range "
        "differs between the arms. Heterogeneity does not move the code toward synergy." + _k10note +
        " Paired within seed, led by the exact sign-flip permutation test.}\n"
        "  \\label{tab:oinfo}\n"
        "  \\resizebox{\\linewidth}{!}{%\n"
        "  \\begin{tabular}{lrrrrrr}\n    \\toprule\n"
        "    Quantity & het & hom & $\\Delta$ & $d_z$ & sign-flip $p$ & wins \\\\\n    \\midrule\n"
        + _orow(r"Most redundant subset (max $O$)", "max") + "\n"
        + _orow(r"Most synergistic subset (min $O$)", "min") + "\n"
        + _orow(r"Range (max $-$ min)", "spread") + "\n"
        "    \\bottomrule\n  \\end{tabular}}\n\\end{table}\n"
    )

# ---------------------------------------------------------------- structure of the code (capacity + time)
CAP = json.load(open(OUT / "capacity_corr_n20_summary.json")) if (OUT / "capacity_corr_n20_summary.json").exists() else None
TG = json.load(open(OUT / "temporal_generalisation_n20_summary.json")) if (OUT / "temporal_generalisation_n20_summary.json").exists() else None
SIG = json.load(open(OUT / "signal_dimension_n20_summary.json")) if (OUT / "signal_dimension_n20_summary.json").exists() else None
structure_tex = ""
if CAP and TG and SIG:
    rows = []

    def _row(r, lbl, fmt="{:.3f}"):
        return (f"{lbl} & {fmt.format(r['het_mean'])} & {fmt.format(r['hom_mean'])} & {r['delta']:+.3f} & "
                f"{r['dz']:+.2f} & {sci(r['t_p'])} & {sci(r['w_p'])} & {sci(r['perm_p'])} & "
                f"{r['wins']}/{r['n']} \\\\")

    c = CAP["paired"]["B_20cond"]
    rows.append(_row(c["alpha_M"], r"Manifold capacity $\alpha_M$"))
    rows.append(_row(c["R_M"], r"Manifold radius $R_M$"))
    rows.append(_row(c["D_M"], r"Manifold dimension $D_M$", "{:.2f}"))
    rows.append(_row(c["center_corr"], "Centroid correlation"))
    # signal dimensionality (cvPCA) removed from the report on 7 Sep 2026: it averages time away, which the
    # temporal-basis account is about; the JSON summary is kept for the record
    t = TG["paired"]
    rows.append(_row(t["diag_mean"], "Instantaneous decoding"))
    rows.append(_row(t["offdiag_mean"], "Cross-time decoding"))
    rows.append(_row(t["long_lag_ratio"], "Long-lag transfer / diagonal"))
    rows.append(_row(t["gen_width_bins"], "Generalisation width (bins)", "{:.2f}"))
    structure_tex = r"""\begin{table}[H]
  \centering
  \small
  \setlength{\tabcolsep}{4pt}
  \caption{The structure of the code. Manifold quantities come from the replica mean-field
  analysis over the twenty digit-by-language class manifolds; capacity rises when radius, manifold dimension and
  centroid correlation fall, so the heterogeneous network moving the favourable way on all four rows, higher on
  capacity and lower on the other three, is one coherent result rather
  than four. Signal dimensionality is the participation ratio of the cross-validated spectrum, which counts only
  directions carrying reliable stimulus-locked signal. The temporal rows train a readout at one time bin and test
  it at every other. The final row is a null and is reported as one: the width of the generalisation window is the
  same in both networks, and the heterogeneous advantage sits at long lags.}
  \label{tab:structure}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{lrrrrrrrr}
    \toprule
    Quantity & het & hom & $\Delta$ & $d_z$ & paired $t$ $p$ & Wilcoxon $p$ & sign-flip $p$ & wins \\
    \midrule
    """ + "\n    ".join(rows) + r"""
    \bottomrule
  \end{tabular}}
\end{table}
"""

# One file per table, so each can be \input at its own point of discussion rather than all four entering the
# chapter together and being scattered by float placement.
HDR = ("% Auto-generated by code/analysis_suite/make_latex_tables.py from docs/analysis_suite/*.json.\n"
       "% Do not edit by hand; regenerate to refresh the numbers.\n\n")
# The canonical thesis is Final Repo/thesis. Resolved relative to this file rather than hard-coded to a
# version name, so the script cannot silently write into a retired copy. It fails loudly if the directory is
# missing, which is the failure mode that produced stale tables twice before.
chdir = REPO / "thesis" / "chapters"
if not chdir.is_dir():
    raise SystemExit(
        f"thesis chapters directory not found: {chdir}\n"
        "This script writes the generated tables straight into the canonical thesis. If the thesis has moved, "
        "update this path rather than pointing it at an older copy."
    )
print(f"writing generated tables into {chdir}")
for name, tex in [("tab_recruitment", recruit_tex), ("tab_oinfo", oinfo_tex),
                  ("tab_scoreboard", scoreboard_tex), ("tab_structure", structure_tex),
                  ("tab_bowl", bowl_tex)]:
    if not tex:
        print(f"skipped {name} (no data)")
        continue
    p = chdir / f"{name}.tex"
    p.write_text(HDR + tex, encoding="utf-8")
    print(f"wrote {p.name} ({len(tex)} chars)")

# The combined file is retired; remove it so a stale copy cannot be \input by mistake.
legacy = chdir / "tables_generated.tex"
if legacy.exists():
    legacy.unlink()
    print("removed legacy tables_generated.tex")
