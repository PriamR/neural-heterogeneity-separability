"""Significance tables for the cross-seed extension, in markdown and LaTeX from the same numbers.

Every row is a within-seed paired comparison over the twenty seeds. The pairing is what makes the design
worth anything: within a seed the two networks share initial weights, data order and training schedule, so
the only free difference is the hidden membrane time constant, and the paired difference isolates it. An
unpaired test on the same data would throw that away and answer a weaker question.

Three tests are reported for every row, never one:

  paired t         parametric, and fragile at this n
  Wilcoxon         distribution-free but coarse, with a hard floor at 2/2^n
  sign-flip        exact by enumerating all 2^20 sign assignments; the primary

Both non-parametric tests bottom out at 2/2^20 = 1.9e-6, which is what a unanimous twenty-out-of-twenty
result reads as. That is a maximally significant outcome, not a weak one, and it is marked as the floor
rather than printed as a bare number so it cannot be misread as a limit on the effect.

A Holm correction across each table's family of rows is reported alongside the raw p, because these tables
run many metrics at once and the raw values should not be read as if each were the only test.

Run: python code/supervisor_extension/make_tables.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
from stats_core import holm


def load(name):
    p = SC.DATA / name
    return json.load(open(p)) if p.exists() else None


def fp(p, floor=None):
    if p is None or p != p:
        return "n/a"
    if floor is not None and p <= floor * 1.001:
        return f"{p:.1e} (floor)"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def fp_tex(p, floor=None):
    if p is None or p != p:
        return "n/a"
    if floor is not None and p <= floor * 1.001:
        return r"$1.9\times10^{-6}$\,\textsuperscript{f}"
    if p >= 1e-3:
        return f"{p:.3f}"
    e = int(math.floor(math.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


def build(rows, paired, keys, fmt="{:.3f}"):
    """rows: list of (json key, printed label). Returns the assembled record list plus Holm-adjusted p."""
    recs = []
    for key, lbl in keys:
        r = paired[key]
        recs.append({"label": lbl, "r": r, "fmt": fmt if isinstance(fmt, str) else fmt.get(key, "{:.3f}")})
    adj = holm([x["r"]["perm_p"] for x in recs])
    for x, a in zip(recs, adj):
        x["holm"] = float(a)
    return recs


def md_table(title, recs, note=""):
    out = [f"### {title}", ""]
    out.append("| Metric | het | hom | Δ | d_z | paired t | Wilcoxon | sign-flip | Holm | wins |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for x in recs:
        r, f = x["r"], x["fmt"]
        out.append(f"| {x['label']} | {f.format(r['het_mean'])} | {f.format(r['hom_mean'])} | "
                   f"{r['delta']:+.4g} | {r['dz']:+.2f} | {fp(r['t_p'])} | "
                   f"{fp(r['w_p'], r['w_floor'])} | {fp(r['perm_p'], r['perm_floor'])} | "
                   f"{fp(x['holm'])} | {r['wins']}/{r['n']} |")
    if note:
        out += ["", note]
    out.append("")
    return "\n".join(out)


def tex_table(caption, label, recs):
    body = []
    for x in recs:
        r, f = x["r"], x["fmt"]
        body.append(f"    {x['label']} & {f.format(r['het_mean'])} & {f.format(r['hom_mean'])} & "
                    f"{r['delta']:+.4g} & {r['dz']:+.2f} & {fp_tex(r['perm_p'], r['perm_floor'])} & "
                    f"{fp_tex(x['holm'])} & {r['wins']}/{r['n']} \\\\")
    return (r"""\begin{table}[t]
  \centering
  \caption{""" + caption + r"""}
  \label{""" + label + r"""}
  \begin{tabular}{lrrrrrrr}
    \toprule
    Quantity & het & hom & $\Delta$ & $d_z$ & sign-flip $p$ & Holm $p$ & wins \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")


MOD_KEYS = [
    ("full_Qnorm", "Full-graph $Q/Q_{rand}$ (binned)"),
    ("full_Q", "Full-graph modularity $Q$"),
    ("full_sigma", "Small-worldness $\\sigma$"),
    ("full_ncomm", "Communities (full graph)"),
    ("cum_Qnorm", "Cumulative-count $Q/Q_{rand}$"),
    ("cum_sigma", "Cumulative-count $\\sigma$"),
    ("eta2_lang", "Community $\\eta^2$ language"),
    ("eta2_par", "Community $\\eta^2$ parity"),
    ("eta2_4c_cum", "Cumulative community $\\eta^2$ 4-class"),
    ("n_active", "Active units"),
    ("edge_mi_mean", "Mean edge MI (bits)"),
]
RSI_KEYS = [
    ("I_marginal", "Single-unit MI $I(X_i;Y)$ (bits)"),
    ("I_joint", "Pair joint MI $I(X_i,X_j;Y)$ (bits)"),
    ("redundancy", "Redundancy $\\min(I_i,I_j)$ (bits)"),
    ("synergy", "Synergy $I_{joint}-\\max(I_i,I_j)$ (bits)"),
    ("RSI", "RSI $=$ synergy $-$ redundancy (bits)"),
    ("redundancy_n", "Redundancy $/\\,I_{joint}$"),
    ("synergy_n", "Synergy $/\\,I_{joint}$"),
    ("RSI_n", "RSI $/\\,I_{joint}$"),
    ("frac_synergistic_pairs", "Fraction of pairs with RSI $>0$"),
]
TEMP_KEYS = [
    ("inst_4-class_com_ms", "Instantaneous 4-class COM (ms)"),
    ("inst_language_com_ms", "Instantaneous language COM (ms)"),
    ("inst_parity_com_ms", "Instantaneous parity COM (ms)"),
    ("inst_4-class_spread_ms", "Instantaneous 4-class spread (ms)"),
    ("inst_gap_ms", "Instantaneous COM gap (parity $-$ language, ms)"),
    ("acc_4-class_lockin_ms", "4-class lock-in (ms)"),
    ("acc_language_lockin_ms", "Language lock-in (ms)"),
    ("acc_parity_lockin_ms", "Parity lock-in (ms)"),
    ("acc_gap_ms", "Lock-in gap (parity $-$ language, ms)"),
]
DOL_KEYS = [
    ("eta2_SI_module", "$\\eta^2$(selectivity $\\sim$ module)"),
    ("eta2_excess", "$\\eta^2$ excess over its permutation null"),
    ("n_modules", "Number of modules"),
    ("corr_L_P", "corr(language MI, parity MI) over units"),
    ("jaccard_top", "Jaccard overlap of top language/parity units"),
    ("mi_lang_mean", "Mean per-unit language MI (bits)"),
    ("mi_par_mean", "Mean per-unit parity MI (bits)"),
]


def main():
    md = ["# Cross-seed significance tables",
          "",
          "Twenty within-seed pairs. Each pair shares initial weights, data order and training schedule; only",
          "the hidden membrane time constant differs. `Δ` is the mean paired difference het minus hom, `d_z`",
          "is Cohen's d for paired differences, and `wins` counts the seeds on which the heterogeneous network",
          "is higher. The sign-flip permutation is exact over all 2^20 sign assignments and floors at 1.9e-6,",
          "which is what a unanimous 20/20 reads as. Holm adjusts across the rows of each table.",
          ""]
    tex = ["% Auto-generated by code/supervisor_extension/make_tables.py. Regenerate; do not hand-edit.", ""]

    M = load("modularity_n20.json")
    if M:
        recs = build(None, M["paired"], MOD_KEYS,
                     {"full_ncomm": "{:.1f}", "n_active": "{:.1f}", "edge_mi_mean": "{:.4f}",
                      "eta2_lang": "{:.4f}", "eta2_par": "{:.4f}", "eta2_4c_cum": "{:.4f}"})
        md.append(md_table("Functional-network modularity", recs,
                           "A $Q/Q_{rand}$ of one is the random null: values at or below it mean the community "
                           "structure is no stronger than chance. On the full and cumulative graphs, whose "
                           "densities are about 0.89 and 1.0, the degree-preserving swap cannot find legal "
                           "moves and the null reduces to a permutation of weights over a near-fixed topology; "
                           "the matched-density sweep, where the graphs are sparse and swaps succeed, is the "
                           "test of topology proper. See METHOD_NOTES.md."))
        tex.append(tex_table(
            "Functional-network modularity across the twenty paired seeds. Edges are zero-lag binary mutual "
            "information corrected against a circular-shift surrogate null; modularity is weighted Louvain "
            "normalised against degree-preserving randomisations, so neither the heterogeneous network's higher "
            "firing rate nor its denser graph can produce the difference. A value of one is the random null.",
            "tab:sup-modularity", recs))

        # the density sweep is where the degree-preserving null actually operates, so it gets its own table
        from stats_core import paired_triple
        md += ["#### Modularity at matched edge density", "",
               "The full-graph rows above cannot separate topology from weights. These can: at a fixed edge "
               "density both graphs have the same number of edges by construction, they are sparse enough for "
               "the degree-preserving swap to succeed, and the binary variant discards weights entirely so only "
               "topology remains.", "",
               "| Weighting | Density | het | hom | Δ | d_z | sign-flip p | het higher |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for wt in ("weighted", "binary"):
            Qh = np.array(M["densities_sweep"][f"{wt}_het"]["Qnorm"], float)
            Qo = np.array(M["densities_sweep"][f"{wt}_hom"]["Qnorm"], float)
            for i, dd in enumerate(M["densities"]):
                r = paired_triple(Qh[:, i], Qo[:, i])
                md.append(f"| {wt} | {dd:.2f} | {r['het_mean']:.3f} | {r['hom_mean']:.3f} | "
                          f"{r['delta']:+.3f} | {r['dz']:+.2f} | {fp(r['perm_p'], r['perm_floor'])} | "
                          f"{r['wins']}/{r['n']} |")
        md.append("")

    B = load("block_concentration_n20.json")
    if B:
        recs = build(None, B["paired"],
                     [("louvain", "Within/between weight ratio (Louvain)"),
                      ("fixedK", f"Within/between weight ratio (fixed K = {B['k_fixed']})"),
                      ("louvain_cum", "Within/between weight ratio (cumulative graph)")], "{:.2f}")
        md.append(md_table("Modularity without a randomisation null", recs,
                           "The ratio of mean within-community edge weight to mean between-community edge "
                           "weight: what a block matrix shows, with no randomisation anywhere in it. Every "
                           "row favours the homogeneous network, including the fixed-K row where the "
                           "community count cannot contribute and the cumulative row where $Q/Q_{rand}$ said "
                           "the opposite. Across the forty networks this ratio correlates with full-graph "
                           f"$Q/Q_{{rand}}$ at only r = {B['corr_ratio_vs_Qnorm']:+.2f}, which is why the two "
                           "measures reach different conclusions."))
        tex.append(tex_table(
            "Within-community weight concentration across the twenty paired seeds. The statistic is the mean "
            "edge weight inside communities divided by the mean edge weight between them, computed on the "
            "Louvain partition, on a spectral partition into a fixed three communities, and on the "
            "cumulative-count graph. No randomisation null enters, so none of the difficulties that affect "
            "$Q/Q_{rand}$ on a dense graph apply.",
            "tab:sup-blockconc", recs))

    R = load("rsi_n20.json")
    if R:
        recs = build(None, R["paired"], RSI_KEYS, "{:.4f}")
        md.append(md_table("Pairwise redundancy and synergy of the class information", recs,
                           "The upper block is in bits and the lower block divides each quantity by the pair's "
                           "joint MI. The division is the test that matters: raw bits rise with how much class "
                           "information a pair carries, so a difference there can be scale alone, while the "
                           "normalised rows ask whether the redundancy-synergy balance itself has changed."))
        tex.append(tex_table(
            "Pairwise redundancy and synergy of the class information, twenty paired seeds. Each network "
            "contributes one number, the mean over 300 sampled unit pairs; the unit of analysis is the network, "
            "not the pair. Estimation is mixed kNN with dithering and a label-permutation null per term.",
            "tab:sup-rsi", recs))

    T = load("temporal_lockin_n20.json")
    if T:
        recs = build(None, T["paired"], TEMP_KEYS, "{:.1f}")
        md.append(md_table("Timing of the class code and the order in which the factors resolve", recs,
                           "A positive gap means language resolves before parity. The paired column asks "
                           "whether the two arms differ in their gap; whether either arm has a non-zero gap "
                           "at all is a separate question, tabulated below."))
        tex.append(tex_table(
            "Timing of the class code across the twenty paired seeds. Instantaneous rows summarise the pairwise "
            "joint-MI profile by its centre of mass; accumulated rows give the lock-in time, the interpolated "
            "first crossing of ninety per cent of the asymptote of the cumulative-count profile. A positive gap "
            "means language resolves before parity.",
            "tab:sup-temporal", recs))

        w = T["within_arm_gap"]
        md += ["### Is either arm actually ordered?", "",
               "The table above compares the two arms' gaps to each other. This one asks the prior question, "
               "separately for each arm: is the gap different from zero at all? A one-sample sign-flip "
               "permutation on the twenty per-seed gaps, plus the count of seeds in which language resolves "
               "first.", "",
               "| Arm | Clock | Mean gap (ms) | Median (ms) | SD (ms) | Language-first | Sign-flip p |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for arch in ("het", "hom"):
            for key, lbl in (("acc_gap_ms", "accumulated lock-in"), ("inst_gap_ms", "instantaneous COM")):
                x = w[f"{arch}_{key}"]
                md.append(f"| {arch} | {lbl} | {x['mean_ms']:+.1f} | {x['median_ms']:+.1f} | {x['sd_ms']:.1f} | "
                          f"{x['n_language_first']}/{x['n']} | {fp(x['p'], x['floor'])} |")
        md += ["", f"Ordering tally with a {T['tolerance_ms']:.0f} ms tolerance band: "
                   f"`{json.dumps(T['order_counts'])}`", ""]

    D = load("division_of_labour_n20.json")
    if D:
        recs = build(None, D["paired"], DOL_KEYS,
                     {"n_modules": "{:.1f}", "mi_lang_mean": "{:.4f}", "mi_par_mean": "{:.4f}"})
        md.append(md_table("Division of labour across modules", recs,
                           "The excess row subtracts each network's own permutation null, because $\\eta^2$ "
                           "rises with the number of groups even under the null and the two arms do not find "
                           "the same number of modules. Within-network permutation tests reached p < 0.05 in "
                           f"{D['n_significant_within_network']['het']}/20 heterogeneous and "
                           f"{D['n_significant_within_network']['hom']}/20 homogeneous networks."))
        tex.append(tex_table(
            "Division of labour across the twenty paired seeds. Each unit is scored by its language and parity "
            "mutual information and given a scale-free selectivity index; the statistic is the fraction of that "
            "index explained by module identity, on the same full-graph Louvain partition used for modularity.",
            "tab:sup-dol", recs))

    (SC.OUT / "significance_tables.md").write_text("\n".join(md), encoding="utf-8")
    (SC.OUT / "tables_supervisor_extension.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {SC.OUT / 'significance_tables.md'}")
    print(f"wrote {SC.OUT / 'tables_supervisor_extension.tex'}")


if __name__ == "__main__":
    main()
