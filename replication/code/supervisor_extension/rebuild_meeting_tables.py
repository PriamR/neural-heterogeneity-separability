"""Regenerate every statistics table in the meeting document with the full test triple.

The suite's convention is that no comparison is reported on one test: paired t, Wilcoxon and the exact
sign-flip permutation always appear together, so a reader can see when they disagree and when a
non-parametric test is sitting on its enumeration floor. The meeting document was hand-written and quietly
dropped two of the three.

Tables are rebuilt from the result JSON rather than edited in place, so the numbers cannot drift from the
analyses and a re-run refreshes them. Each table is located by an anchor line unique to it.

Both non-parametric tests floor at 2/2^20 = 1.9e-6 at this n, which is what a unanimous 20/20 reads as; a
value sitting on the floor is marked rather than printed as a bare number, so the ceiling is not mistaken
for a limit on the effect.

Run: python code/supervisor_extension/rebuild_meeting_tables.py
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
from stats_core import paired_triple

REPO = SC.REPO
DOC = REPO / "docs" / "MEETING - Structure and Replication (2026-07-30).md"
A = REPO / "docs" / "analysis_suite"
B = REPO / "data_cache" / "supervisor_extension"

J = {
    "score": json.load(open(A / "summary.json"))["scoreboard"],
    "shat": json.load(open(A / "shattering_n20_summary.json"))["paired"],
    "rank": json.load(open(A / "shattering_vs_rank_n20_summary.json")),
    "vc": json.load(open(A / "vc_shattering_n20_summary.json"))["paired"],
    "cap": json.load(open(A / "capacity_corr_n20_summary.json"))["paired"],
    "sig": json.load(open(A / "signal_dimension_n20_summary.json"))["paired"],
    "tg": json.load(open(A / "temporal_generalisation_n20_summary.json"))["paired"],
    "mod": json.load(open(B / "modularity_n20.json")),
    "bc": json.load(open(B / "block_concentration_n20.json"))["paired"],
    "rsi": json.load(open(B / "rsi_n20.json"))["paired"],
    "tmp": json.load(open(B / "temporal_lockin_n20.json"))["paired"],
    "dol": json.load(open(B / "division_of_labour_n20.json"))["paired"],
}
Z = np.load(B / "structure_matrices_n20.npz")


def fp(p, floor=None):
    if p is None or p != p:
        return "n/a"
    if floor is not None and p <= floor * 1.001:
        return "1.9e-06 †"
    if p >= 1e-3:
        return f"{p:.3f}"
    e = int(math.floor(math.log10(p)))
    return f"{p / 10 ** e:.1f}e{e:+03d}"


def r_row(label, r, fmt="{:.3f}", bold=False, winlabel=None):
    w = winlabel if winlabel is not None else f"{r['wins']}/{r['n']}"
    lab = f"**{label}**" if bold else label
    return (f"| {lab} | {fmt.format(r['het_mean'])} | {fmt.format(r['hom_mean'])} | {r['delta']:+.4g} | "
            f"{r['dz']:+.2f} | {fp(r['t_p'])} | {fp(r['w_p'], r.get('w_floor'))} | "
            f"{fp(r['perm_p'], r.get('perm_floor'))} | {w} |")


HEAD = ("| Metric | het | hom | Δ | d_z | paired t | Wilcoxon | sign-flip | wins |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|")


def table(rows, head=HEAD):
    return head + "\n" + "\n".join(rows)


# ---------------------------------------------------------------- table builders
def t_scoreboard():
    order = [("Task accuracy", "Task accuracy", "{:.3f}"),
             ("Linear decoder accuracy", "Linear decoder accuracy", "{:.3f}"),
             ("PCA n90 (1 ms)", "PCA n₉₀", "{:.1f}"),
             ("PCA participation ratio (1 ms)", "PCA participation ratio", "{:.2f}"),
             ("PC-reconstruction @ m=16", "PC-reconstruction @ m=16", "{:.3f}"),
             ("PC-reconstruction @ m=32", "PC-reconstruction @ m=32", "{:.3f}"),
             ("Active neurons", "Active units", "{:.1f}"),
             ("Mean firing rate", "Mean firing rate", "{:.3f}")]
    return table([r_row(lbl, J["score"][k], f) for k, lbl, f in order])


def t_shattering():
    return table([r_row("A, 8 conditions", J["shat"]["A_8cond"], "{:.4f}"),
                  r_row("B, 20 conditions", J["shat"]["B_20cond"], "{:.4f}")],
                 HEAD.replace("| Metric |", "| Set |"))


def t_rank():
    rows = [r_row(str(m), J["rank"]["paired_by_rank"][str(m)], "{:.3f}") for m in J["rank"]["ranks"]]
    return table(rows, HEAD.replace("| Metric |", "| Retained rank m |"))


def t_vc():
    return table([r_row("Shattering load P*", J["vc"]["P_star"], "{:.2f}"),
                  r_row("α_VC = P*/N", J["vc"]["alpha_vc"], "{:.3f}")])


def t_cap(setkey):
    d = J["cap"][setkey]
    return table([r_row("Manifold capacity α_M", d["alpha_M"], "{:.4f}"),
                  r_row("Manifold radius R_M", d["R_M"], "{:.4f}"),
                  r_row("Manifold dimension D_M", d["D_M"], "{:.3f}"),
                  r_row("Centroid correlation", d["center_corr"], "{:.4f}")])


def t_signal():
    return table([r_row("Signal participation ratio", J["sig"]["signal_pr"], "{:.3f}"),
                  r_row("Reliable directions", J["sig"]["n_reliable"], "{:.1f}")])


def t_tempgen():
    o = [("diag_mean", "Instantaneous decoding (diagonal)"), ("offdiag_mean", "Cross-time decoding (off-diagonal)"),
         ("gen_ratio", "Generalisation ratio"), ("long_lag_acc", "Long-lag accuracy"),
         ("long_lag_ratio", "Long-lag / diagonal"), ("gen_width_bins", "Generalisation width (bins)")]
    return table([r_row(l, J["tg"][k], "{:.4f}" if k != "gen_width_bins" else "{:.2f}") for k, l in o])


def t_mod():
    o = [("full_Qnorm", "Full-graph Q/Q_rand", "{:.3f}"), ("cum_Qnorm", "Cumulative-count Q/Q_rand", "{:.3f}"),
         ("full_sigma", "Small-worldness σ", "{:.3f}"), ("eta2_lang", "Community η² language", "{:.4f}"),
         ("eta2_par", "Community η² parity", "{:.4f}")]
    return table([r_row(l, J["mod"]["paired"][k], f) for k, l, f in o])


def t_blockconc():
    o = [("louvain", "Louvain (binned graph)"), ("fixedK", "Fixed K = 3, spectral"),
         ("louvain_cum", "Louvain (cumulative graph)")]
    rows = []
    for k, l in o:
        r = J["bc"][k]
        rows.append(r_row(l, r, "{:.2f}", winlabel=f"{r['n'] - r['wins']}/{r['n']}"))
    return table(rows, HEAD.replace("| Metric |", "| Partition |").replace("| wins |", "| hom higher |"))


def t_sigma_density():
    Sh = np.array(J["mod"]["densities_sweep"]["binary_het"]["sigma"], float)
    So = np.array(J["mod"]["densities_sweep"]["binary_hom"]["sigma"], float)
    rows = []
    for i, d in enumerate(J["mod"]["densities"]):
        h, o = Sh[:, i], So[:, i]
        ok = np.isfinite(h) & np.isfinite(o)
        r = paired_triple(h[ok], o[ok])
        rows.append(r_row(f"{d:.2f}", r, "{:.2f}", winlabel=f"{r['n'] - r['wins']}/{r['n']}"))
    return table(rows, HEAD.replace("| Metric |", "| Density |").replace("| wins |", "| hom higher |"))


def t_density_sweep():
    rows = []
    for wt in ("weighted", "binary"):
        Qh = np.array(J["mod"]["densities_sweep"][f"{wt}_het"]["Qnorm"], float)
        Qo = np.array(J["mod"]["densities_sweep"][f"{wt}_hom"]["Qnorm"], float)
        for i, d in enumerate(J["mod"]["densities"]):
            r = paired_triple(Qh[:, i], Qo[:, i])
            bold = wt == "binary" and r["perm_p"] < 0.05
            rows.append(r_row(f"{wt}, d = {d:.2f}", r, "{:.3f}", bold=bold,
                              winlabel=f"{r['n'] - r['wins']}/{r['n']}"))
    return table(rows, HEAD.replace("| Metric |", "| Weighting, density |").replace("| wins |", "| hom higher |"))


def t_tuning():
    rows = []
    for kind, klbl in (("binned", "binned"), ("cumulative", "cumulative")):
        for f, flbl in (("eta2_lang", "language"), ("eta2_par", "parity")):
            v = {a: np.array([Z[f"{s}_{a}__tun_{kind}_{f}"].mean() for s in SC.SEEDS], float)
                 for a in ("het", "hom")}
            r = paired_triple(v["het"], v["hom"])
            rows.append(r_row(f"{klbl}, {flbl}", r, "{:.4f}", bold=(f == "eta2_lang")))
    return table(rows, HEAD.replace("| Metric |", "| Graph, factor |"))


def t_rsi():
    o = [("I_marginal", "Single-unit MI I(X_i;Y)"), ("I_joint", "Pair joint MI"),
         ("redundancy", "Redundancy min(I_i,I_j)"), ("synergy", "Synergy I_joint − max(I_i,I_j)"),
         ("RSI", "RSI = synergy − redundancy"), ("redundancy_n", "Redundancy / I_joint"),
         ("synergy_n", "Synergy / I_joint"), ("RSI_n", "RSI / I_joint")]
    return table([r_row(l, J["rsi"][k], "{:.4f}", bold=(k in ("RSI", "RSI_n"))) for k, l in o])


def t_timing():
    o = [("acc_4-class_lockin_ms", "4-class lock-in (ms)"), ("acc_language_lockin_ms", "Language lock-in (ms)"),
         ("acc_parity_lockin_ms", "Parity lock-in (ms)"),
         ("inst_language_com_ms", "Instantaneous language COM (ms)"),
         ("acc_gap_ms", "Lock-in gap (parity − language, ms)"),
         ("inst_gap_ms", "Instantaneous COM gap (ms)")]
    return table([r_row(l, J["tmp"][k], "{:.1f}") for k, l in o])


def t_dol():
    o = [("eta2_SI_module", "η²(selectivity ~ module)", True),
         ("eta2_excess", "η² above its own permutation null", True),
         ("corr_L_P", "corr(language MI, parity MI)", False),
         ("jaccard_top", "Jaccard, top language/parity units", False)]
    rows = []
    for k, l, bold in o:
        r = J["dol"][k]
        rows.append(r_row(l, r, "{:.3f}", bold=bold, winlabel=f"{r['n'] - r['wins']}/{r['n']}"))
    return table(rows, HEAD.replace("| wins |", "| hom higher |"))


def t_blockmeans():
    """Within- and between-community block means, the two rows behind the aggregate figure."""
    import make_aggregate_block_figure as AG
    rows = []
    for lbl, pick in (("Within-community (diagonal)", "within"), ("Between-community (off-diagonal)", "between")):
        v = {}
        for arch in ("het", "hom"):
            out = []
            for sd in SC.SEEDS:
                M = AG.block_means(Z[f"{sd}_{arch}__W"], sd)
                eye = np.eye(M.shape[0], dtype=bool)
                out.append(np.nanmean(M[eye] if pick == "within" else M[~eye]))
            v[arch] = np.array(out, float)
        r = paired_triple(v["het"], v["hom"])
        rows.append(r_row(lbl, r, "{:.3f}", bold=(pick == "between"),
                          winlabel=f"{r['n'] - r['wins']}/{r['n']}"))
    return table(rows, HEAD.replace("| Metric |", "| Block |").replace("| wins |", "| hom higher |"))


# (section heading, which table under it, builder). Anchoring on a row's first cell is not safe: the
# fidelity table in section 8 also opens with "| Full-graph Q/Q_rand |", so a row-based anchor for the
# modularity table silently overwrote it. Headings are unique, so the target is addressed as "the Nth table
# following this heading".
TABLES = [
    ("## 1. Where the two networks start", 0, t_scoreboard),
    ("## 2. Shattering dimensionality", 1, t_shattering),          # 0 is the condition-set descriptor
    ("## 3. Rank-resolved shattering", 0, t_rank),
    ("## 4. VC shattering", 0, t_vc),
    ("**Set B, the 20 digit-by-language class manifolds:**", 0, lambda: t_cap("B_20cond")),
    ("**Set A, the 8 condition manifolds**", 0, lambda: t_cap("A_8cond")),
    ("### Signal dimensionality", 0, t_signal),
    ("### Temporal generalisation", 0, t_tempgen),
    ("### 9a. By the package's statistic, a null", 0, t_mod),
    ("### 9c. Without a null, the claim replicates", 0, t_blockconc),
    ("### 9d. Small-worldness, measured where the null works", 0, t_sigma_density),
    ("### 9e. And it survives density matching", 0, t_density_sweep),
    ("**This localises the modularity difference", 0, t_blockmeans),
    ("### 9g. The double dissociation is half a dissociation", 0, t_tuning),
    ("## 10. Redundancy and synergy", 0, t_rsi),
    ("### Timescale replicates without qualification", 0, t_timing),
    ("## 12. Division of labour", 0, t_dol),
]


def replace_table(text, heading, which, new_table):
    """Swap the `which`-th markdown table appearing after `heading` for `new_table`."""
    lines = text.splitlines()
    h = next((i for i, l in enumerate(lines) if l.startswith(heading)), None)
    if h is None:
        return text, False
    i, seen = h + 1, -1
    while i < len(lines):
        if lines[i].startswith("|"):
            start = i
            while i + 1 < len(lines) and lines[i + 1].startswith("|"):
                i += 1
            seen += 1
            if seen == which:
                return "\n".join(lines[:start] + new_table.splitlines() + lines[i + 1:]), True
        elif lines[i].startswith("# "):
            return text, False                     # ran into the next part without finding it
        i += 1
    return text, False


def main():
    s = DOC.read_text(encoding="utf-8")

    # the fidelity table is prose-formatted and must never be touched by a table rebuild; it was destroyed
    # once by a row-based anchor, so it is restored here if a previous run clobbered it
    if "| Quantity | Package (seed 1001) | This code (seed 1001) |" not in s:
        fid = ("| Quantity | Package (seed 1001) | This code (seed 1001) |\n"
               "|---|---|---|\n"
               "| Full-graph Q/Q_rand | het 0.62, hom 1.39 | het 0.626, hom 1.365 |\n"
               "| Cumulative-count Q/Q_rand | het 1.10, hom 1.87 | het 1.118, hom 1.810 |\n"
               "| Active units | het 47, hom 42 | het 47, hom 42 |\n"
               "| Mean edge MI (bits) | het 0.021, hom 0.012 | het 0.0205, hom 0.0123 |\n"
               "| η²(SI ~ module) | het 0.09 (p 0.26), hom 0.27 (p 0.007) | "
               "het 0.090 (p 0.257), hom 0.274 (p 0.006) |\n"
               "| hom lock-in, language then parity | 232 ms, 408 ms | 217 ms, 403 ms |")
        marker = "seed 1001 appears on its own; everything after it is twenty paired seeds."
        lines = s.splitlines()
        h = next(i for i, l in enumerate(lines) if marker in l)
        start = h + 1
        while start < len(lines) and not lines[start].startswith("|"):
            start += 1
        end = start
        while end + 1 < len(lines) and lines[end + 1].startswith("|"):
            end += 1
        s = "\n".join(lines[:start] + fid.splitlines() + lines[end + 1:])
        print("  restored the section 8 fidelity table")

    ok = 0
    for i, (heading, which, builder) in enumerate(TABLES, 1):
        s, done = replace_table(s, heading, which, builder())
        # this console cannot render every heading, and a print must never abort the run before the write
        safe = heading.encode("ascii", "replace").decode("ascii")
        print(f"  {'ok  ' if done else 'MISS'} [{i:>2}] {safe[:52]}")
        ok += done
    footnote = ("\n\n> † The Wilcoxon and sign-flip tests both floor at 2/2²⁰ = 1.9×10⁻⁶ at this n, which is "
                "what a unanimous 20/20 reads as. A p marked † sits on that floor: it is a maximally "
                "significant result, not a weak one.\n")
    if "floor at 2/2²⁰" not in s:
        s = s.replace("\n---\n\n## Contents", footnote + "\n---\n\n## Contents", 1)
    DOC.write_text(s, encoding="utf-8")
    print(f"\nreplaced {ok}/{len(TABLES)} tables")


if __name__ == "__main__":
    main()
