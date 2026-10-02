"""Assemble the cross-seed replication report from the result JSON.

The report is generated rather than written by hand so that its numbers cannot drift from the analyses. The
prose that carries the argument lives in REPLICATION.md; what this file produces is the evidence section that
document points at: for each of the supervisor's seed-1001 findings, the n = 20 value, the paired statistics
and a verdict.

The verdict vocabulary is deliberately narrow. "Replicates" means the direction held and the paired test
rejects at the 0.05 level after Holm adjustment within its table. "Holds in direction" means the sign is
consistent but the test does not reach that bar. "Does not replicate" means the direction failed or the
result is a null where a difference was claimed. A finding that reverses gets said so in those words.

Run: python code/supervisor_extension/make_report.py
"""
import json
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


def verdict(r, expect_sign, holm_p=None):
    """expect_sign: +1 if the supervisor's seed-1001 result had het above hom, -1 if below, 0 if a null.

    A sign that disagrees with seed 1001 is only a reversal if the disagreement is itself significant.
    Where it is not, the honest statement is that the claimed effect is absent, which is a different and
    weaker conclusion than the effect running the other way.
    """
    d, p = r["delta"], holm_p if holm_p is not None else r["perm_p"]
    if expect_sign == 0:
        return "null confirmed" if p >= 0.05 else "was read as a null, but differs at n = 20"
    if np.sign(d) != expect_sign:
        return "**reverses**" if p < 0.05 else "claimed effect absent"
    return "replicates" if p < 0.05 else "holds in direction, not significant"


# (json key, printed label, seed-1001 value as the supervisor reported it, expected sign of het minus hom)
CLAIMS = {
    "modularity_n20.json": [
        ("full_Qnorm", "Full-graph Q/Q_rand", "het 0.62, hom 1.39", -1),
        ("cum_Qnorm", "Cumulative-count Q/Q_rand", "het 1.10, hom 1.87", -1),
        ("full_sigma", "Small-worldness sigma", "hom up to 3.3, het about 1", -1),
        ("eta2_lang", "Community eta2 language", "het 0.13 higher", +1),
        ("eta2_par", "Community eta2 parity", "hom 0.23 higher", -1),
        ("eta2_4c_cum", "Cumulative community eta2 4-class", "het 0.11, hom 0.03", +1),
    ],
    "block_concentration_n20.json": [
        ("louvain", "Within/between weight ratio (Louvain)", "the block matrices", -1),
        ("fixedK", "Within/between weight ratio (fixed K)", "the block matrices", -1),
        ("louvain_cum", "Within/between weight ratio (cumulative)", "the block matrices", -1),
    ],
    "rsi_n20.json": [
        ("redundancy", "Redundancy (bits)", "het 0.104, hom 0.051", +1),
        ("synergy", "Synergy (bits)", "het 0.187, hom 0.097", +1),
        ("RSI", "RSI (bits)", "het 0.082, hom 0.046", +1),
        ("redundancy_n", "Redundancy / I_joint", "fraction survives", +1),
        ("synergy_n", "Synergy / I_joint", "fraction survives", +1),
        ("RSI_n", "RSI / I_joint", "null at seed 1001", 0),
    ],
    "temporal_lockin_n20.json": [
        ("inst_4-class_com_ms", "Instantaneous 4-class COM (ms)", "het 479, hom 413", +1),
        ("acc_4-class_lockin_ms", "4-class lock-in (ms)", "het 536, hom 360", +1),
        ("acc_language_lockin_ms", "Language lock-in (ms)", "het 612, hom 232", +1),
        ("acc_parity_lockin_ms", "Parity lock-in (ms)", "het 512, hom 408", +1),
        ("acc_gap_ms", "Lock-in gap, parity minus language (ms)", "het -100, hom +176", -1),
    ],
    "division_of_labour_n20.json": [
        ("eta2_SI_module", "eta2(selectivity ~ module)", "het 0.09, hom 0.27", -1),
        ("eta2_excess", "eta2 above its own permutation null", "not reported", -1),
        ("corr_L_P", "corr(language MI, parity MI)", "het overlapping, hom disjoint", +1),
        ("jaccard_top", "Jaccard of top language/parity units", "het 0.16, hom 0.05", +1),
    ],
}

SECTION = {
    "modularity_n20.json": ("Modularity of the functional network, by the package's statistic",
                            "The topological claim: the homogeneous graph is modular and small-world, the "
                            "heterogeneous one is not. Read this table together with the next one: on these "
                            "dense graphs Q/Q_rand is unreliable in both arms, so the nulls below are "
                            "evidence about the statistic rather than about the networks."),
    "block_concentration_n20.json": ("Modularity of the functional network, without a null",
                                     "The same question asked by comparing within-community edge weight "
                                     "against between-community edge weight directly. This is what the block "
                                     "matrices display, and it is the version that should be believed."),
    "rsi_n20.json": ("Redundancy and synergy of the class information",
                     "The information claim: heterogeneous pairs carry more of both, but the balance between "
                     "them is unchanged."),
    "temporal_lockin_n20.json": ("Timing and the order in which the factors resolve",
                                 "The processing claim: the homogeneous network resolves language first and "
                                 "parity second, the heterogeneous one integrates both concurrently and late."),
    "division_of_labour_n20.json": ("Division of labour across modules",
                                    "The keystone: module identity predicts what a unit codes for in the "
                                    "homogeneous network and not in the heterogeneous one."),
}


def main():
    out = ["# Cross-seed replication of the information analyses",
           "",
           f"Every result below is a within-seed paired comparison over {len(SC.SEEDS)} seeds, against the "
           "supervisor's",
           "seed-1001 values from `ANALYSIS_SUMMARY.md`. The seed-1001 column is what was claimed; the het and",
           "hom columns are the twenty-seed means; the verdict applies the Holm-adjusted sign-flip p within",
           "each section.",
           ""]

    for fname, claims in CLAIMS.items():
        M = load(fname)
        if not M:
            continue
        title, blurb = SECTION[fname]
        out += [f"## {title}", "", blurb, ""]
        rows = [(k, lbl, s1001, sgn, M["paired"][k]) for k, lbl, s1001, sgn in claims if k in M["paired"]]
        adj = holm([r[4]["perm_p"] for r in rows])
        out += ["| Quantity | seed 1001 | het (n=20) | hom (n=20) | Δ | d_z | sign-flip p | Holm | wins | verdict |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for (k, lbl, s1001, sgn, r), a in zip(rows, adj):
            out.append(f"| {lbl} | {s1001} | {r['het_mean']:.4g} | {r['hom_mean']:.4g} | {r['delta']:+.4g} | "
                       f"{r['dz']:+.2f} | {fp(r['perm_p'], r['perm_floor'])} | {fp(a)} | "
                       f"{r['wins']}/{r['n']} | {verdict(r, sgn, a)} |")
        out.append("")

        if fname == "temporal_lockin_n20.json":
            w = M["within_arm_gap"]
            out += ["The paired row above asks whether the two arms differ in their gap. Whether either arm has",
                    "a gap at all is the prior question, and it is the one the sequential reading depends on:",
                    "", "| Arm | Mean gap (ms) | Median | Language-first | Sign-flip p |", "|---|---:|---:|---:|---:|"]
            for arch in ("het", "hom"):
                x = w[f"{arch}_acc_gap_ms"]
                out.append(f"| {arch} | {x['mean_ms']:+.1f} | {x['median_ms']:+.1f} | "
                           f"{x['n_language_first']}/{x['n']} | {fp(x['p'], x['floor'])} |")
            out += ["", f"Ordering tally with a {M['tolerance_ms']:.0f} ms tolerance: `{json.dumps(M['order_counts'])}`", ""]

        if fname == "division_of_labour_n20.json":
            s = M["n_significant_within_network"]
            out += [f"Within-network permutation tests of module alignment reached p < 0.05 in "
                    f"{s['het']}/20 heterogeneous and {s['hom']}/20 homogeneous networks.", ""]

    p = SC.OUT / "replication_evidence.md"
    p.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
