"""Assemble the supervisor-review presentation markdown from the stats JSONs, so every number in the tables
comes from the same source as the figures. Run after make_presentation_figs.py, build_report.py,
run_ablation_repr.py and make_ablation_dyn_fig.py.

Section order follows the evaluation chapter's argument: the gap, what changed, the information-structure
null, the dimensional account and its readout test, the fixed-tau ladder, the controls that rule out the
obvious alternatives, the abstraction geometry, and the ablation.

Run: python code/analysis_suite/make_presentation_md.py
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

REPO = HERE.parent.parent
OUT = REPO / "docs" / "analysis_suite"
P = json.load(open(OUT / "presentation_stats.json"))
T = json.load(open(OUT / "tables.json"))
GEO = json.load(open(OUT / "geometry_n20_summary.json")) if (OUT / "geometry_n20_summary.json").exists() else None
AR = json.load(open(OUT / "ablation_repr.json")) if (OUT / "ablation_repr.json").exists() else None
AD = json.load(open(OUT / "ablation_dyn.json")) if (OUT / "ablation_dyn.json").exists() else None


def fp(p, floor=None):
    return S.fmt_p(p, floor)


def paired_table(rows, a="het", b="hom"):
    """rows: list of sig_row dicts. Render as a markdown significance table."""
    if isinstance(rows, dict):
        rows = [rows]
    h = (f"| comparison | {a} mean | {b} mean | delta | Cohen dz | paired-t p | Wilcoxon p | "
         f"sign-flip p | wins/n |\n|---|---|---|---|---|---|---|---|---|")
    out = [h]
    for r in rows:
        out.append(f"| {r['comparison']} | {r['a_mean']:.3f} | {r['b_mean']:.3f} | {r['delta']:+.3f} | "
                   f"{r['dz']:+.2f} | {fp(r['t_p'])} | {fp(r['w_p'], r.get('w_floor'))} | "
                   f"{fp(r['perm_p'], r['perm_floor'])} | {r['wins']}/{r['n']} |")
    return "\n".join(out)


def fmt_row(r, label):
    return (f"| {label} | {r['het_mean']:.3f} | {r['hom_mean']:.3f} | {r['delta']:+.3f} | {r['dz']:+.2f} | "
            f"{fp(r['t_p'])} | {fp(r['w_p'], r.get('w_floor'))} | {fp(r['perm_p'], r['perm_floor'])} | "
            f"{r['wins']}/{r['n']} |")


def geo_row(key, label):
    return fmt_row(GEO["paired"][key], label)


def geo_dichotomy_row(field, label):
    """Paired test on one CCGP dichotomy, computed from the per-network records.

    The mean over the three dichotomies is deliberately not reported. With four factorial conditions and all
    three balanced dichotomies, each trained boundary is tested once with each label polarity, so the twelve
    train/test splits pair off and the mean is identically 0.5 for every network (verified: the three values
    sum to 1.5 to machine precision on all 40 networks). It measures nothing. The per-dichotomy values do.
    """
    from stats_core import paired_triple
    import numpy as np
    seeds = sorted(GEO["per_network"]["het"], key=int)
    a = np.array([GEO["per_network"]["het"][s][field] for s in seeds])
    b = np.array([GEO["per_network"]["hom"][s][field] for s in seeds])
    return fmt_row(paired_triple(a, b, label), label)


md = []
w = md.append

w("""# Final-paper figures and tables: numbers for supervisor review

The task-performance and geometry evidence for the temporal-heterogeneity study, in the order the evaluation
chapter argues it: the accuracy gap, what changed in the network, the information-structure null, the
dimensional account and its readout test, the fixed-tau ladder, the controls, the abstraction geometry, and the
ablation. Each figure carries its paired significance table.

**What was run.** Spoken Heidelberg Digits, four classes (chance 25%), 700 input channels into 64 recurrent LIF
hidden units into a 4-way readout, evaluated on 2264 held-out trials. Networks come in matched pairs: the
heterogeneous arm learns one membrane time constant per hidden neuron (64 free constants), the homogeneous arm
learns a single constant shared by all 64. Everything else, including initialisation, data order and training
schedule, is identical within a pair. Twenty seeds for the learned pair; ten of those seeds also carry the
scaled-init and fixed-tau controls.

**Reading conventions.**
- Every het-versus-hom comparison is **paired within seed**, so the only free difference is the hidden time
  constant.
- Significance is the **test triple**: paired t, Wilcoxon signed-rank, and the exact **sign-flip permutation**
  test (the primary). Its two-sided floor is `2/2^n`: **1.9e-6 at n = 20**, **0.002 at n = 10**. A p at the floor
  is *maximally* significant and is marked `(floor)`.
- Error bars and bands are **95% confidence intervals** (mean +/- t_{.975,n-1} * SE).
- Controls are compared to the learned homogeneous network on the same ten seeds. The `2 ms` rung is really
  **3.26 ms** (it clips at the floor), so it and the `3.5 ms` rung are one timescale trained twice; the spread
  between them is a useful read on run-to-run variation at fixed tau. Marked `2 ms*` in tables.

---

## 1. The accuracy gap

### 1.1 Per-seed accuracy difference (het minus hom)

![Per-seed accuracy](figs/presentation/G1a_accuracy_per_seed.png)
""")
w(paired_table(P["G1a"]))
w("""
The heterogeneous network wins the pairwise accuracy comparison on 18 of 20 seeds. The two reversals are
opposite cases. Seed 710 is a heterogeneous shortfall, the lowest het accuracy in the set, against a homogeneous
network whose constant stayed short. Seed 1004 is the other way round: its homogeneous constant drifted to the
199.5 ms ceiling and reached het-level accuracy from there. Section 2 shows that this is not a coincidence.

### 1.2 Accuracy by network type

![Accuracy grouped](figs/presentation/G1b_accuracy_grouped.png)
""")
w(paired_table(P["G1b"], a="control", b="learned hom"))
w("""
No control reaches the heterogeneous network. Freezing any single timescale costs eight to twelve accuracy
points against the learned homogeneous baseline, and the scaled-init control (Section 7.3) costs ten.

---

## 2. What changed: the learned time constants

![Tau distribution](figs/presentation/G3_tau_distribution.png)
""")
_g3 = P["G3"]
_long = {k: v for k, v in _g3["hom_taus_sorted"].items() if v > 20}
w(f"""
The heterogeneous per-neuron constants, averaged across the twenty seeds, cover the whole clip range with a
pooled median of **{_g3['het_pooled_median_ms']:.1f} ms**; **{_g3['het_frac_on_ceiling']:.1%}** of het units sit on
the 199.5 ms ceiling (a clip artefact, which is why het is summarised by its median).

The homogeneous constant is a single number per seed, and those numbers are **not all short**. They run from
**{_g3['hom_tau_min_ms']:.1f} ms to {_g3['hom_tau_max_ms']:.1f} ms**, a fiftyfold spread across seeds, with a
median of **{_g3['hom_median_ms']:.1f} ms**. Most seeds settle in the 4 to 11 ms band, but
**{_g3['hom_n_over_20ms']} of 20 land long**: {', '.join(f'seed {k} at {v:.1f} ms' for k, v in _long.items())}.

Those long solutions are worth looking at, because the only two seeds where the homogeneous network reaches
heterogeneous-level accuracy carry the two longest constants: seed 1004 at the ceiling (a 0.016 reversal) and
seed 1007 at 28.8 ms (a tie, het ahead by 0.004). The relationship is real but weak, and should be stated that
way: across all twenty seeds the rank correlation between the homogeneous constant and the accuracy gap is
**rho = {_g3['gap_vs_homtau_spearman']:.2f}, p = {_g3['gap_vs_homtau_p']:.3f}**, and the third long seed breaks
it outright, with seed 660 at 24.4 ms showing the **largest** gap in the whole set
({_g3['gap_by_long_seed']['660']:+.3f}). A long single constant looks necessary for the homogeneous network to
close the gap on its own, but it is plainly not sufficient. Worth a supervisor's view on whether this is worth
reporting as a within-experiment observation or holding back as underpowered at n = 20.

---

## 3. Not an information-structure effect (O-information)

### 3.1 Most-redundant and most-synergistic subset vs subset size

![Min/max O line](figs/F4_5_minmax_oinfo_clean.png)

### 3.2 O-information at k = 8 (reliability ceiling) and k = 10

![O-info bar](figs/presentation/G5b_oinfo_k8_bar.png)
""")
_OI = json.load(open(OUT / "oinfo_summary_clean.json"))
_lab = {"min": "min O (most synergistic)", "max": "max O (most redundant)", "spread": "spread (max - min)"}
for _k in (8, 10):
    _tab = _OI.get(f"k{_k}_table") or []
    _fpos = _OI.get(f"frac_pos_k{_k}") or {}
    if not _tab:
        continue
    w(f"\n**Subset size k = {_k}** (n = {_OI['n_by_k'][str(_k)]} paired seeds)\n")
    w(f"| statistic (O at k={_k}) | het | hom | delta | Cohen dz | sign-flip p | wins/n |\n"
      "|---|---|---|---|---|---|---|")
    for _r in _tab:
        w(f"| {_lab[_r[0]]} | {_r[1]} | {_r[2]} | {_r[3]} | {_r[4]} | {_r[5]} | {_r[6]} |")
    w(f"| fraction of subsets with O > 0 | {_fpos['het']:.0%} | {_fpos['hom']:.0%} | | | | |")
w("""
Both codes are **redundancy-dominated**, and nothing separates them at either subset size. The most-synergistic
subsets sit at the same slightly negative value (p = 0.66 at k = 8, p = 0.25 at k = 10), and neither the
most-redundant subset nor the interaction range comes close to significance (p = 0.15 at k = 8, and p = 0.46 and
0.43 at k = 10). Where there is any lean at all it is toward the homogeneous network having the wider range, not
the heterogeneous one. Heterogeneity does not shift the code toward synergy and does not buy a wider interaction
range. This null is what redirects the study from information structure to geometry.

**On the two subset sizes.** k = 8 is the reliability ceiling: with 512000 samples the entropy estimates are
already bias-dominated beyond it, and the redundancy fraction climbing from 78% at k = 8 to 87% at k = 10 in
*both* arms is that bias rather than a property of either code. k = 10 is therefore a robustness check, not a
second result, and it behaves as one should: the null holds and the effects shrink rather than grow. Both sizes
now cover all twenty paired seeds. The sweep also holds k = 12 records, but for only twelve of the twenty pairs,
so the figure stops at k = 10 where every point is the full set.

---

## 4. The dimensional account (PCA)

### 4.1 het vs hom

![PCA scree het/hom](figs/presentation/G7a_pca_scree_hom_het.png)
""")
w(paired_table([P["G7a"]["n90"], P["G7a"]["pr"], P["G7a"]["top5"]]))
w("""
The bulk of the spectrum is shared; the two networks separate in the tail. het reaches 90% of its variance
several components later (higher n90 and participation ratio) while concentrating *less* in its top five modes,
the signature of a code spread across more low-variance directions.

### 4.2 All network types

![PCA scree all](figs/presentation/G7b_pca_scree_fixed.png)
""")
w(paired_table(P["G7b"]["n90_vs_wdhom"], a="control", b="learned hom"))
w("")
w(paired_table(P["G7b"]["pr_vs_hom"], a="control", b="learned hom"))
w("""
Every frozen rung truncates the spectrum well below the learned networks, and the scaled-init control sits with
the learned homogeneous network rather than with het: raising activity does not spread the code. The two
dimension measures disagree on the rungs, and the disagreement is worth stating rather than smoothing over. On
n90 every rung sits nine to thirteen components below the learned homogeneous network, unanimously across seeds.
On participation ratio only the two short rungs separate (p = 0.006 and 0.025); the 15 ms and 35 ms rungs are
**not** significantly below the learned hom (p = 0.16 and 0.15). n90 is a tail measure and PR is weighted toward
the top of the spectrum, so freezing tau truncates the tail without much changing the head, the same shape of
effect Section 4.1 found between het and hom.

---

## 5. Does the extra dimension reach the readout? (PC-reconstruction)

### 5.1 het vs hom

![PC-recon het/hom](figs/presentation/G9a_pc_recon_hom_het.png)
""")
w(paired_table([P["G9a"][f"m={m}"] for m in [4, 8, 16, 32, 64]]))
w("""
The hidden state is projected onto its first m principal components, reconstructed, and pushed through the
network's own trained readout, so this asks whether the extra directions carry task information rather than
just variance. Below the crossover near m = 8 the homogeneous code is level or ahead, its task information
packed into a few components. Above it, het pulls away and climbs to a higher ceiling: hom is capped in a
low-dimensional subspace, het keeps gaining from directions hom does not have.

### 5.2 All network types

![PC-recon all](figs/presentation/G9b_pc_recon_all.png)
""")
_ceil = P["G9b"]["ceiling"]
w("| network | reconstruction ceiling (m = 64) |\n|---|---|")
for _k in ["het", "hom", "scaled", "2 ms*", "3.5 ms", "15 ms", "35 ms"]:
    w(f"| {_k} | {_ceil[_k]:.3f} |")
w("")
w(paired_table(P["G9b"]["m16_vs_hom"], a="control", b="learned hom"))
w("")
w(paired_table(P["G9b"]["m64_vs_hom"], a="control", b="learned hom"))
w("""
The two tables together are the point. At m = 16 three of the four fixed rungs are statistically level with the
learned homogeneous network: given sixteen components they reconstruct just as well. By m = 64 all four have
fallen eight to thirteen points behind, because they stopped gaining after roughly sixteen to twenty-four
components while hom kept climbing. The rungs are not worse at using a low-dimensional code, they are worse at
having a high-dimensional one. The scaled-init control tells the same story from the other side, saturating
lowest of all. The m = 64 ceiling is the intact network, so that column matches task accuracy to rounding.

---

## 6. The fixed-tau ladder

![Fixed-tau ladder](figs/presentation/G6_fixed_ladder_perseed.png)
""")
w(T["T4.8"])
w("""
One line per control seed, across its learned pair and the four frozen rungs. No fixed timescale reaches the
heterogeneous network; the best rung (15 ms) still sits eight points below the learned homogeneous baseline, and
every seed's learned spread stands above its whole fixed arm. The accuracy rows repeat Section 1.2; the
dimensional evidence for why the ladder falls short is in Sections 4.2 and 5.2, where the rungs appear alongside
these accuracy numbers.

The two shortest rungs are one timescale trained twice (3.26 and 3.5 ms), and they differ by 0.023 in accuracy.
That is a free estimate of run-to-run spread at fixed tau, and it is smaller than every rung-to-hom gap in the
table.

---

## 7. Controls: it is not recruitment and not firing rate

### 7.1 Per-seed active-neuron difference (het minus hom)

![Per-seed active](figs/presentation/G2a_active_per_seed.png)
""")
w(paired_table(P["G2a"]))
w("""
Recruitment is a **null**: the two networks are active on the same number of units, so the accuracy gap is not a
recruitment gap.

### 7.2 Active neurons by network type

![Active grouped](figs/presentation/G2b_active_grouped.png)
""")
w(paired_table(P["G2b"], a="control", b="learned hom"))
w("""
### 7.3 Firing rate

![Firing rate](figs/presentation/G4_firing_rate.png)
""")
w(paired_table(P["G4"]["paired"]))
w("")
w(paired_table(P["G4"]["grouped"], a="control", b="learned hom"))
w("""
Firing rate is modestly higher in het. The scaled-init control settles it: it fires at **0.215**, roughly twice
the learned homogeneous rate and well above het's own **0.130**, and its accuracy is **0.501**, twenty points
below het. Raising activity does not raise accuracy, so rate is not the mechanism.

---

## 8. Abstraction geometry and memory
""")
if GEO:
    w("""
![CCGP and shattering](figs/F4_6_ccgp.png)

![Memory capacity](figs/F4_14_memory.png)
""")
    w("| geometry measure | het | hom | delta | Cohen dz | paired-t p | Wilcoxon p | sign-flip p | wins/n |\n"
      "|---|---|---|---|---|---|---|---|---|")
    w(geo_row("shatter", "Shattering (mean decodability)"))
    w(geo_row("dec_XOR", "XOR decodability"))
    w(geo_dichotomy_row("ccgp_Parity", "CCGP, parity dichotomy"))
    w(geo_dichotomy_row("ccgp_Language", "CCGP, language dichotomy"))
    w(geo_dichotomy_row("ccgp_XOR", "CCGP, XOR dichotomy"))
    w(geo_row("MC", "Memory capacity"))
    w(geo_row("R_M", "Manifold radius R_M (lower is tighter)"))
    w(geo_row("D_M", "Manifold dimension D_M"))
    w(geo_row("alpha_c", "Manifold capacity alpha_c"))
    w(f"""
Shattering separates the two networks as cleanly as accuracy does: het is ahead on **all twenty seeds**, with the
sign-flip p at its floor, and it leads on the non-factorisable XOR dichotomy too. Memory capacity follows.

**Cross-condition generalisation does not separate them.** Only the parity dichotomy generalises above chance,
and it does so in both networks; language sits at chance and XOR below it, in both. So the heterogeneous
advantage is on the shattering (dimensionality) side, not the abstraction side.

One caution on that block. The **mean** CCGP over the three dichotomies is not reported here, and should not be
reported anywhere, because it is an algebraic constant. With four factorial conditions and all three balanced
dichotomies, every trained boundary is tested once under each label polarity, so the twelve train/test splits
pair off and the three CCGP values sum to exactly 1.5 on every network in the set. Their mean is 0.5 by
construction, for any data whatsoever, and the same identity pins the shuffle null to 0.5. Only the
per-dichotomy values carry information.

Manifold radius is the robust manifold result (het is tighter, and the sign-flip test is on the difference so
the wins column counts seeds where het is *larger*). Dimension is weak, and **alpha_c should not be read as a
discriminator**: the sweep is sensitive to its P/N grid and to the separability solver, and although neither
network is grid-floored here (het {GEO['alpha_c_floored']['het']}/20, hom {GEO['alpha_c_floored']['hom']}/20),
the gap is negligible.
""")
else:
    w("\n> geometry_n20_summary.json not found; run run_geometry_n20.py.\n")

w("""
---

## 9. Neuron ablation

Two versions. The **representational** ablation (9.1) drops neurons and re-decodes the class from the remaining
rate code with a fresh linear decoder, so it measures a neuron's contribution to the *decodable code*. The
**dynamical lesion** (9.2, 9.3) disconnects the neuron inside the recurrent forward pass and reads task accuracy
through the network's own readout, so it measures the contribution to the *computation*. The dynamical version is
the authoritative one.

The two are not on the same axis and must not be compared directly: the intact representational decode sits near
0.83 for het and 0.74 for hom, above either network's own task accuracy, because the decoder is refit. Both
figures therefore carry a second panel normalised to each network's intact value, which is the only fair way to
ask whether one degrades more gracefully than the other.

### 9.1 Representational ablation (re-decode the remaining rate code)

![Representational ablation](figs/presentation/G8a_ablation_repr.png)
""")
if AR and "tests" in AR:
    w(paired_table(AR["tests"]["frac_het_long_vs_hom"], a="het long-first", b="hom random"))
    w("")
    w(paired_table(AR["tests"]["frac_het_long_vs_short"], a="het long-first", b="het short-first"))
    w("""
Once the head start is removed the two codes degrade at nearly the same rate. All eight deltas point the same
way (het retains slightly more), and three levels reach p < 0.05 uncorrected, but the largest gap anywhere is
4.7 points at 56 units and nothing would survive correction for eight tests. Read it as a weak trend, not a
result. Ablation order is a null here too: long-first and short-first are within about one point of each other
at every level. At the level of the decodable rate code, class information sits evenly across the timescale
range.
""")
w("""
### 9.2 Dynamical lesion, het vs hom

![Dynamical ablation het/hom](figs/presentation/G8b_ablation_dyn_hom_het.png)
""")
if AD and "tests" in AD:
    w(paired_table(AD["tests"]["frac_het_long_vs_hom"], a="het long-first", b="hom random"))
    w("""
Inside the running network the picture reverses. het starts ten points higher, so the absolute curves cross
around 24 lesioned units; the normalised panel shows why. From 32 units on, het retains a **significantly
smaller** fraction of its intact accuracy than hom (sign-flip p = 0.002 or below at 32, 40, 56 and 63). The
heterogeneous solution is not a more robust one. It buys accuracy with a distributed code that needs its units.
""")
    w(paired_table(AD["tests"]["frac_het_long_vs_short"], a="het long-first", b="het short-first"))
    w("""
Ablation order does matter here, and only here. Removing the eight longest-tau units leaves 79% of accuracy;
removing the eight shortest leaves 64%, so the short units are worth roughly twice as much per neuron
(sign-flip p = 4.5e-4, 17 of 20 seeds). The gap closes by 24 units removed, as it must once both orders have
taken out most of the network. So the long-tau units carry class information that a fresh decoder can
read (Section 9.1) but that the network's own dynamics barely rely on, while the short-tau units are
load-bearing. That is consistent with the short single-tau optimum the derivation predicts, and it is the reason
the 15 ms rung is the best of the fixed ladder rather than the 35 ms one.

### 9.3 Dynamical lesion with the fixed-tau rungs

![Dynamical ablation all](figs/presentation/G8c_ablation_dyn_all.png)

The fixed rungs start lower and fall to chance sooner, with no ordering to test since their constants are equal.
""")

w("""
---

*Next: participation-ratio calculations and their visualisation, the empirical basis of the temporal-basis
theorem (to be added after this review).*
""")

full = "\n".join(md)
(OUT / "Presentation - Figures and Tables.md").write_text(full, encoding="utf-8")
print(f"wrote {OUT / 'Presentation - Figures and Tables.md'}")


# =====================================================================================
# Visuals-only copy: headings, figures and tables, with every explanatory paragraph removed.
# Kept: the H1 title, section/subsection headings, image embeds, markdown tables, and the short bold
# captions that label a table (e.g. "**Subset size k = 8** (...)"), which are identified by being followed
# by a table. Dropped: the intro/methods/reading-conventions prose, and every interpretation paragraph.
# =====================================================================================
def strip_to_visuals(text):
    lines = text.split("\n")
    n = len(lines)

    def next_nonblank(i):
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        return lines[j] if j < n else ""

    keep = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        is_heading = s.startswith("#")
        is_image = s.startswith("![")
        is_table = s.startswith("|")
        is_rule = s == "---"
        # a bold-led line is a table caption only if a table row follows it
        is_caption = s.startswith("**") and next_nonblank(i).lstrip().startswith("|")
        if is_heading or is_image or is_table or is_rule or is_caption:
            keep.append(ln)
        elif s == "":
            keep.append("")

    # collapse blank runs, drop rules that are not between content, tidy start/end
    out = []
    for ln in keep:
        if ln == "" and (not out or out[-1] == ""):
            continue
        out.append(ln)
    # remove a rule that directly precedes another rule or sits at the very top
    tidy = []
    for ln in out:
        if ln.strip() == "---" and (not tidy or tidy[-1].strip() == "---"):
            continue
        tidy.append(ln)
    while tidy and tidy[0].strip() in ("", "---"):
        tidy.pop(0)
    while tidy and tidy[-1].strip() in ("", "---"):
        tidy.pop()
    return "\n".join(tidy) + "\n"


visuals = strip_to_visuals(full)
# swap the H1 so the two files are not mistaken for each other
visuals = visuals.replace("# Final-paper figures and tables: numbers for supervisor review",
                          "# Figures and tables only (visuals, no commentary)", 1)
# drop the CCGP/shattering figure from this file (shattering is given as code in the appendix instead)
visuals = "\n".join(ln for ln in visuals.split("\n")
                    if "F4_6_ccgp.png" not in ln)

# Code appendix: how the key quantities are computed. Static reference text, appended after stripping so the
# fenced blocks survive (the visuals stripper drops everything that is not a heading, figure or table).
CODE_APPENDIX = r'''
---

## Appendix: how the calculations work

Reference snippets for the measures in this deck. `X` is the `[trials, 64]` matrix of per-trial hidden
firing rates, `y` the 4-way class label (the four classes are a 2x2 grid, `condition = 2*is_german + is_odd`).

### Shattering (cross-validated decodability of the dichotomies)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np

# three balanced ways to split the four classes into two groups of two
DICH = {"Language": {0: +1, 1: +1, 2: -1, 3: -1},   # English vs German
        "Parity":   {0: +1, 1: -1, 2: +1, 3: -1},   # even vs odd
        "XOR":      {0: +1, 1: -1, 2: -1, 3: +1}}    # the interaction

def decodability(X, lab):
    # can a linear classifier separate the +1 trials from the -1 trials, from the 64 rates?
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    accs = []
    for tr, te in cv.split(X, lab):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), lab[tr])
        accs.append(clf.score(sc.transform(X[te]), lab[te]))
    return float(np.mean(accs))

# relabel each trial +/-1 per dichotomy, decode it, average the three
dec = {k: decodability(X, np.array([DICH[k][c] for c in y])) for k in DICH}
shattering = float(np.mean([dec["Language"], dec["Parity"], dec["XOR"]]))
```

### Network memory function

```python
from sklearn.linear_model import Ridge

# hidden: [trials, time_bins, 64] hidden activity over time; u: [trials, time_bins] input drive over time
def memory_function(hidden, u, maxlag=25):
    n, NB, D = hidden.shape
    idx = np.random.default_rng(0).permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    r2 = []
    for tau in range(maxlag + 1):
        # predict the input that arrived tau bins earlier from the current hidden state
        Xtr, ytr = hidden[tr, tau:, :].reshape(-1, D), u[tr, :NB - tau].reshape(-1)
        Xte, yte = hidden[te, tau:, :].reshape(-1, D), u[te, :NB - tau].reshape(-1)
        sc = StandardScaler().fit(Xtr)
        pred = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
        ss_res = np.sum((yte - pred) ** 2); ss_tot = np.sum((yte - yte.mean()) ** 2) + 1e-12
        r2.append(max(0.0, 1 - ss_res / ss_tot))
    return np.array(r2)          # R^2 against lag; memory capacity MC = r2.sum()
```

### O-information of a neuron subset

```python
from syntropy.discrete.utils import construct_joint_distribution
from syntropy.discrete import o_information

# subset: [k, T] binary spike trains of k chosen neurons (1 ms bins, strided to 4 ms)
def o_info(subset):
    jd = construct_joint_distribution(subset.astype(int))
    return float(o_information(jd)[1])       # > 0 redundant, < 0 synergistic

# per network and per subset size k in {2,4,6,8,10}: draw random k-neuron subsets,
# and keep the most redundant (max O) and most synergistic (min O) subset at each k
```

### Effective dimension (PCA)

```python
# state: [samples, 64] time-resolved hidden state, samples = trials x time bins
C   = np.cov(state.T)                            # 64 x 64 covariance
ev  = np.linalg.eigvalsh(C)[::-1].clip(min=0)    # variance per principal component, largest first
evr = ev / ev.sum()                              # explained-variance ratio
n90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)   # PCs to reach 90% of the variance
pr  = float((ev.sum() ** 2) / np.square(ev).sum())      # participation ratio
top5 = float(evr[:5].sum())                             # variance held in the top 5 PCs
```

### Scaled-network setup

```python
import torch

# Start a single-constant (homogeneous) network at the heterogeneous firing rate by scaling its initial
# input and recurrent weights by a constant c (chosen per seed to match het's trained rate), then retrain.
C_SCALE = 6.87                                   # per-seed constant matching het's rate

def scale_init(state_dict, c):
    sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    sd["network.0.w"] = sd["network.0.w"] * c    # input weights
    sd["network.0.v"] = sd["network.0.v"] * c    # recurrent weights
    return sd                                    # output weights left unscaled

model.load_state_dict(scale_init(model.state_dict(), C_SCALE))
# ... then train with the usual recipe
```
'''

visuals = visuals.rstrip() + "\n" + CODE_APPENDIX
(OUT / "Presentation - Visuals Only.md").write_text(visuals, encoding="utf-8")
print(f"wrote {OUT / 'Presentation - Visuals Only.md'}")
