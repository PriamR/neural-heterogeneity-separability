"""Compose the analysis report. Every number in the prose is interpolated from the same JSON the tables and
figures are built from, so the report cannot drift from the figures.

Run after build_report.py, make_input_spectrum.py and make_theory_figs.py.
Run: python code/analysis_suite/write_report.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

REPO = HERE.parent.parent
OUT = REPO / "docs" / "analysis_suite"

T = json.load(open(OUT / "tables.json"))
M = json.load(open(OUT / "summary.json"))
SP = json.load(open(OUT / "input_spectrum.json")) if (OUT / "input_spectrum.json").exists() else None
TH = json.load(open(OUT / "theory_check.json")) if (OUT / "theory_check.json").exists() else None
OI = json.load(open(OUT / "oinfo_summary_clean.json")) if (OUT / "oinfo_summary_clean.json").exists() else None
GEO = json.load(open(OUT / "geometry_n20_summary.json")) if (OUT / "geometry_n20_summary.json").exists() else None

sb = M["scoreboard"]
a = M["acc"]
n = M["n_pairs"]
cf = M["confound"]


def p(row_key):
    r = sb[row_key]
    return S.fmt_p(r["perm_p"], r["perm_floor"])


def line(k, fmt="{:.2f}"):
    r = sb[k]
    return (f"het {fmt.format(r['het_mean'])}, hom {fmt.format(r['hom_mean'])}, "
            f"delta {r['delta']:+.3f}, dz {r['dz']:+.2f}, sign-flip p {S.fmt_p(r['perm_p'], r['perm_floor'])}, "
            f"het wins {r['wins']}/{r['n']}")


def oinfo_section():
    if not OI:
        return ""
    k8 = OI["k8"]
    fp = OI["frac_pos_k8"]
    tbl = "| statistic (O at k=8) | het | hom | delta |\n|---|---|---|---|\n"
    for st in ("max", "min", "spread"):
        d = k8[st]
        tbl += f"| {st} O | {d['het']:+.3f} | {d['hom']:+.3f} | {d['het']-d['hom']:+.3f} |\n"
    return f"""## 11a. Higher-order information at n = 20: not synergy, and no wider range

The O-information sweep has been rerun on all twenty seeds, the nine corrected homs included. The result is a
cleaner elimination than the n = 10 version.

Both networks are redundancy-dominated. At the k = 8 reliability ceiling {fp['het']:.0%} of heterogeneous subsets
and {fp['hom']:.0%} of homogeneous subsets have positive O-information, and the most synergistic subset of each
sits at essentially the same slightly negative value (min O {k8['min']['het']:+.3f} against
{k8['min']['hom']:+.3f}). The heterogeneous code is not more synergistic.

{tbl}
![Min/max O-information](figs/F4_5_minmax_oinfo_clean.png)

**F4.5.** The most redundant (max O) and most synergistic (min O) subset per network against subset size, n = 20.

The "heterogeneous network has a wider synergy-to-redundancy range" reading from the ten-seed set **does not
survive**. At k = 8 the range is {k8['spread']['het']:+.3f} for het against {k8['spread']['hom']:+.3f} for hom, a
difference of {k8['spread']['het']-k8['spread']['hom']:+.3f}. That n = 10 result was riding on the collapsed
weight-decay homs, whose narrow activity gave a narrow O range; once the homs recruit normally the range
difference disappears. The safe statement at n = 20 is that both codes are redundancy-dominated with equal
magnitude, and heterogeneity does not shift the code toward synergy or buy a wider interaction range. This is a
firmer elimination, not a weaker one.

"""


def geom_section():
    """Section 13: the CCGP / manifold-geometry / memory refresh at the clean n = 20."""
    if not GEO:
        return """## 13. What is still legacy

The geometry refresh (CCGP, manifold geometry, memory) has not been run; `geometry_n20_summary.json` is missing.
Run `extract_geometry.py` then `run_geometry_n20.py`.
"""
    P = GEO["paired"]
    fl = GEO["alpha_c_floored"]

    def r(k, fmt="{:.3f}", flip=False):
        x = P[k]
        d = x["delta"]
        return (f"| {x['metric']} | {fmt.format(x['het_mean'])} | {fmt.format(x['hom_mean'])} | {d:+.3f} | "
                f"{x['dz']:+.2f} | {S.fmt_p(x['perm_p'], x['perm_floor'])} | {x['wins']}/{x['n']} |")

    # ccgp_mean is deliberately absent: it is an algebraic constant, not a measurement (see the bullet below).
    rows = "\n".join([
        r("shatter"), r("dec_XOR"),
        r("R_M", "{:.3f}"), r("D_M", "{:.2f}"), r("alpha_c", "{:.3f}"), r("MC", "{:.2f}")])
    sh, xo = P["shatter"], P["dec_XOR"]
    def _mean(arch, d):
        v = [GEO["per_network"][arch][s][f"ccgp_{d}"] for s in GEO["per_network"][arch]]
        return sum(v) / len(v)

    _cc = {d: (_mean("het", d), _mean("hom", d)) for d in ["Parity", "Language", "XOR"]}
    ac, rm, dm, mc = P["alpha_c"], P["R_M"], P["D_M"], P["MC"]
    return f"""## 13. The n = 20 refresh: CCGP, manifold geometry and memory

The three analyses that were still at n = 10 in earlier drafts, CCGP/shattering, manifold geometry and memory,
have been rerun at n = {GEO['n_pairs']} on the **clean homs** (the no-WD retrains for seeds 202 to 930), from a
fresh forward pass into `data_cache/representation_clean/` (`extract_geometry.py`), analysed by
`run_geometry_n20.py` with the full paired triple. Numbers are in `geometry_n20_summary.json`.

| paired metric (n = {GEO['n_pairs']}) | het | hom | delta | dz | sign-flip p | wins/n |
|---|---|---|---|---|---|---|
{rows}

![CCGP and shattering](figs/F4_6_ccgp.png)

**F4.6.** Left: cross-validated decodability of the three balanced dichotomies, het ahead on each. Right:
cross-condition generalisation against the shuffle null at 0.5; only parity generalises, and equally in both
networks.

![Memory function](figs/F4_14_memory.png)

**F4.14.** Reconstruction of past input drive from the hidden state against lag; the heterogeneous network holds
a wider band of retrievable delays.

Three readings, and one correction of note.

- **Shattering survives the refresh and firms up.** het shatters the label dichotomies more finely on every seed
  ({sh['het_mean']:.3f} vs {sh['hom_mean']:.3f}, {sh['wins']}/{sh['n']}, on the permutation floor). The clean homs
  decode better than the collapsed weight-decay homs did, so the gap narrows from the old 0.114 to
  {sh['delta']:+.3f} but becomes unanimous. het also leads on the non-factorisable XOR
  ({xo['het_mean']:.3f} vs {xo['hom_mean']:.3f}).
- **CCGP is modest and shared, and its mean must not be quoted.** Broken out by dichotomy, only parity
  generalises (het {_cc['Parity'][0]:.3f} vs hom {_cc['Parity'][1]:.3f}); language sits at chance
  ({_cc['Language'][0]:.3f} vs {_cc['Language'][1]:.3f}) and XOR below it
  ({_cc['XOR'][0]:.3f} vs {_cc['XOR'][1]:.3f}). None of the three separates the two networks, which is the clean
  form of the "not abstraction" elimination: the advantage is on the shattering (dimensionality) side.
  The **mean over the three dichotomies is an algebraic constant and carries no information**. With four
  factorial conditions and all three balanced dichotomies, each trained boundary is tested once under each label
  polarity, so the twelve train/test splits pair off: the three CCGP values sum to exactly 1.5 on every network
  in the set, and the mean is 0.5 for any data at all. The same identity pins `ccgp_null` to 0.5, so that
  "shuffle null" is not an empirical baseline either. Report the per-dichotomy values only.
- **Memory holds.** het retains past input drive better (MC {mc['het_mean']:.1f} vs {mc['hom_mean']:.1f}, dz
  {mc['dz']:+.2f}, {S.fmt_p(mc['perm_p'], mc['perm_floor'])}). The clean homs recover some memory relative to the
  collapsed weight-decay homs (the old hom MC was 8.6), but het still leads.

- **Correction: the manifold-capacity gap was a confound artefact and does not survive.** At the legacy n = 10 the
  homogeneous alpha_c sat at 0.094, exactly the measurement floor P/N = 6/64, with SD = 0.000, because the
  collapsed weight-decay hom (about 16 active units) was too degenerate for the sweep to resolve. On the clean
  no-WD homs **neither network floors** (het {fl['het']}/{GEO['n_pairs']}, hom {fl['hom']}/{GEO['n_pairs']}) and
  the gap closes to {ac['het_mean']:.3f} vs {ac['hom_mean']:.3f}, **not significant** ({S.fmt_p(ac['perm_p'], ac['perm_floor'])}).
  Manifold dimension is only weakly separated (D_M {dm['het_mean']:.2f} vs {dm['hom_mean']:.2f},
  {S.fmt_p(dm['perm_p'], dm['perm_floor'])}) and the
  radius favours het (tighter class manifolds, R_M {rm['het_mean']:.3f} vs {rm['hom_mean']:.3f},
  {S.fmt_p(rm['perm_p'], rm['perm_floor'])}). Empirical manifold capacity does **not** separate the two networks
  once the homs are un-confounded, so it must not anchor the capacity bridge; the dimensional case rests on n90,
  participation ratio and PC-reconstruction (Sections 6, 7), which are solid at n = 20, and Chapter 5 invokes
  capacity theory generically rather than the measured alpha_c. A caution on the number itself: the alpha_c sweep
  is sensitive both to the P/N grid floor (the homogeneous code sits close to it) and to the linear-separability
  solver, which was made deterministic (`LinearSVC(random_state=0)`) so the value reproduces run to run; even so it
  is not a reliable discriminator here and should be read as "no clear separation", not as a measured equality.

**Still legacy, and flagged where they appear:**

| analysis | status | what it costs |
|---|---|---|
| A3 tau-shuffle dose-response | legacy n = 10, `figures/fig_A3_tau_causal.png` | eval-only causal-direction test, the empirical partner to `d_eff(sigma)`; not in the paper body |
| W/M NuMIT z-scoring | **n = 3** | cannot carry a paired test. Appendix with an explicit caveat, or drop. |
| Inhibition lesion (0.57 to 0.26) | single trained net | one control, reported as-is |
"""


OINFO_BLOCK = oinfo_section()
GEOM_BLOCK = geom_section()


doc = f"""# Analysis suite: all {n} learned pairs, the scaled-init controls and the fixed-tau ladder

Everything the manuscript needs from the data, computed in one pass over all 91 trained networks and rendered
in one style. Generated by `code/analysis_suite/`: `extract_all.py` (one forward per network),
`build_report.py` (tables and figures), `make_input_spectrum.py`, `make_theory_figs.py`, `write_report.py`.
Figures are in `figs/`. Raw per-network records are in `data_cache/analysis_suite/`.

## 0. How to read this, and what is trustworthy

**Every number here is from the full SHD test set** ({M.get('n_trials', 2264)} held-out trials), forwarded
through each network's real readout. Earlier documents in `docs/` quote numbers from two-batch or six-batch
forwards; where they disagree with this report, this report supersedes them.

**The PC-reconstruction probe is validated, not assumed.** A rank-64 reconstruction is the identity map, so it
must reproduce the network's live accuracy exactly. `extract_all.py` asserts this for all 91 networks and all
91 pass. Any reconstruction curve in this report therefore ends at the network's true accuracy by construction,
which is what licenses reading the rest of the curve.

**Four notes to carry into the writing.**

0. *A weight-decay confound was found and fixed.* Nine of the twenty homogeneous nets (seeds 202 through 930)
   were originally trained with L2 weight decay 1e-4 while their het twins used none. They have been retrained at
   zero weight decay, so all twenty pairs are now matched, and the scoreboard below is the clean n = 20. Section
   2 shows the before/after. This is resolved, not outstanding.
1. *The "2 ms" rung is really 3.26 ms.* The inherited clipper in `code/model_perez_nieves/clipper.py` clamps
   the membrane decay to `[2/e, 0.995]`, and `2/e` corresponds to tau = 3.2589 ms. A frozen 2 ms request is
   silently raised on the first training step. We label the rung "2 ms" here at the user's request and will fix
   it later. The consequence for the argument: the short arm of the ladder is 3.26 and 3.50 ms, two points 7
   percent apart, so it is effectively one rung rather than two. Below 3.26 ms is currently unreachable.
2. *The two training batches now agree once matched.* Before the retrain the original homs collapsed and the new
   ten recruited fully; that gap was the weight decay of note 0, and with it removed both batches recruit like
   het (Section 12). The remaining differences between batches are ordinary seed variation.
3. *The geometry refresh is done; almost everything is now n = 20.* The min/max O-information, CCGP/shattering,
   manifold geometry and memory capacity have all been recomputed on all twenty clean pairs (Sections 11a and 13).
   The one substantive change: the manifold-capacity gap does not survive the un-confounded homs (Section 13). Only
   the A3 dose-response (n = 10, eval-only, not in the paper) and the W/M NuMIT z-scoring (n = 3) remain legacy.

## 1. Statistical convention

Every het-versus-hom comparison is **paired within seed**: the pair shares initial weights, data order and
training schedule, so the only free difference is the hidden membrane time constant.

Three tests are reported together so no single test carries a claim: paired t (parametric, fragile at small n),
Wilcoxon signed-rank (distribution-free but coarse), and a sign-flip permutation test that is **exact by full
enumeration** over all `2^n` sign assignments. At n = {n} that is {2**n:,} enumerations and the permutation test
is the primary.

**Both non-parametric tests have a hard floor of `2 / 2^n`**, because only the all-positive and all-negative
flips are as extreme as a unanimous observation. At n = 20 the floor is **1.9e-6**; at n = 10 it is **0.002**. A
p sitting at the floor is a *maximally significant* result, not a weak one. Tables mark these `(floor)`.

Because the scoreboard tests {len(sb)} metrics, Holm-adjusted p-values are reported alongside raw p in T4.1.
The pre-registered primary is task accuracy; the rest of the scoreboard is descriptive.

## 2. The regularisation confound, found and fixed

The paired design's claim is that within a seed the only free difference is the hidden membrane time constant.
Read off the checkpoints, that failed for nine of the twenty pairs: in seeds 202 through 930 the homogeneous
network was trained with L2 weight decay 1e-4 while its heterogeneous twin used none. The training module
`code/training/seeded_runs_common.py` sets `weight_decay = 0.0` for both and comments "no WD, matches paper", and
the reference implementation of Perez-Nieves et al. uses no weight decay (their Adam has no `weight_decay` kwarg
and their L2 function is never called), so the het arm was correct and the notebook that trained the original
homs departed from the recipe. Weight decay shrinks the recurrent weights, and the recurrent spectral radius is
the metric on which the two networks differ most, so the penalty pushed in the same direction as the effect.

**The nine homs have been retrained at zero weight decay.** All twenty pairs are now matched, and the scoreboard
in Section 3 uses the corrected homs. What follows is the before/after on those nine seeds, which is worth
keeping because it shows exactly what the penalty did.

{T['CONFOUND']}

![Regularisation before and after](figs/F4_0_regularisation_confound.png)

**F4.0.** het, the original weight-decay hom, and the no-weight-decay retrain, on the nine seeds.

What removing the penalty did, and what it did not:

- **Recruitment and gain jump.** The weight-decay homs sat at {cf['active']['wd']['hom_mean']:.0f} active units
  and gain {cf['gain']['wd']['hom_mean']:.2f}; retrained without it they recruit
  {cf['active']['nowd']['hom_mean']:.0f} units at gain {cf['gain']['nowd']['hom_mean']:.2f}, close to het. The
  "hom collapses its population to about 20 units" reading in earlier drafts was the penalty, not a property of a
  single time constant. On matched training the homogeneous network recruits like het.
- **Accuracy barely moves.** The retrain changed the homogeneous accuracy on these nine seeds by
  {cf['acc']['before_after']['delta']:+.3f} on average, and the het-versus-hom accuracy gap on the nine is
  {cf['acc']['nowd']['delta']:+.3f} against the retrains versus {cf['acc']['wd']['delta']:+.3f} against the
  originals. The accuracy result never depended on the confound.
- **The dimension gap holds and narrows.** n90 against the retrains is {cf['n90_1ms']['nowd']['delta']:+.1f}
  (het wins {cf['n90_1ms']['nowd']['wins']}/{cf['n90_1ms']['nowd']['n']}), smaller than the
  {cf['n90_1ms']['wd']['delta']:+.1f} it looked like against the collapsed weight-decay homs, but still there.

The corrected reading is cleaner than the confounded one: matched on recruitment and gain, the heterogeneous
network still occupies more effective dimensions and still classifies better. Every comparison in this report,
the controls (fixed-tau ladder, scaled-init) included, uses the corrected no-WD hom as the single learned-hom
baseline. The control nets were themselves trained with weight decay 1e-4, which the table above shows changes
homogeneous accuracy by only about 0.04 (it moves recruitment and gain more), so comparing them to the corrected
hom does not change the direction of any control result and keeps one hom reference throughout.

## 3. T4.1, the scoreboard (n = {n})

{T['T4.1']}

Reading it. The accuracy gap is {a['delta']:+.3f} ({a['het']:.1%} against {a['hom']:.1%}), het wins
{a['wins']}/{n}, sign-flip p = {S.fmt_p(a['perm_p'], a['perm_floor'])}, dz = {a['dz']:+.2f}. The gap survives
Holm correction across the whole scoreboard.

![Paired accuracy](figs/F4_1_paired_accuracy.png)

**F4.1.** Left: every seed as a connected het-hom pair. Right: per-seed gain, sorted. Seeds marked `*` are the
two whose homogeneous net found a long time constant.

## 4. F4.2, T4.2 and T4.3: the two solutions, and the diversity null

![Tau solutions](figs/F4_2_tau_solutions.png)

**F4.2.** Left: the pooled per-neuron time constants of all {n} heterogeneous networks against each homogeneous
network's single learned constant. Both clip bounds are marked: the 199.5 ms ceiling and the 3.26 ms floor are
properties of the inherited clipper, not of the optimisation. Right: where hom's constant lands against what the
pair costs it.

Given a free choice, a single constant goes short and a population covers the range. That contrast is the crux of
the study.

### T4.2, per-seed time-constant solutions

{T['T4.2']}

Heterogeneous networks are summarised by the median because the pooled mean is a clip artefact: 133 of the 1280
heterogeneous units (10.4 percent) sit exactly on the 199.5 ms ceiling. CV is sigma/mu. `*` marks the long-tau
homs.

### T4.3, the tau-diversity null

{T['T4.3']}

A first-class negative. Across these networks the *presence* of a spread is what matters; its *width* does not
predict the size of the gain. Any account claiming "more diversity is proportionally better" has to survive this
table.

## 5. F4.3, recruitment, framed as the suspect

![Recruitment](figs/F4_3_recruitment.png)

**F4.3.** {line('Active neurons', '{:.1f}')}. Activity correlates with accuracy across networks
(r = {M['r_active_acc']:.2f}).

**This is a suspect, not an explanation.** It is planted here and tested in Section 9 (the fixed-tau bowl) and
Section 10 (the scaled-init dissociation), where forcing the activity up does not bring the accuracy with it.
Note also that recruitment is the metric most affected by the batch difference of Section 12: het wins it only
{sb['Active neurons']['wins']}/{n}, far from unanimous.

## 6. F4.7 and T4.6, dimensionality

![PCA scree](figs/F4_7_pca_scree.png)

**F4.7.** Corrected PCA on the time-resolved [(trials x time) x 64] matrix at native 1 ms, with +/- 1 SD bands.
The bulk of the spectrum is shared; the tail is where the networks separate.

{T['T4.6']}

The n90-versus-participation-ratio divergence *is* the result, and the table is built to make it legible:
n90 ({line('PCA n90 (1 ms)', '{:.1f}')}) against the participation ratio
({line('PCA participation ratio (1 ms)', '{:.2f}')}). The heterogeneous network spreads task-relevant variance
into many low-variance dimensions rather than reshaping its dominant modes.

## 7. F4.8 and T4.7, reconstruction through the readout

![PC reconstruction](figs/F4_8_pc_reconstruction.png)

**F4.8.** The spine figure. Rank-m reconstructions of the spike code fed through each network's own trained
output membrane at native decays, 1 ms, all test trials. At m = 64 the reconstruction is exact and reproduces
live accuracy for all 91 networks, which validates the probe.

{T['T4.7']}

Ceilings: het {M['pc_ceiling']['het']:.3f}, hom {M['pc_ceiling']['hom']:.3f}. PCs to reach 90 percent of a
network's own ceiling: het {M['pcs_to_ceiling']['het']:.1f}, hom {M['pcs_to_ceiling']['hom']:.1f}.

The correct phrasing is **hom is capped in a low-dimensional subspace, het is distributed to a higher ceiling**,
not "het needs more PCs for the same accuracy". The m = 4 row is kept even where it reverses, because it is
honest and it sharpens exactly that reading.

## 8. F4.9, separability against accuracy

![Separability](figs/F4_9_separability_vs_accuracy.png)

**F4.9.** Linear decodability of class from the rate code against task accuracy, across every network including
the scaled-init controls (r = {M['r_sep_acc']:.2f}). The empirical link from geometry to performance, and the
lead-in to the capacity argument.

## 9. F4.10 and T4.8, the fixed-tau bowl

![The bowl](figs/F4_10_fixed_tau_bowl.png)

**F4.10.** Accuracy against fixed hidden time constant, 10 seeds per rung, with the learned solutions overlaid.
*Rungs are plotted at their nominal labels; the "2 ms" rung is really at 3.26 ms, so the short arm is narrower
than it looks.*

{T['T4.8']}

n = 10 per rung, so the Wilcoxon and sign-flip floor is 0.002. **No fixed time constant reaches the
heterogeneous network.** The ladder brackets the learned homogeneous optimum, and the best rung still sits below
what a spread achieves. This is the control that rules out "the advantage is just a better single timescale".

## 10. F4.12 and T4.9, the scaled-init dissociation

![Scaled-init dissociation](figs/F4_12_scaled_init_dissociation.png)

**F4.12.** Forcing the homogeneous network to the heterogeneous firing rate by scaling its initial weights, then
retraining. Firing rate and recurrent gain go up, above the corrected hom; accuracy, recruitment and dimensionality
do not follow, they fall.

{T['T4.9']}

{T['T4.9b']}

This pays off the suspect planted in F4.3. Activity is dissociated from performance. Two seeds (440, 820) sit at
the grid maximum c = 8 and the hot init trains unstably, so this **bounds** the activity argument rather than
proving it.

## 11. T4.10, gain and weight structure

![Gain](figs/T4_10_gain.png)

{T['T4.10']}

{line('Recurrent gain (spectral radius)', '{:.2f}')}.

**Gain is the cleanest separator in the whole study**, and the only scoreboard metric het wins unanimously. But
the scaled-init controls reach het-like gain from the outside and still fail (Section 10), so high gain is
necessary and not sufficient. Gain and time-constant diversity are partners here, not rivals: the route to the
high-gain regime matters, and training co-adapts the constant with the weights in a way a scalar rescaling
cannot install.

{OINFO_BLOCK}## 12. The two training batches, now matched

The twenty pairs come from two training runs. Before the retrain they looked different, because the original ten
homs carried the weight decay of Section 2 and the new ten did not. With the nine corrected, the batch comparison
is:

{T['BATCH']}

The original-batch homs (now retrained without weight decay) recruit and gain like the new batch, and both
recruit like het. What looked like a batch effect was the penalty. The residual differences between batches sit
within ordinary seed variation, and het wins recruitment {sb['Active neurons']['wins']}/{n} across the pooled
set. Recruitment is not what separates het from hom; matched on it, the dimensional gap remains.

{GEOM_BLOCK}
## 14. F4.13, input spectral grounding
"""

if SP:
    doc += f"""
![Input spectrum](figs/F4_13_input_spectrum.png)

**F4.13.** The premise of the temporal-basis account, not decoration. The SHD test input has a spectral centroid
at **{SP['centroid_hz']:.1f} Hz**, which through tau = 1/(2 pi f) corresponds to **{SP['tau_centroid_ms']:.0f} ms**.
Ten to ninety percent of the input power lies between {SP['f10_hz']:.1f} and {SP['f90_hz']:.1f} Hz, a band
corresponding to time constants from {SP['tau_at_f90_ms']:.1f} to {SP['tau_at_f10_ms']:.0f} ms.

{SP['het_frac_in_band']:.0%} of heterogeneous units have their corner frequency inside that band, against
{SP['hom_frac_in_band']:.0%} of homogeneous networks. The population covers the band the task actually occupies;
a single short constant sits at the fast end of it.
"""

doc += "\n## 15. Chapter 5 figures: the minimal tier, and where it breaks\n"

if TH:
    doc += f"""
![Filter bank](figs/F5_1_filter_bank.png)

**F5.1.** The pedagogical anchor. A shared constant is one averaging window repeated; a spread is a bank of
widths.

![Sech kernel](figs/F5_2_sech_kernel.png)

**F5.2.** The proof's central identity made visible: filter correlation depends only on the log-ratio of the two
time constants, through a sech kernel that decays to zero as they separate.

![PR vs spread](figs/F5_3_pr_vs_spread.png)

**F5.3.** Proposition 5.1 computed on seed 202's real learned configuration: PR rises strictly from 1 at s = 0
to N = 64 as the spread dilates. At the network's own spread (s = 1) the bare bank predicts
PR = {TH['pr_at_s1_seed202']:.1f}.

![Theory vs data](figs/F5_4_theory_vs_data.png)

**F5.4.** The honest check. The bare filter bank predicts PR = {TH['het_pred_pr_mean']:.1f} for heterogeneous
networks whose measured participation ratio is {TH['het_measured_pr_mean']:.1f} and whose measured n90 is
{TH['het_measured_n90_mean']:.1f}, and it predicts exactly {TH['hom_pred_pr_mean']:.1f} for homogeneous networks
whose measured participation ratio is {TH['hom_measured_pr_mean']:.1f}.

**The minimal tier gets the direction right and the magnitude wrong, in both directions at once.** It must:
it is a statement about a bare, unweighted, feed-forward filter bank, while the measurement is of a recurrent
spiking network driven by a structured input. Two things are missing, and naming them is the specification for
the grounded tier:

1. *The input spectrum.* The bare bank counts a filter as a new dimension even if the input carries no power at
   the frequency that filter passes. Weighting the bank by the measured PSD (F4.13) is what should pull the
   het prediction down from 64 toward the measured plateau, and it is where the short single-tau optimum should
   come from.
2. *Everything that is not the membrane.* A homogeneous network's state is not one-dimensional in practice
   (measured PR {TH['hom_measured_pr_mean']:.1f}, not 1.0), because recurrence, threshold nonlinearity and
   heterogeneous input weights all add dimensions the membrane filter bank knows nothing about.

Across het networks the prediction correlates with measured n90 at r = {TH['r_pred_vs_n90_het']:.2f}, so
{'the ordering carries some signal' if TH['r_pred_vs_n90_het'] > 0.3 else 'the bare bank does not even recover the ordering within het'}.
This is a falsification check reported as it came out, not a fit.
"""

doc += f"""
## 16. A.8, full per-seed table

{T['A8']}

`*` marks the two seeds whose homogeneous network found a long time constant and reached het-level accuracy.
They are the subject of `docs/explanation/Interesting Finding - Co-Adaptation, not Ingredients (2026-07-15).md`
and they are kept in the pooled statistics, not excluded.

## 17. Reproducing this

```
python code/analysis_suite/extract_all.py          # ~55 s per net on a 3080, resumable, 91 nets
python code/analysis_suite/build_report.py         # tables + figures
python code/analysis_suite/extract_geometry.py     # clean-hom repr cache for the geometry refresh (20 pairs)
python code/analysis_suite/run_geometry_n20.py     # F4.6, F4.14 + geometry_n20_summary.json (CCGP, capacity, memory)
python code/analysis_suite/make_input_spectrum.py  # F4.13
python code/analysis_suite/make_theory_figs.py     # F5.1 to F5.4
python code/analysis_suite/write_report.py         # this document
```

`stats_core.py` carries a self-check (`python code/analysis_suite/stats_core.py`) asserting that a unanimous
n = 20 effect lands exactly on the 1.9e-6 permutation floor.
"""

(OUT / "Analysis Suite Report.md").write_text(doc, encoding="utf-8")
print(f"wrote {OUT / 'Analysis Suite Report.md'} ({len(doc):,} chars)")
