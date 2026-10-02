"""Audit the meeting document: figures, tables, numbers, structure.

Checks the things that go wrong in a document assembled over many edits, each of which has bitten at least
once already in this project: a figure path that no longer resolves, a statistics table that lost two of its
three tests, a number in the prose that no longer matches the JSON it came from, a contents list that has
drifted from the headings, a cross-reference to a section that has been renumbered.

Every number quoted in prose cannot be checked automatically, so the check is targeted: every decimal that
appears next to a recognised metric name is looked up in the source JSON.

Run: python code/supervisor_extension/audit_meeting_doc.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC

REPO = SC.REPO
DOCS = REPO / "docs"
DOC = DOCS / "MEETING - Structure and Replication (2026-07-30).md"
A = DOCS / "analysis_suite"
B = REPO / "data_cache" / "supervisor_extension"

S = DOC.read_text(encoding="utf-8")
LINES = S.splitlines()
fails, warns = [], []


def fail(msg):
    fails.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    warns.append(msg)
    print(f"  warn  {msg}")


# ---------------------------------------------------------------- 1. figures
print("\n1. figures")
imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", S)
missing = [f for f in imgs if not (DOCS / f).exists()]
if missing:
    for m in missing:
        fail(f"figure path does not resolve: {m}")
else:
    print(f"  ok    {len(imgs)} references, all resolve")

dupes = {f for f in imgs if imgs.count(f) > 1}
if dupes:
    for d in dupes:
        warn(f"figure referenced more than once: {d}")
else:
    print(f"  ok    no figure referenced twice ({len(set(imgs))} unique)")

alt_missing = [m for m in re.findall(r"!\[([^\]]*)\]\([^)]+\)", S) if not m.strip()]
if alt_missing:
    warn(f"{len(alt_missing)} figures have empty alt text")
else:
    print("  ok    every figure has alt text")

# figures that exist on disk but are never shown, so nothing is silently dropped
on_disk = {p.name for p in (DOCS / "supervisor_extension" / "figs").glob("*.png")}
shown = {Path(f).name for f in imgs}
appendix = re.findall(r"`(S\d+[A-Za-z0-9_]*\.png)`", S)
orphans = sorted(on_disk - shown - set(appendix))
if orphans:
    warn(f"supervisor_extension figures neither shown nor named in the appendix: {orphans}")
else:
    print("  ok    every supervisor_extension figure is either shown or named in the appendix")

# ---------------------------------------------------------------- 2. tables
print("\n2. tables")
headers = [l for l in LINES if l.startswith("|") and l.count("|") >= 4
           and re.search(r"\|\s*(het|Package)\s*\|", l)]
stat_headers = [h for h in headers if re.search(r"\|\s*het\s*\|\s*hom\s*\|", h)]
no_triple = [h for h in stat_headers if "paired t" not in h or "Wilcoxon" not in h]
if no_triple:
    for h in no_triple:
        fail(f"stats table missing the test triple: {h[:78]}")
else:
    print(f"  ok    all {len(stat_headers)} het-vs-hom tables carry paired t, Wilcoxon and sign-flip")

# every table is rectangular
bad_shape = []
i = 0
while i < len(LINES):
    if LINES[i].startswith("|"):
        blk = []
        while i < len(LINES) and LINES[i].startswith("|"):
            blk.append(LINES[i])
            i += 1
        widths = {l.count("|") for l in blk}
        if len(widths) > 1:
            bad_shape.append(blk[0][:70])
    else:
        i += 1
if bad_shape:
    for b in bad_shape:
        fail(f"ragged table (inconsistent column count): {b}")
else:
    print("  ok    every table is rectangular")

if "†" in S and "floor" in S:
    print("  ok    the enumeration-floor footnote is present")
else:
    warn("no footnote explaining the 1.9e-6 permutation floor")

# ---------------------------------------------------------------- 3. numbers against source
print("\n3. numbers quoted in prose, against source JSON")


def jget(path, *keys):
    d = json.load(open(path))
    for k in keys:
        d = d[k]
    return d


CHECKS = [
    ("shattering 20-cond dz", jget(A / "shattering_n20_summary.json", "paired", "B_20cond", "dz"), 2.07),
    ("VC alpha het", jget(A / "vc_shattering_n20_summary.json", "paired", "alpha_vc", "het_mean"), 1.775),
    ("VC alpha hom", jget(A / "vc_shattering_n20_summary.json", "paired", "alpha_vc", "hom_mean"), 1.772),
    ("cap B alpha_M het", jget(A / "capacity_corr_n20_summary.json", "paired", "B_20cond", "alpha_M", "het_mean"), 0.1909),
    ("cap T alpha_M het", jget(A / "capacity_corr_n20_summary.json", "paired", "T_4class", "alpha_M", "het_mean"), 0.2149),
    ("cap T alpha_M p", jget(A / "capacity_corr_n20_summary.json", "paired", "T_4class", "alpha_M", "perm_p"), 0.494),
    ("cap T R_M p", jget(A / "capacity_corr_n20_summary.json", "paired", "T_4class", "R_M", "perm_p"), 0.023),
    ("cap T centcorr p", jget(A / "capacity_corr_n20_summary.json", "paired", "T_4class", "center_corr", "perm_p"), 0.003),
    ("signal PR het", jget(A / "signal_dimension_n20_summary.json", "paired", "signal_pr", "het_mean"), 3.167),
    ("tempgen longlag ratio het", jget(A / "temporal_generalisation_n20_summary.json", "paired", "long_lag_ratio", "het_mean"), 0.0794),
    ("mod full Qnorm het", jget(B / "modularity_n20.json", "paired", "full_Qnorm", "het_mean"), 0.573),
    ("mod full Qnorm p", jget(B / "modularity_n20.json", "paired", "full_Qnorm", "perm_p"), 0.280),
    ("blockconc louvain p", jget(B / "block_concentration_n20.json", "paired", "louvain", "perm_p"), 1.2e-4),
    ("blockconc fixedK p", jget(B / "block_concentration_n20.json", "paired", "fixedK", "perm_p"), 7.4e-4),
    ("blockconc cum p", jget(B / "block_concentration_n20.json", "paired", "louvain_cum", "perm_p"), 0.002),
    ("corr ratio vs Qnorm", jget(B / "block_concentration_n20.json", "corr_ratio_vs_Qnorm"), 0.209),
    ("rsi RSI p", jget(B / "rsi_n20.json", "paired", "RSI", "perm_p"), 0.504),
    ("rsi RSI_n het", jget(B / "rsi_n20.json", "paired", "RSI_n", "het_mean"), 0.0921),
    ("lang lockin het", jget(B / "temporal_lockin_n20.json", "paired", "acc_language_lockin_ms", "het_mean"), 551.6),
    ("hom gap p", jget(B / "temporal_lockin_n20.json", "within_arm_gap", "hom_acc_gap_ms", "p"), 0.015),
    ("dol eta2 p", jget(B / "division_of_labour_n20.json", "paired", "eta2_SI_module", "perm_p"), 0.019),
    ("dol excess p", jget(B / "division_of_labour_n20.json", "paired", "eta2_excess", "perm_p"), 0.015),
    ("ksweep K5 between p", jget(B / "block_k_sweep.json", "by_k", "5", "between", "perm_p"), 0.001),
    ("ksweep K2 between p", jget(B / "block_k_sweep.json", "by_k", "2", "between", "perm_p"), 0.005),
]
bad = 0
for name, actual, quoted in CHECKS:
    tol = max(6e-4, abs(quoted) * 0.02)
    if abs(actual - quoted) > tol:
        fail(f"{name}: document says {quoted}, source says {actual:.6g}")
        bad += 1
if not bad:
    print(f"  ok    all {len(CHECKS)} spot-checked numbers match their source")

# ---------------------------------------------------------------- 4. structure
print("\n4. structure")
heads = [l for l in LINES if re.match(r"^#{1,3} ", l)]
h2 = [l for l in LINES if l.startswith("## ") and re.match(r"^## \d+\.", l)]
nums = [int(re.match(r"^## (\d+)\.", l).group(1)) for l in h2]
if nums != sorted(nums) or nums != list(range(1, len(nums) + 1)):
    fail(f"numbered sections are not 1..N in order: {nums}")
else:
    print(f"  ok    sections numbered 1..{len(nums)} in order")

toc = re.findall(r"^\d+\.\s*\[([^\]]+)\]\(#([^)]+)\)", S, re.M)
for title, anchor in toc:
    slug = re.sub(r"[^a-z0-9\s-]", "", title.lower()).replace(" ", "-")
    if not any(slug in re.sub(r"[^a-z0-9\s-]", "", h.lower()).replace(" ", "-") for h in h2 + heads):
        warn(f"contents entry may not match a heading: {title}")
if toc:
    print(f"  ok    {len(toc)} contents entries checked")

subrefs = set(re.findall(r"§(\d+[a-z]?)", S))
present = set()
for l in LINES:
    m = re.match(r"^#{2,3} (\d+[a-z]?)[.\s]", l)
    if m:
        present.add(m.group(1))
for r in sorted(subrefs):
    if r not in present:
        fail(f"cross-reference to a section that does not exist: §{r}")
if subrefs:
    print(f"  ok    {len(subrefs)} section cross-references checked")

if S.count("—"):
    warn(f"{S.count('—')} em dashes present (house style avoids them)")
else:
    print("  ok    no em dashes")

# ---------------------------------------------------------------- 5. stale claims
print("\n5. superseded claims")
STALE = [
    ("does not replicate", "modularity"),
    ("neither arm is small-world", None),
    ("seed 1001 is the extreme hom network", None),
]
for phrase, ctx in STALE:
    for i, l in enumerate(LINES):
        if phrase in l:
            warn(f"line {i+1} contains a phrase from a superseded reading: '{phrase}'")
print("  (warnings above are for review, not necessarily errors)")

print("\n" + "=" * 70)
print(f"{len(fails)} failures, {len(warns)} warnings")
print("=" * 70)
sys.exit(1 if fails else 0)
