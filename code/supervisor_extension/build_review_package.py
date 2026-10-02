"""Assemble a self-contained review package: data, code, results, figures and a runnable notebook.

Mirrors the structure of the package this work replies to, so the two sit side by side. The one place it
departs is the data: that package shipped two spike caches and said the missing ones were the first
outstanding item, so this one ships all forty.

Forty raw caches are 1.3 GB, which is not sendable. They are binary, so `np.packbits` along the time axis
gives an exact eightfold reduction before compression and about 24x after, putting the set at roughly 55 MB.
The round trip is verified per file at write time rather than assumed, and `load_spikes` in the package
reverses it.

Run: python code/supervisor_extension/build_review_package.py [--out DIR] [--no-data]
"""
import argparse
import hashlib
import re
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC

REPO = SC.REPO
DEFAULT_OUT = REPO.parent / "Cross-Seed Replication Package"

CODE_FILES = [
    ("code/supervisor_extension", ["sup_common.py", "build_missing_caches.py", "run_modularity_n20.py",
                                   "run_rsi_n20.py", "run_temporal_lockin_n20.py",
                                   "run_division_of_labour_n20.py", "run_structure_matrices_n20.py",
                                   "run_block_concentration_n20.py", "run_block_k_sweep.py",
                                   "make_figures.py", "make_structure_figures.py",
                                   "make_across_seed_figures.py", "make_aggregate_block_figure.py",
                                   "make_tables.py", "make_report.py", "rebuild_meeting_tables.py",
                                   "audit_fidelity.py", "audit_numerics.py", "audit_meeting_doc.py"]),
    ("code/analysis_suite", ["stats_core.py", "style.py", "redraw_paired_figs.py", "run_shattering_n20.py",
                             "run_vc_shattering_n20.py", "run_shattering_vs_rank_n20.py",
                             "run_capacity_corr_n20.py", "run_capacity_corr_task_n20.py",
                             "run_signal_dimension_n20.py", "run_temporal_generalisation_n20.py",
                             "mftma_capacity.py", "mftma_capacity_corr.py"]),
]

RESULT_JSON = [
    ("data_cache/supervisor_extension", ["modularity_n20.json", "rsi_n20.json", "temporal_lockin_n20.json",
                                         "division_of_labour_n20.json", "block_concentration_n20.json",
                                         "block_k_sweep.json", "structure_scalars_n20.json",
                                         "audit_fidelity.json"]),
    ("docs/analysis_suite", ["shattering_n20_summary.json", "vc_shattering_n20_summary.json",
                             "shattering_vs_rank_n20_summary.json", "capacity_corr_n20_summary.json",
                             "signal_dimension_n20_summary.json",
                             "temporal_generalisation_n20_summary.json", "summary.json"]),
]

# The concise hand-edited report is what gets sent; the full 14-section version stays in docs/ for our own
# use. Pass --full-report to ship that one instead.
REPORT_CONCISE = "docs/REPORT - Structure and Replication (concise, sendable).md"
REPORT_FULL = "docs/MEETING - Structure and Replication (2026-07-30).md"
NARRATIVE = "docs/THESIS NARRATIVE - Separability.md"

DOCS = [
    ("docs/supervisor_extension/significance_tables.md", "significance_tables.md"),
    ("docs/supervisor_extension/replication_evidence.md", "replication_evidence.md"),
    ("docs/supervisor_extension/METHOD_NOTES.md", "METHOD_NOTES.md"),
]

NO_DATA_NOTE = """
## The spike caches are supplied separately

This copy carries no `spike_caches/` directory: the forty binarised spike matrices are about 46 MB packed
and travel by shared link rather than by email. Nothing in the write-up depends on having them.

The notebook ships **already executed**, so every number and figure in it is readable as it stands. Of its
cells, only the three that demonstrate a method live on one pair of networks read the caches; the cross-seed
statistics are recomputed from `results/*.json`, which is included here. Drop the `spike_caches/` folder from
the link beside this README and the whole notebook re-runs end to end.
"""



README = """# Neural heterogeneity promotes robust learning by increasing representational separability

Twenty paired seeds, forty networks. Each pair shares initial weights, data order and training schedule; the
only free difference is the hidden membrane time constant.

## Where to start

**`1. NARRATIVE - Separability.md`** is the argument, in ten figures: heterogeneous networks perform better,
they learn a higher-dimensional representation and actually use it, and the reason is that the representation
is more separable. It ends with a table of everything held in reserve and where to find it.

**`2. SUPPORTING - Structure and Replication.md`** is the cross-seed replication of the information analyses
from the O-Info Review Package, which ran on seed 1001 alone. It is supporting material for the narrative's
third result, not a second argument.

**`Cross-Seed Replication.ipynb`** ships already executed, so every number in it reads without re-running.
Each method is demonstrated live on one pair of networks; the twenty-seed statistics are recomputed in the
notebook from `results/*.json` rather than read from a stored summary.

## Contents

| | |
|---|---|
| `1. NARRATIVE - Separability.md` | the argument, ten figures |
| `2. SUPPORTING - Structure and Replication.md` | the cross-seed replication |
| `Cross-Seed Replication.ipynb` | runnable review notebook, pre-executed |
| `significance_tables.md` | every comparison run, with paired t, Wilcoxon and exact sign-flip |
| `replication_evidence.md` | claim-by-claim verdicts against the original package |
| `METHOD_NOTES.md` | where this departs from the original package, and why |
| `spike_caches/` | all forty binarised spike matrices, bit-packed, plus MANIFEST.json |
| `code/` | every script, including the audits |
| `results/` | per-network result values, so any statistic can be re-derived |
| `figures/` | only the figures the two documents reference |

## Reading the statistics

Three tests are always reported together, never one: paired t, Wilcoxon, and an exact sign-flip permutation
over all 2^20 sign assignments. Both non-parametric tests floor at 2/2^20 = 1.9e-6, which is what a unanimous
20/20 reads as; a p at that floor is a maximally significant result, not a weak one.

The unit of analysis is the network. The original package tests pair-level results with a Mann-Whitney over
unit pairs; pairs inside a network are not independent draws, so that p describes variation among pairs rather
than between architectures. Here each network contributes one number and the twenty within-seed differences
are the unit.

## Reading the figures

Bars are means over the twenty networks with a **95% CI on the mean** as the whisker, faint dots are
individual networks, and each bar group carries its own permutation p and the number of seeds moving with the
mean. Swept parameters are drawn as lines with a shaded 95% CI band.

## Requirements

`numpy scipy scikit-learn networkx matplotlib`. The notebook needs nothing else; the full pipeline
additionally uses `joblib`, `h5py`, `syntropy` and `mftma`.
"""


def pack_caches(out_dir):
    """Bit-pack every cache, verifying the round trip on each before it is written."""
    dst = out_dir / "spike_caches"
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    manifest = {}
    for seed in SC.SEEDS:
        for arch in ("het", "hom"):
            src = SC.cache_path(seed, arch)
            a = np.load(src)
            pk = np.packbits(a, axis=1)
            back = np.unpackbits(pk, axis=1)[:, :a.shape[1]]
            assert np.array_equal(a, back), f"packbits round trip failed for {src.name}"
            p = dst / f"4class_seed{seed}_{arch}_spk.npz"
            np.savez_compressed(p, packed=pk, shape=np.array(a.shape))
            sz = p.stat().st_size
            total += sz
            manifest[f"{seed}_{arch}"] = {"file": p.name, "shape": list(a.shape),
                                          "bytes": sz, "active_bin_fraction": float(a.mean()),
                                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
            print(f"  {p.name:38s} {sz/1e6:5.2f} MB", flush=True)
    # the notebook needs the labels without the SHD file, so they travel with the caches
    import h5py
    with h5py.File(SC.SHD_TEST, "r") as f:
        manifest["_shd_labels"] = f["labels"][:SC.N_TRIALS].astype(int).tolist()
    manifest["_note"] = ("bit-packed along the time axis; unpack with "
                         "np.unpackbits(packed, axis=1)[:, :shape[1]]. _shd_labels are the raw 0-19 SHD "
                         "labels for the first 2048 test trials, in file order, matching the cache.")
    json.dump(manifest, open(dst / "MANIFEST.json", "w"), indent=1)
    print(f"\n  forty caches: {total/1e6:.0f} MB packed")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-data", action="store_true", help="skip the spike caches")
    ap.add_argument("--all-figures", action="store_true",
                    help="ship every figure in the repository, not only the referenced ones")
    ap.add_argument("--full-report", action="store_true",
                    help="ship the full 14-section report instead of the concise sendable one")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()

    print("code")
    for rel, files in CODE_FILES:
        d = out / rel
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = REPO / rel / f
            if src.exists():
                shutil.copy2(src, d / f)
            else:
                print(f"  missing, skipped: {rel}/{f}")
    print(f"  {sum(len(f) for _, f in CODE_FILES)} files")

    print("\nresults")
    for rel, files in RESULT_JSON:
        d = out / "results"
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = REPO / rel / f
            if src.exists():
                shutil.copy2(src, d / f)
            else:
                print(f"  missing, skipped: {rel}/{f}")
    print(f"  {len(list((out / 'results').glob('*.json')))} result files")

    print("\nfigures")
    fd = out / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    available = {p.name: p for sub in ("supervisor_extension/figs", "analysis_suite/figs")
                 for p in (REPO / "docs" / sub).glob("*.png")}
    if args.all_figures:
        wanted = set(available)
    else:
        # only what the shipped documents actually point at, plus anything they name in prose. Shipping
        # every figure in the repository was the single loudest complaint about the previous package.
        wanted = set()
        for src_rel in [REPORT_FULL if args.full_report else REPORT_CONCISE, NARRATIVE] + \
                       [d[0] for d in DOCS]:
            src = REPO / src_rel
            if not src.exists():
                continue
            txt = src.read_text(encoding="utf-8")
            wanted |= {Path(f).name for f in re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", txt)}
            wanted |= set(re.findall(r"`([A-Za-z0-9_]+\.png)`", txt))
            # reserve tables name figures by tag only, e.g. `F4_25` or `S1b`
            for tag in re.findall(r"`((?:F4|F5|T4|S)\d+[A-Za-z0-9_]*)`", txt):
                wanted |= {n for n in available if n.startswith(tag + "_") or n == tag + ".png"}
    missing = sorted(w for w in wanted if w not in available)
    for name in sorted(wanted & set(available)):
        shutil.copy2(available[name], fd / name)
    print(f"  {len(list(fd.glob('*.png')))} figures kept of {len(available)} available")
    if missing:
        print(f"  referenced but not found: {missing}")

    print("\ndocuments")
    for src_rel, dst_name in DOCS:
        src = REPO / src_rel
        if src.exists():
            shutil.copy2(src, out / dst_name)
    nsrc = REPO / NARRATIVE
    if nsrc.exists():
        ntxt = nsrc.read_text(encoding="utf-8")
        for old in ("supervisor_extension/figs/", "analysis_suite/figs/"):
            ntxt = ntxt.replace("](" + old, "](figures/")
        (out / "1. NARRATIVE - Separability.md").write_text(ntxt, encoding="utf-8")
        print("  narrative: 1. NARRATIVE - Separability.md")
    report_src = REPO / (REPORT_FULL if args.full_report else REPORT_CONCISE)
    rp = out / "2. SUPPORTING - Structure and Replication.md"
    shutil.copy2(report_src, rp)
    print(f"  report: {report_src.name}")
    # the repo copy references figures by a docs-relative path; inside the package they are all in figures/
    txt = rp.read_text(encoding="utf-8")
    for old in ("supervisor_extension/figs/", "analysis_suite/figs/"):
        txt = txt.replace("](" + old, "](figures/")
    readme = README
    if args.no_data:
        # neither document may promise a directory this copy does not carry
        txt = txt.replace(
            "- Spike caches for all forty networks: `spike_caches/` (bit-packed, with `MANIFEST.json`)",
            "- Spike caches for all forty networks: supplied separately by shared link (bit-packed, "
            "with `MANIFEST.json`)")
        readme = (README.replace(
            "| `spike_caches/` | all forty binarised spike matrices, bit-packed, plus MANIFEST.json |\n", "")
            + NO_DATA_NOTE)
    rp.write_text(txt, encoding="utf-8")
    (out / "README.md").write_text(readme, encoding="utf-8")
    # the notebook is regenerated so it always matches the shipped scripts
    import subprocess
    subprocess.run([sys.executable, str(HERE / "build_review_notebook.py"),
                    "--out", str(out / "Cross-Seed Replication.ipynb")], check=True)
    print(f"  {len(DOCS)} documents plus README, figure paths rewritten to figures/")

    if not args.no_data:
        print("\nspike caches (bit-packed, round trip verified per file)")
        pack_caches(out)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\npackage at {out}")
    print(f"total {size/1e6:.0f} MB, built in {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    main()
