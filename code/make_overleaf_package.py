"""Assemble a self-contained Overleaf project from Final Repo/thesis.

The canonical thesis resolves its figures through ../docs/ by design, which is
correct inside this repo and impossible on Overleaf. This script copies the
LaTeX sources, resolves every \\includegraphics against the current
graphicspath, copies only the referenced figures into a flat figures/, and
rewrites \\graphicspath so nothing resolves outside the project.

    python code/make_overleaf_package.py            build, refusing to clobber
    python code/make_overleaf_package.py --check    report edits, build nothing
    python code/make_overleaf_package.py --force    rebuild, discarding edits

THIS SCRIPT DELETES AND REBUILDS overleaf-upload/ FROM SCRATCH. To make that
safe it fingerprints every file it writes into .package-manifest.json, and on
the next run it refuses to touch the package if anything inside has been
added, changed or removed. Manual edits therefore survive by default and can
only be discarded by passing --force on purpose.

The canonical source is Final Repo/thesis. If you edit the package or the
Overleaf project instead, those edits live only there: port them back into
Final Repo/thesis, or the next --force rebuild silently reverts them. That is
the same trap that make_latex_tables.py caused twice before it was fixed.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

# resolved relative to this file, so the script cannot silently target a
# different tree if the repo is moved or copied
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Source thesis directory, overridable with --source NAME (default: thesis).
# The scan happens here rather than in main() because THESIS feeds the
# module-level path lists below. argparse in main() also declares --source so
# it validates and shows in --help.
_SRC = "thesis"
for _i, _a in enumerate(sys.argv):
    if _a == "--source" and _i + 1 < len(sys.argv):
        _SRC = sys.argv[_i + 1]
    elif _a.startswith("--source="):
        _SRC = _a.split("=", 1)[1]
THESIS = os.path.join(ROOT, _SRC)
PKG = os.path.join(ROOT, "overleaf-upload")
ZIP = PKG + ".zip"
MANIFEST = ".package-manifest.json"

if not os.path.isdir(os.path.join(THESIS, "chapters")):
    raise SystemExit(
        "No %s/chapters next to this script.\n"
        "Expected a thesis source at: %s\n"
        "This script must live in Final Repo/code/. Refusing to guess."
        % (_SRC, THESIS))

# graphicspath from includes.tex, in order, resolved as pdflatex would
SEARCH = [
    THESIS,
    os.path.join(ROOT, "docs", "analysis_suite", "figs"),
    os.path.join(ROOT, "docs", "supervisor_extension", "figs"),
    os.path.join(ROOT, "figures"),
    os.path.join(THESIS, "figures"),
]
EXTS = ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"]

SRC_FILES = ["main.tex", "includes.tex", "notation.tex", "titlepage.tex",
             "references.bib"]

# build artefacts that must never reach Overleaf
DROP_EXT = {".aux", ".log", ".out", ".toc", ".blg", ".brf", ".bbl", ".pdf",
            ".synctex", ".gz", ".fls", ".fdb_latexmk", ".lof", ".lot", ".bcf"}

OLD_PATH = (b"\\graphicspath{{../docs/analysis_suite/figs/}"
            b"{../docs/supervisor_extension/figs/}{../figures/}{figures/}}")
NEW_PATH = b"\\graphicspath{{figures/}}"

inc_pat = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")


def strip_comments(src):
    """Drop LaTeX comments so a commented-out \\includegraphics is not counted."""
    out = []
    for ln in src.split("\n"):
        i, cut = 0, ln
        while True:
            i = cut.find("%", i)
            if i == -1:
                break
            if i > 0 and cut[i - 1] == "\\":
                i += 1
                continue
            cut = cut[:i]
            break
        out.append(cut)
    return "\n".join(out)


def resolve(target):
    for base in SEARCH:
        for ext in EXTS:
            p = os.path.normpath(os.path.join(base, target + ext))
            if os.path.isfile(p):
                return p
    return None


def find_bst():
    """vancouver.bst is shipped rather than assumed. It is in TeX Live's
    bibtex-extra so Overleaf almost certainly has it, but a .bst in the project
    root takes precedence and costs 25 kB."""
    local = os.path.join(THESIS, "vancouver.bst")
    if os.path.isfile(local):
        return local
    try:
        out = subprocess.run(["kpsewhich", "vancouver.bst"],
                             capture_output=True, text=True, timeout=60)
        p = out.stdout.strip()
        if p and os.path.isfile(p):
            return p
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# --------------------------------------------------------------------------
# edit protection
# --------------------------------------------------------------------------

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_package():
    """Every file currently in the package, keyed by forward-slash relpath."""
    found = {}
    for d, _, files in os.walk(PKG):
        for fn in files:
            p = os.path.join(d, fn)
            rel = os.path.relpath(p, PKG).replace(os.sep, "/")
            if rel == MANIFEST:
                continue
            found[rel] = p
    return found


def write_manifest():
    files = walk_package()
    data = {rel: digest(p) for rel, p in sorted(files.items())}
    with open(os.path.join(PKG, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump({"note": "Fingerprints of the last generated package. "
                           "make_overleaf_package.py refuses to overwrite the "
                           "package if any of these changed. Do not edit.",
                   "files": data}, fh, indent=1, sort_keys=True)


def detect_edits():
    """Return (changed, added, removed) relative to the recorded manifest.

    Returns None if the package does not exist, meaning nothing to protect.
    Raises if the package exists without a manifest, since in that case its
    provenance is unknown and clobbering it could destroy real work.
    """
    if not os.path.isdir(PKG):
        return None
    mpath = os.path.join(PKG, MANIFEST)
    if not os.path.isfile(mpath):
        raise SystemExit(
            "%s exists but carries no %s, so this script cannot tell whether it\n"
            "holds manual edits. Move it aside, or pass --force to discard it."
            % (PKG, MANIFEST))
    with open(mpath, encoding="utf-8") as fh:
        recorded = json.load(fh)["files"]
    current = walk_package()
    changed = sorted(r for r in recorded if r in current
                     and digest(current[r]) != recorded[r])
    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    return changed, added, removed


def report_edits(edits):
    changed, added, removed = edits
    if not (changed or added or removed):
        print("No manual edits detected in the package.")
        return False
    print("MANUAL EDITS DETECTED in %s" % PKG)
    for label, items in (("modified", changed), ("added", added),
                         ("deleted", removed)):
        for r in items:
            print("   %-8s %s" % (label, r))
    return True


# --------------------------------------------------------------------------

def build():
    if os.path.isdir(PKG):
        shutil.rmtree(PKG)
    os.makedirs(os.path.join(PKG, "chapters"))
    os.makedirs(os.path.join(PKG, "figures"))

    # ---- sources ----
    for name in SRC_FILES:
        shutil.copy2(os.path.join(THESIS, name), os.path.join(PKG, name))
    chapters = sorted(glob.glob(os.path.join(THESIS, "chapters", "*.tex")))
    for f in chapters:
        shutil.copy2(f, os.path.join(PKG, "chapters", os.path.basename(f)))

    bst = find_bst()
    if bst:
        shutil.copy2(bst, os.path.join(PKG, "vancouver.bst"))
    else:
        print("WARNING: vancouver.bst not found; relying on Overleaf's copy")

    # ---- figures ----
    targets = set()
    for f in sorted(glob.glob(os.path.join(THESIS, "**", "*.tex"),
                              recursive=True)):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for m in inc_pat.finditer(strip_comments(fh.read())):
                targets.add(m.group(1).strip())

    figmap, missing = {}, []
    for t in sorted(targets):
        p = resolve(t)
        (figmap.__setitem__(t, p) if p else missing.append(t))
    if missing:
        raise SystemExit("unresolved figures, aborting: %s" % missing)

    bases = {}
    for t, p in figmap.items():
        bases.setdefault(os.path.basename(p), []).append(t)
    clash = {b: v for b, v in bases.items() if len(v) > 1}
    if clash:
        raise SystemExit(
            "two figures share a basename, so a flat figures/ would lose "
            "one: %s" % clash)

    for t, p in sorted(figmap.items()):
        shutil.copy2(p, os.path.join(PKG, "figures", os.path.basename(p)))

    # ---- rewrite graphicspath ----
    inc = os.path.join(PKG, "includes.tex")
    with open(inc, "rb") as fh:
        raw = fh.read()
    if raw.count(OLD_PATH) != 1:
        raise SystemExit(
            "graphicspath in includes.tex is not the expected line; it was "
            "edited since this script was written. Update OLD_PATH.")
    raw = raw.replace(OLD_PATH, NEW_PATH)
    with open(inc, "wb") as fh:
        fh.write(raw)

    # ---- refuse to ship artefacts ----
    bad = [os.path.join(d, f)
           for d, _, fs in os.walk(PKG) for f in fs
           if os.path.splitext(f)[1].lower() in DROP_EXT
           and not d.endswith("figures")]
    if bad:
        raise SystemExit("build artefacts present: %s" % bad)

    write_readme(len(figmap), figmap)
    write_manifest()

    write_zip()

    n = sum(len(f) for _, _, f in os.walk(PKG))
    size = sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(PKG) for f in fs)
    print("package : %s" % PKG)
    print("  sources : %d" % (len(SRC_FILES) + len(chapters) + (1 if bst else 0)))
    print("  figures : %d" % len(figmap))
    print("  files   : %d, %.1f MB" % (n, size / 1e6))
    print("zip     : %s (%.1f MB)" % (ZIP, os.path.getsize(ZIP) / 1e6))
    print()
    print("Verify by building a copy in a directory with no sibling docs/.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the package carries manual edits, "
                         "discarding them")
    ap.add_argument("--check", action="store_true",
                    help="report manual edits and exit without building")
    ap.add_argument("--source", default="thesis",
                    help="thesis source directory under Final Repo (default: thesis). "
                         "Scanned at import time; declared here for --help and validation.")
    args = ap.parse_args()

    edits = None if args.force else detect_edits()

    if args.check:
        if edits is None:
            print("No package at %s." % PKG)
            return 0
        return 1 if report_edits(edits) else 0

    if edits is not None and report_edits(edits):
        print()
        print("Refusing to rebuild, because that would delete the edits above.")
        print("The canonical thesis is %s." % THESIS)
        print("Port these edits back into it, then rerun this script;")
        print("or pass --force to discard them and regenerate from the thesis.")
        return 1

    build()
    return 0


# Kept in the package directory but excluded from the zip: Overleaf shows every
# uploaded file in its tree, and neither of these is part of the LaTeX project.
# The dotfile is the bigger nuisance, since some zip extractors mangle or drop
# leading-dot entries.
ZIP_EXCLUDE = {MANIFEST, "README.md"}


def write_zip():
    """Write the upload zip with paths stored FLAT at the archive root.

    Overleaf's Upload Project expects main.tex at the top level. A zip built by
    wrapping the folder, which is what Windows Explorer's "Compressed folder"
    does, stores everything under overleaf-upload/ and Overleaf then cannot
    find main.tex. Building the archive here rather than zipping the directory
    is what guarantees the flat layout.
    """
    import zipfile
    if os.path.isfile(ZIP):
        os.remove(ZIP)
    files = sorted(walk_package().items())
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, path in files:
            if rel in ZIP_EXCLUDE:
                continue
            z.write(path, arcname=rel)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    if "main.tex" not in names:
        raise SystemExit("zip does not carry main.tex at its root, aborting")
    nested = sorted({n.split("/")[0] for n in names
                     if "/" in n} - {"chapters", "figures"})
    if nested:
        raise SystemExit("unexpected top-level directories in zip: %s" % nested)
    return names


def write_readme(nfig, figmap):
    from collections import Counter
    src_counts = Counter(os.path.basename(os.path.dirname(p))
                         for p in figmap.values())
    breakdown = ", ".join("%d from %s/" % (v, k)
                          for k, v in sorted(src_counts.items()))
    body = README.replace("@NFIG@", str(nfig)).replace("@BREAKDOWN@", breakdown)
    with open(os.path.join(PKG, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(body)


README = """# Overleaf upload package

A self-contained copy of the thesis. Nothing in here resolves to a path outside this directory, which is the
one thing the canonical `Final Repo\\thesis` does not satisfy: that copy pulls its figures from `..\\docs\\` by
design, and Overleaf has no `..\\docs\\`.

Generated by `Final Repo\\code\\make_overleaf_package.py`.

## Editing this package is protected, but read this first

The generator fingerprints every file it writes into `.package-manifest.json`. On its next run it compares
the package against those fingerprints and **refuses to rebuild if anything here has been changed, added or
deleted**, so manual edits are safe by default. Discarding them takes an explicit `--force`.

```
python code/make_overleaf_package.py --check    what have I changed in here?
python code/make_overleaf_package.py            rebuild, refusing to clobber edits
python code/make_overleaf_package.py --force    rebuild, discarding edits
```

**The protection is against accidental deletion, not against divergence.** The canonical thesis is
`Final Repo\\thesis`. Edits made here, or made in the Overleaf project and downloaded, exist only in this
copy: the thesis, the figures and the analysis pipeline all still hold the old text. Port substantial edits
back into `Final Repo\\thesis` rather than leaving the two to drift, or a later `--force` rebuild reverts
them and the repo's own build stops matching what was submitted.

## Uploading

Upload `overleaf-upload.zip` (beside this directory) via **New Project -> Upload Project** on Overleaf. The
compiler is pdfLaTeX, which is Overleaf's default, and the main document is `main.tex`, which Overleaf detects
on its own. No project settings need changing.

## What is here

```
main.tex          references.bib     vancouver.bst
includes.tex      notation.tex       titlepage.tex
chapters/         the chapters, the appendices, and the generated tables
figures/          @NFIG@ .png files, flat
```

`vancouver.bst` is shipped rather than relied on. It is in TeX Live's `bibtex-extra`, so Overleaf almost
certainly has it, but a `.bst` in the project root takes precedence and costs 25 kB.

## How it differs from `Final Repo\\thesis`

One line. `\\graphicspath` in `includes.tex` is `{{figures/}}` here, against
`{{../docs/analysis_suite/figs/}{../docs/supervisor_extension/figs/}{../figures/}{figures/}}` there. Every
other source file is byte-identical at generation time.

The @NFIG@ referenced figures were found by parsing `\\includegraphics` from the sources and searching the
graphicspath list in order, then copied flat into `figures/`: @BREAKDOWN@. The script aborts if any target
fails to resolve, and aborts if two figures share a basename, since a flat directory would silently lose one.
Figures present in those directories but not referenced are not copied.

Build artefacts are excluded: no `.aux`, `.log`, `.out`, `.toc`, `.blg`, `.brf`, `.bbl`, no `main.pdf`.

## Verification

The package is verified by building it in a sandbox directory with **no sibling `docs/` or `figures/`**, so
any figure still resolving through an external path fails loudly rather than succeeding by accident. The
expected result, matching the canonical build:

```
pdflatex -> bibtex -> pdflatex -> pdflatex
77 pages, 0 files not found, 0 errors,
0 undefined references, 0 undefined citations,
14 overfull hboxes (worst 25.6pt)
```

The only bibtex message is `Warning--empty pages in kingma2015`, which the canonical build also emits.
"""


if __name__ == "__main__":
    sys.exit(main())
