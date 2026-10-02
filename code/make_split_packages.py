"""Two Overleaf packages from thesis-restructure: the main paper and the
appendices, each self-contained and light enough to compile under a free-tier
timeout. The heavy montage figures live only in the appendix package.

Cross-references between the two halves are resolved with xr-hyper reading a
frozen label file extracted from the full build, so Appendix~\\ref{...} in the
main paper and Section~\\ref{...} in the appendix both show the right number
rather than ??. The frozen numbers are correct as of the last full build; they
are exact again once the two halves are reassembled in thesis-restructure.

Run (after a full build of thesis-restructure so main.aux is current):
    python code/make_split_packages.py
"""
import os
import re
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "thesis-restructure")

SEARCH = [
    SRC,
    os.path.join(ROOT, "docs", "analysis_suite", "figs"),
    os.path.join(ROOT, "docs", "supervisor_extension", "figs"),
    os.path.join(ROOT, "figures"),
    os.path.join(SRC, "figures"),
]
EXTS = ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"]
SHARED = ["includes.tex", "notation.tex", "titlepage.tex", "references.bib"]
BST = os.path.join(SRC, "vancouver.bst")

MAIN_CH = ["introduction", "background", "methods", "evaluation", "conclusion"]
APX_CH = ["mechanism", "appendix"]

inc_pat = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
lbl_pat = re.compile(r"\\label\{([^}]*)\}")
OLD_PATH = ("\\graphicspath{{../docs/analysis_suite/figs/}"
            "{../docs/supervisor_extension/figs/}{../figures/}{figures/}}")
NEW_PATH = "\\graphicspath{{figures/}}"


def strip_comments(s):
    out = []
    for ln in s.split("\n"):
        i, c = 0, ln
        while True:
            i = c.find("%", i)
            if i == -1:
                break
            if i > 0 and c[i - 1] == "\\":
                i += 1
                continue
            c = c[:i]
            break
        out.append(c)
    return "\n".join(out)


def resolve(target):
    for base in SEARCH:
        for e in EXTS:
            p = os.path.normpath(os.path.join(base, target + e))
            if os.path.isfile(p):
                return p
    return None


def labels_in(files):
    s = set()
    for f in files:
        s |= set(lbl_pat.findall(open(f, encoding="utf-8", errors="replace").read()))
    return s


def read_main_tex():
    return open(os.path.join(SRC, "main.tex"), encoding="utf-8").read()


def newlabels_for(aux_text, wanted):
    """The \\newlabel lines from the full aux whose label is in `wanted`."""
    out = []
    for m in re.finditer(r"\\newlabel\{([^}]*)\}", aux_text):
        if m.group(1) in wanted:
            line_end = aux_text.find("\n", m.start())
            out.append(aux_text[m.start():line_end if line_end != -1 else None])
    return out


def build(pkg, chapters, other_labels_aux, is_appendix):
    d = os.path.join(ROOT, pkg)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "chapters"))
    os.makedirs(os.path.join(d, "figures"))

    # shared preamble files; includes.tex gets the flat graphicspath
    for name in SHARED:
        src = os.path.join(SRC, name)
        dst = os.path.join(d, name)
        if name == "includes.tex":
            raw = open(src, encoding="utf-8").read()
            if raw.count(OLD_PATH) != 1:
                raise SystemExit("graphicspath not found as expected in includes.tex")
            open(dst, "w", encoding="utf-8").write(raw.replace(OLD_PATH, NEW_PATH))
        else:
            shutil.copy2(src, dst)
    if os.path.isfile(BST):
        shutil.copy2(BST, os.path.join(d, "vancouver.bst"))

    # every chapters/*.tex is copied (tables are \input by chapters); only the
    # listed chapters are compiled via the tailored main.tex
    for f in os.listdir(os.path.join(SRC, "chapters")):
        if f.endswith(".tex"):
            shutil.copy2(os.path.join(SRC, "chapters", f),
                         os.path.join(d, "chapters", f))

    # frozen cross-reference labels for the other half
    with open(os.path.join(d, "otherhalf.aux"), "w", encoding="utf-8") as fh:
        fh.write("\\relax\n" + "\n".join(other_labels_aux) + "\n")

    # tailored main.tex
    main = read_main_tex()
    xr = "\\usepackage{xr-hyper}\n\\externaldocument{otherhalf}\n"
    main = main.replace("\\input{includes}", "\\input{includes}\n" + xr, 1)
    if is_appendix:
        # drop the main body, keep preamble + appendices; suppress the title page
        main = main.replace("\\input{titlepage}", "")
        for ch in MAIN_CH:
            main = main.replace("\\input{chapters/%s}" % ch, "")
    else:
        # drop the appendix inputs and the \appendix switch
        for ch in APX_CH:
            main = main.replace("\\input{chapters/%s}" % ch, "")
        main = main.replace("\\appendix", "")
    open(os.path.join(d, "main.tex"), "w", encoding="utf-8").write(main)

    # figures referenced by the compiled chapters (+ titlepage)
    compiled = [os.path.join(d, "chapters", ch + ".tex") for ch in chapters]
    compiled += [os.path.join(d, "titlepage.tex")]
    # follow one level of \input into tab_*.tex etc.
    targets = set()
    scan = list(compiled)
    for f in compiled:
        s = open(f, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\\input\{chapters/([^}]*)\}", s):
            p = os.path.join(d, "chapters", m.group(1) + ".tex")
            if os.path.isfile(p):
                scan.append(p)
    for f in scan:
        s = strip_comments(open(f, encoding="utf-8", errors="replace").read())
        for m in inc_pat.finditer(s):
            targets.add(m.group(1).strip())

    figmap, missing = {}, []
    for t in sorted(targets):
        p = resolve(t)
        (figmap.__setitem__(t, p) if p else missing.append(t))
    if missing:
        raise SystemExit("%s: unresolved figures %s" % (pkg, missing))
    for t, p in figmap.items():
        shutil.copy2(p, os.path.join(d, "figures", os.path.basename(p)))

    # remove the uncompiled chapter files so the package is clean
    keep = set(chapters)
    # keep table files that the compiled chapters input
    for f in scan:
        keep.add(os.path.splitext(os.path.basename(f))[0])
    for f in os.listdir(os.path.join(d, "chapters")):
        base = os.path.splitext(f)[0]
        if base not in keep:
            os.remove(os.path.join(d, "chapters", f))

    # zip, flat, main.tex at root
    zpath = d + ".zip"
    if os.path.isfile(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, files in os.walk(d):
            for fn in files:
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, d).replace(os.sep, "/"))
    nfig = len(figmap)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(d) for f in fs)
    print("%s: %d figures, %.1f MB, zip %s" % (pkg, nfig, size / 1e6, os.path.basename(zpath)))


def main():
    aux = open(os.path.join(SRC, "main.aux"), encoding="utf-8", errors="replace").read()
    chdir = os.path.join(SRC, "chapters")
    main_files = [os.path.join(chdir, c + ".tex") for c in MAIN_CH]
    apx_files = [os.path.join(chdir, c + ".tex") for c in APX_CH]
    # follow \input inside the appendix chapters (e.g. app_infomethods) so their
    # labels are captured for the main package's cross-references
    for c in APX_CH:
        s = open(os.path.join(chdir, c + ".tex"), encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\\input\{chapters/([^}]*)\}", s):
            p = os.path.join(chdir, m.group(1) + ".tex")
            if os.path.isfile(p):
                apx_files.append(p)
    # include the table files' labels with the main half
    for extra in os.listdir(chdir):
        if extra.startswith("tab_") and extra.endswith(".tex"):
            main_files.append(os.path.join(chdir, extra))
    main_labels = labels_in(main_files)
    apx_labels = labels_in(apx_files)

    build("overleaf-main", MAIN_CH, newlabels_for(aux, apx_labels), is_appendix=False)
    build("overleaf-appendix", APX_CH, newlabels_for(aux, main_labels), is_appendix=True)
    print("\nverify each by building in a directory with no sibling docs/.")


if __name__ == "__main__":
    main()
