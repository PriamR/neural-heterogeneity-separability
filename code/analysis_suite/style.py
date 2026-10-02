"""House style for every figure in the paper. Import this and nothing else for colours, sizes and helpers,
so the whole suite is visually consistent.

Conventions (September 2026 polish pass):
  * Figure text is descriptive only. Titles are short noun phrases in sentence case, one per panel, lettered
    (a), (b), ... on multi-panel figures. The narrative lives in the caption.
  * Axis labels and legend entries start with a capital letter and carry units. No abbreviations (no het/hom).
  * Heterogeneous is red and homogeneous is blue everywhere. The fixed-tau rungs are a grey ramp (dark = short
    constant, light = long) so the two learned networks are the only coloured series on a control figure; the
    scaled-init control is amber and appears only in the appendix.
  * Statistics quoted on a figure use the paired t for the twenty-pair comparisons and the sign-flip
    permutation for the ten-seed controls, matching the prose.
  * Recessive chrome: horizontal hairline grid only, no top/right/left spines, thin marks, white rings on dots,
    white gaps between bars, bands at low opacity.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- palette
HET = "#d1495b"      # heterogeneous (learned tau spread)
HOM = "#3167b3"      # homogeneous (single learned tau)
SCALED = "#e0962f"   # scaled-init homogeneous control
INHIB = "#5f8f5f"    # lesion / inhibition control (unused in the paper)
CHANCE = "#8d8d8d"   # chance line, nulls, reference lines
ACCENT = "#2f2f2f"   # primary text, fits, difference curves
MUTED = "#6b6b6b"    # secondary text
GRID = "#e4e4e4"     # hairline grid
AXIS = "#bdbdbd"     # bottom spine and tick marks
DOT = "#1e1e1e"      # per-seed dots on bars

COL = {"het": HET, "hom": HOM, "scaled": SCALED}

LABEL = {
    "het": "Heterogeneous",
    "hom": "Homogeneous",
    "scaled": "Scaled-init homogeneous",
}

# fixed-tau ladder: dark (short tau) to light (long tau); a neutral ramp so it never competes with red/blue
FIXED_RAMP = {2.0: "#303030", 3.5: "#5c5c5c", 15.0: "#858585", 35.0: "#adadad", 200.0: "#cfcfcf"}
FIXED_LABEL = {2.0: "Fixed 2 ms", 3.5: "Fixed 3.5 ms", 15.0: "Fixed 15 ms", 35.0: "Fixed 35 ms",
               200.0: "Fixed 200 ms"}

CHANCE_LEVEL = 0.25   # 4-class task


# ---------------------------------------------------------------- rc
def _font_family():
    for name in ("Arial", "Helvetica", "Liberation Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return [name, "DejaVu Sans"]
        except Exception:
            continue
    return ["DejaVu Sans"]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _font_family(),
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.size": 10,
        "text.color": ACCENT,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.titlecolor": ACCENT,
        "axes.labelsize": 9.5,
        "axes.labelcolor": ACCENT,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": ACCENT,
        "ytick.labelcolor": ACCENT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.3,
        "legend.labelcolor": ACCENT,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "errorbar.capsize": 0,
    })


# ---------------------------------------------------------------- titles and reference lines
def panel_title(ax, letter, text=None, **kw):
    """'(a) Task accuracy' style panel title, bold, left-aligned. Pass letter=None for a single-panel figure."""
    t = f"({letter}) {text}" if letter else (text or "")
    ax.set_title(t, loc="left", fontweight="bold", **kw)


def chance_line(ax, level=CHANCE_LEVEL, label="Chance"):
    ax.axhline(level, color=CHANCE, ls=(0, (2, 2)), lw=1.0, zorder=0, label=label)


def ref_line(ax, y=None, x=None, label=None, color=CHANCE, ls=(0, (2, 2)), lw=1.0):
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=lw, zorder=0, label=label)
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=lw, zorder=0, label=label)


def note(ax, text, loc="upper left", fontsize=8, color=MUTED):
    """Small muted text inside the axes for a statistic. Never narrative."""
    pos = {"upper left": (0.02, 0.97, "left", "top"), "upper right": (0.98, 0.97, "right", "top"),
           "lower left": (0.02, 0.03, "left", "bottom"), "lower right": (0.98, 0.03, "right", "bottom"),
           "lower center": (0.5, 0.03, "center", "bottom"), "upper center": (0.5, 0.97, "center", "top")}[loc]
    ax.text(pos[0], pos[1], text, transform=ax.transAxes, ha=pos[2], va=pos[3], fontsize=fontsize,
            color=color, linespacing=1.3)


def note_below(ax, text, dy=-0.17, fontsize=8, color=MUTED):
    """A statistics line under the axes, below the tick labels, so it never touches the data."""
    ax.text(0.5, dy, text, transform=ax.transAxes, ha="center", va="top", fontsize=fontsize, color=color)


def end_label(ax, x, y, text, color=ACCENT, dx=5, dy=0, fontsize=8.5, **kw):
    """Direct label at the end of a line, in text ink, offset in points."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha="left",
                fontsize=fontsize, color=color, **kw)


def legend(ax, loc="best", **kw):
    return ax.legend(loc=loc, frameon=False, **kw)


def plain_log_ticks(ax, ticks, axis="x"):
    """Plain numbers (1, 10, 100) on a log axis instead of powers of ten."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    a.set_minor_formatter(NullFormatter())


def plain_log_ticks(ax, ticks, axis="x"):
    """Plain numerals (1, 10, 100) on a log axis instead of 10^n."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def legend_below(ax, ncol=3, dy=-0.16, **kw):
    """Legend centred under the axes, for scatter panels where every corner holds data."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, frameon=False, **kw)


# ---------------------------------------------------------------- marks
def band(ax, x, mean, half, color, label=None, alpha=0.12, lw=2.0, **kw):
    """Mean line with a band of +/- half. The caller decides whether half is an SD or a 95% CI and the caption
    states which."""
    ax.plot(x, mean, color=color, label=label, lw=lw, **kw)
    ax.fill_between(x, np.asarray(mean) - np.asarray(half), np.asarray(mean) + np.asarray(half),
                    color=color, alpha=alpha, lw=0, zorder=1)


def dots(ax, x, y, color, s=34, label=None, zorder=4, marker="o", **kw):
    """Scatter dots with a white ring so they stay legible where they overlap."""
    return ax.scatter(x, y, s=s, color=color, label=label, zorder=zorder, marker=marker,
                      edgecolor="white", linewidth=0.8, **kw)


def bars(ax, x, heights, colors, width=0.58, err=None, label=None, zorder=2):
    """Thin bars with a white gap; whiskers in text ink, no caps."""
    return ax.bar(x, heights, width, yerr=err, color=colors, label=label, edgecolor="white", linewidth=1.0,
                  error_kw=dict(lw=1.2, ecolor=ACCENT, capsize=0), zorder=zorder)


def seed_dots(ax, x, values, width=0.58, rng=None, s=10, alpha=0.45):
    """Per-network values as faint jittered dots on top of a bar."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, float)
    jit = rng.uniform(-width * 0.28, width * 0.28, len(values))
    ax.scatter(np.full(len(values), x) + jit, values, s=s, color=DOT, alpha=alpha, lw=0, zorder=4)


# ---------------------------------------------------------------- statistics text
def ci95_mean(v):
    """Half-width of the 95% CI on the mean of v, ignoring NaN."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    return (1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def sig_stars(p):
    return "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def fmt_p(p, floor=None):
    """Format a p-value. Scientific notation is set in mathtext so it reads as 9.3 x 10^-7, not 9.3e-07.
    If the value is at an enumeration floor, say so."""
    if p != p:
        return "n/a"
    if floor is not None and p <= floor * 1.001:
        return _sci(p) + " (floor)"
    return _sci(p) if p < 1e-3 else f"{p:.3f}"


def _sci(p):
    s = f"{p:.1e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def stats_line(r, test="t", wins_label="seeds"):
    """One formal line of statistics for a paired result dict from stats_core.paired_triple.
    test = 't' (paired t, the twenty-pair convention) or 'perm' (sign-flip, the ten-seed control convention)."""
    if test == "t":
        p_txt = f"paired $t$ $p$ = {fmt_p(r['t_p'])}"
    else:
        p_txt = f"sign-flip $p$ = {fmt_p(r['perm_p'], r.get('perm_floor'))}"
    return (f"$\\Delta$ = {r['delta']:+.3f},  $d_z$ = {r['dz']:+.2f},  {p_txt},  "
            f"{r['wins']}/{r['n']} {wins_label}")


# ---------------------------------------------------------------- paired idioms
def paired_slopes(ax, left, right, color_left=HET, color_right=HOM,
                  labels=(LABEL["het"], LABEL["hom"]), s=38):
    """Connected pairs, one thin grey line per seed, coloured dots at either end. The pairing is the design,
    so the connectors are kept but recessive."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    for a, b in zip(left, right):
        ax.plot([0, 1], [a, b], color="#c4c4c4", lw=0.9, alpha=0.9, zorder=1)
    dots(ax, np.zeros_like(left), left, color_left, s=s, label=labels[0])
    dots(ax, np.ones_like(right), right, color_right, s=s, label=labels[1])
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)


def arch_legend(fig, bbox=(0.5, -0.04), ncol=2, extra=(), loc="lower center", fontsize=9):
    """One legend for the whole figure, below the panels."""
    handles = [Patch(color=HET, label=LABEL["het"]), Patch(color=HOM, label=LABEL["hom"])]
    handles += [Patch(color=c, label=l) for c, l in extra]
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, ncol=ncol + len(extra), frameon=False,
               fontsize=fontsize)


def paired_bars(ax, measures, vals, stats=None, labels=(LABEL["het"], LABEL["hom"]), colors=(HET, HOM),
                points=True, width=0.34, test="t", rotate=0, show_stats=True):
    """Grouped bars, heterogeneous against homogeneous, one group per measure, 95% CI whiskers on the
    across-seed mean, per-seed dots. `vals` is {arch: array (n_measures, n_seeds)}; `stats` is a list of
    paired-result dicts (one per measure) whose paired t p and win count are printed under the group."""
    measures = list(measures)
    x = np.arange(len(measures), dtype=float)
    rng = np.random.default_rng(0)
    for i, arch in enumerate(("het", "hom")):
        V = np.atleast_2d(np.asarray(vals[arch], float))
        m = np.nanmean(V, axis=1)
        h = np.array([ci95_mean(row) for row in V])
        pos = x + (i - 0.5) * (width + 0.04)
        bars(ax, pos, m, colors[i], width=width, err=h, label=labels[i])
        if points:
            for j, row in enumerate(V):
                seed_dots(ax, pos[j], row, width=width, rng=rng)
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=rotate, ha="right" if rotate else "center")
    ax.tick_params(axis="x", length=0)
    if len(measures) == 1:
        ax.set_xlim(-0.75, 0.75)
    if stats and show_stats:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.16)
        for j, r in enumerate(stats):
            if r is None:
                continue
            p = r["t_p"] if test == "t" else r["perm_p"]
            floor = None if test == "t" else r.get("perm_floor")
            who = LABEL["het"] if r["delta"] > 0 else LABEL["hom"]
            k = r["wins"] if r["delta"] > 0 else r["n"] - r["wins"]
            txt = f"$p$ = {fmt_p(p, floor)}\n{who} higher on {k}/{r['n']}"
            ax.text(x[j], hi + (hi - lo) * 0.02, txt, ha="center", va="bottom", fontsize=7.6,
                    color=MUTED, linespacing=1.25)
    return x


# ---------------------------------------------------------------- save
def save(fig, path, close=True):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    print(f"  saved {path.name}", flush=True)
