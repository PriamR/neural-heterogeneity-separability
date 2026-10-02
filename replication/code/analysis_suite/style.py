"""House style for every figure in the paper. Import this and nothing else for colours, sizes and helpers,
so the whole suite is visually consistent.

Palette rationale: het and hom keep the colours used since the first analyses (red / blue) so old and new
figures read the same. The controls get their own hues that never collide with het or hom: amber for the
scaled-init nets, a purple ramp for the fixed-tau ladder (dark = short tau, light = long tau).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- palette
HET = "#d1495b"      # heterogeneous (learned tau spread)
HOM = "#30638e"      # homogeneous (single learned tau)
SCALED = "#e8a33d"   # scaled-init hom control
INHIB = "#5f8f5f"    # lesion / inhibition control
CHANCE = "#8d8d8d"   # chance line, nulls, reference lines
ACCENT = "#2f2f2f"   # annotation text / arrows

COL = {"het": HET, "hom": HOM, "scaled": SCALED}

# fixed-tau ladder: dark (short tau) to light (long tau)
FIXED_RAMP = {2.0: "#3d1f52", 3.5: "#6a3d8f", 15.0: "#9b6dc4", 35.0: "#c4a8e0", 200.0: "#ded0f0"}

LABEL = {
    "het": "heterogeneous",
    "hom": "homogeneous",
    "scaled": "scaled-init hom",
}

CHANCE_LEVEL = 0.25   # 4-class task


# ---------------------------------------------------------------- rc
def apply():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "errorbar.capsize": 3,
    })


# ---------------------------------------------------------------- helpers
def chance_line(ax, level=CHANCE_LEVEL, label="chance (4-class)"):
    ax.axhline(level, color=CHANCE, ls=":", lw=1.2, zorder=0, label=label)


def band(ax, x, mean, sd, color, label=None, alpha=0.15, **kw):
    """Mean line with a +/- 1 SD band. The suite's only band convention."""
    ax.plot(x, mean, color=color, label=label, **kw)
    ax.fill_between(x, np.asarray(mean) - np.asarray(sd), np.asarray(mean) + np.asarray(sd),
                    color=color, alpha=alpha, lw=0, zorder=1)


def paired_slopes(ax, left, right, color_left=HET, color_right=HOM, jitter=0.0,
                  labels=("heterogeneous", "homogeneous")):
    """Connected het-hom pairs, one line per seed. Retained for one-off diagnostics only.

    Not used in the reported figures: twenty crossing lines read as noise rather than as a result, and the
    package this replies to does not use the idiom anywhere. Use `paired_bars` instead, which carries the
    same paired statistics in the annotation.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    for a, b in zip(left, right):
        ax.plot([0, 1], [a, b], color="#b0b0b0", lw=0.9, alpha=0.8, zorder=1)
    ax.scatter(np.zeros_like(left) + jitter, left, color=color_left, s=32, zorder=3, label=labels[0])
    ax.scatter(np.ones_like(right) - jitter, right, color=color_right, s=32, zorder=3, label=labels[1])
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)


# ---------------------------------------------------------------- paired bars
def ci95_mean(v):
    """Half-width of the 95% CI on the mean of v, ignoring NaN."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    return (1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def arch_legend(fig, bbox=(0.5, -0.07), ncol=2, extra=(), loc="lower center"):
    """One legend for the whole figure, below the panels.

    Below rather than inside: every bar panel now carries a significance annotation in its own headroom, and
    an in-axes legend collides with it. Below the axes there is nothing to collide with at any figure size.
    """
    from matplotlib.patches import Patch
    handles = [Patch(color=HET, label=LABEL["het"]), Patch(color=HOM, label=LABEL["hom"])]
    handles += [Patch(color=c, label=l) for c, l in extra]
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, ncol=ncol, frameon=False, fontsize=9)


def paired_bars(ax, measures, vals, stats=None, labels=("heterogeneous", "homogeneous"),
                colors=(HET, HOM), points=True, width=0.36, star_only=False, rotate=0):
    """Grouped bars, het against hom, one group per measure, with 95% CI whiskers on the across-seed mean.

    This is the idiom the package uses for every group comparison, and it is what the reported figures use.
    `vals` is {arch: array of shape (n_measures, n_seeds)}; `stats` is the list of paired-result dicts that
    supplies delta, d_z, the permutation p and the win count, printed under each group so a figure lifted
    into a talk carries its own statistics.

    Per-seed values are drawn as faint jittered dots on top of the bars. Bars alone would say how big the
    effect is without saying whether the seeds agree about it; the dots and the win count both answer that.
    """
    measures = list(measures)
    x = np.arange(len(measures), dtype=float)
    rng = np.random.default_rng(0)
    for i, arch in enumerate(("het", "hom")):
        V = np.atleast_2d(np.asarray(vals[arch], float))
        m = np.nanmean(V, axis=1)
        h = np.array([ci95_mean(row) for row in V])
        pos = x + (i - 0.5) * width
        # a white edge keeps the two bars of a group from reading as one block
        ax.bar(pos, m, width, yerr=h, color=colors[i], label=labels[i], capsize=3,
               edgecolor="white", linewidth=1.0,
               error_kw=dict(lw=1.2, ecolor="#3a3a3a"), zorder=2)
        if points:
            for j, row in enumerate(V):
                jit = rng.uniform(-width * 0.26, width * 0.26, len(row))
                ax.scatter(np.full(len(row), pos[j]) + jit, row, s=9, color="#1e1e1e",
                           alpha=0.30, lw=0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=rotate, ha="right" if rotate else "center")
    if not any(str(m).strip() for m in measures):
        ax.tick_params(axis="x", length=0)          # a lone unlabelled tick reads as a missing label
    if len(measures) == 1:
        ax.set_xlim(-0.75, 0.75)                    # one group should not stretch across the whole panel
    if stats:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * (0.10 if star_only else 0.20))
        for j, r in enumerate(stats):
            if r is None:
                continue
            if star_only:
                txt = sig_stars(r["perm_p"])
            else:
                # name which arm the win count belongs to, so the fraction cannot be read the wrong way
                who = (f"het higher {r['wins']}/{r['n']}" if r["delta"] > 0
                       else f"hom higher {r['n'] - r['wins']}/{r['n']}")
                txt = (f"{sig_stars(r['perm_p'])}  p {fmt_p(r['perm_p'], r.get('perm_floor'))}\n{who}")
            ax.text(x[j], hi + (hi - lo) * 0.02, txt, ha="center", va="bottom", fontsize=7.6,
                    color="#3a3a3a", linespacing=1.25)
    return x


def sig_stars(p):
    return "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def fmt_p(p, floor=None):
    """Format a p-value. If it is at the enumeration floor, say so rather than printing a bare number."""
    if p != p:
        return "n/a"
    if floor is not None and p <= floor * 1.001:
        return f"{p:.1e} (floor)"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def save(fig, path, close=True):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  saved {path.name}", flush=True)
