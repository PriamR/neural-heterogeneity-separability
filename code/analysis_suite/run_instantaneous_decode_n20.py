"""Instantaneous decodability of language and parity through the trial.

The four-class label factorises as language (English vs German, y // 2) and
parity (even vs odd, y % 2). This asks, at each timestep, how linearly
decodable each factor is from the hidden state, separately for the two arms.
The question is whether the heterogeneous code holds both factors at once and
throughout, while the homogeneous code resolves them one at a time and cannot
sustain both.

For each of the twenty seeds and both arms:
  - load the clean representation cache, hidden state of shape (trials, 50, 64);
  - at each of the 50 timesteps, fit a standardised logistic decoder with
    four-fold stratified cross-validation and record balanced accuracy, once
    for language and once for parity.

Curves are averaged across the twenty seeds. Three summaries are reported,
each as a within-seed paired het-vs-hom test through stats_core:
  - sustain: mean accuracy over the late window (last 40% of the trial), per
    factor, which is where the arms separate;
  - both-held: mean over the trial of min(language, parity) accuracy, high only
    when both factors are decodable at the same time;
  - peak offset: |argmax(language) - argmax(parity)| in timesteps, a serial
    code resolves the two factors at different times and scores higher.

Writes data_cache/analysis_suite/instantaneous_decode_n20.json and
docs/analysis_suite/figs/instantaneous_decode_langparity.png.

Run: python code/analysis_suite/run_instantaneous_decode_n20.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S                                # sets the Agg backend
import matplotlib.pyplot as plt
from stats_core import paired_triple

REPO = HERE.parent.parent                       # Final Repo
CACHE = REPO / "data_cache" / "representation_clean"
OUTJSON = REPO / "data_cache" / "analysis_suite" / "instantaneous_decode_n20.json"
FIGS = REPO / "docs" / "analysis_suite" / "figs"

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
ARCHS = ("het", "hom")
FOLDS = 4
LATE_FRAC = 0.60          # late window starts at 60% of the trial

# language and parity get their own hues, held across both arch panels, so the
# reader compares the shape of the het panel against the hom panel
COL_LANG = "#1b7837"      # green
COL_PAR = "#762a83"       # purple


def decode_curve(H, labels):
    """Four-fold CV balanced accuracy at each timestep. H (N, T, 64)."""
    N, T, _ = H.shape
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    splits = list(skf.split(np.zeros(N), labels))
    cls = np.unique(labels)
    acc = np.zeros(T)
    for t in range(T):
        Xt = H[:, t, :]
        fold = []
        for tr, te in splits:
            mu = Xt[tr].mean(0)
            sd = Xt[tr].std(0) + 1e-6
            clf = LogisticRegression(max_iter=200, C=1.0)
            clf.fit((Xt[tr] - mu) / sd, labels[tr])
            pred = clf.predict((Xt[te] - mu) / sd)
            fold.append(np.mean([(pred[labels[te] == c] == c).mean() for c in cls]))
        acc[t] = np.mean(fold)
    return acc


def main():
    S.apply()
    t_start = time.time()
    curves = {a: {"lang": [], "par": []} for a in ARCHS}
    T = None
    for seed in SEEDS:
        for a in ARCHS:
            d = np.load(CACHE / f"repr_seed{seed}_{a}.npz")
            H = d["hidden"].astype(np.float64)
            y = d["y"]
            lang = (y // 2).astype(int)
            par = (y % 2).astype(int)
            curves[a]["lang"].append(decode_curve(H, lang))
            curves[a]["par"].append(decode_curve(H, par))
            T = H.shape[1]
        print(f"  seed {seed} done ({time.time()-t_start:.0f}s)", flush=True)

    arr = {a: {k: np.array(v) for k, v in curves[a].items()} for a in ARCHS}   # (20, T)
    x = np.arange(T) / (T - 1)                     # fraction of trial
    late = np.arange(int(LATE_FRAC * T), T)

    # ---- per-seed summaries for the paired tests
    def sustain(a, task):
        return arr[a][task][:, late].mean(axis=1)          # (20,)

    def both_held(a):
        return np.minimum(arr[a]["lang"], arr[a]["par"]).mean(axis=1)   # (20,)

    def peak_offset(a):
        pl = arr[a]["lang"].argmax(axis=1)
        pp = arr[a]["par"].argmax(axis=1)
        return np.abs(pl - pp).astype(float)               # (20,)

    stats = {
        "sustain_lang": paired_triple(sustain("het", "lang"), sustain("hom", "lang"),
                                      "late language decodability", higher_is_better=True),
        "sustain_par": paired_triple(sustain("het", "par"), sustain("hom", "par"),
                                     "late parity decodability", higher_is_better=True),
        "both_held": paired_triple(both_held("het"), both_held("hom"),
                                   "both factors held at once", higher_is_better=True),
        "peak_offset": paired_triple(peak_offset("het"), peak_offset("hom"),
                                     "language/parity peak offset", higher_is_better=True),
    }

    print("\n--- paired het vs hom (n=20) ---")
    for key, r in stats.items():
        print(f"  {key:<14} het {r['het_mean']:+.3f}  hom {r['hom_mean']:+.3f}  "
              f"d {r['delta']:+.3f}  dz {r['dz']:+.2f}  p {S.fmt_p(r['perm_p'], r['perm_floor'])}  "
              f"{r['wins']}/{r['n']}")

    # ---- figure: two panels, het then hom, language and parity curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, a in zip(axes, ARCHS):
        for task, col, lab in (("lang", COL_LANG, "language"), ("par", COL_PAR, "parity")):
            m = arr[a][task].mean(axis=0)
            sem = arr[a][task].std(axis=0, ddof=1) / np.sqrt(arr[a][task].shape[0])
            S.band(ax, x, m, sem, col, label=lab, lw=2.2)
        ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="chance")
        ax.axvspan(x[late[0]], x[-1], color=S.CHANCE, alpha=0.06, lw=0)
        ax.set_title(S.LABEL[a])
        ax.set_xlabel("time (fraction of trial)")
        ax.set_ylim(0.45, 0.9)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("decode accuracy (balanced)")
    fig.suptitle("Instantaneous decodability of the two task factors",
                 fontweight="bold", y=1.02)
    FIGS.mkdir(parents=True, exist_ok=True)
    S.save(fig, FIGS / "instantaneous_decode_langparity.png")

    out = {
        "analysis": "instantaneous_language_parity_decode",
        "seeds": SEEDS, "n": len(SEEDS), "n_timesteps": T, "folds": FOLDS,
        "late_window_frac": LATE_FRAC,
        "x_fraction": x.tolist(),
        "mean_curves": {a: {task: arr[a][task].mean(axis=0).tolist()
                            for task in ("lang", "par")} for a in ARCHS},
        "sem_curves": {a: {task: (arr[a][task].std(axis=0, ddof=1)
                                  / np.sqrt(arr[a][task].shape[0])).tolist()
                           for task in ("lang", "par")} for a in ARCHS},
        "per_seed_curves": {a: {task: arr[a][task].tolist()
                                for task in ("lang", "par")} for a in ARCHS},
        "stats": stats,
    }
    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUTJSON.name} and instantaneous_decode_langparity.png "
          f"({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
