"""Inventory, verify, and consolidate every 4-class 64n checkpoint into one folder in the paper repo.

Verifies per file: loads, nb_recurrent==64, hidden tau matches the filename claim, tau-training flag
matches the batch, and a test accuracy exists. Writes MANIFEST.md + manifest.json alongside the copies.
"""
import json, re, shutil, sys
from pathlib import Path
import numpy as np
import torch

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = RROOT / "Paper Repo (Clean)"
DEST = REPO / "checkpoints" / "all_networks_4class_64n"

SRC_LEARNED = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"
SRC_CTRL = REPO / "checkpoints" / "seeded_run_1_64n_controls" / "4class_lognormal"
SRC_SHORT = RROOT / "Short-Tau + Scaled-Init Controls (Colab Upload)" / "outputs" / "checkpoints"
SRC_SMALL = RROOT / "Small Test (Colab Upload)" / "outputs" / "checkpoints"
SRC_FIXED = RROOT / "Fixed-Tau Sweep (Colab Upload)" / "outputs" / "checkpoints"

SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_NEW = list(range(1001, 1011))
SEEDS_ALL = SEEDS_ORIG + SEEDS_NEW


def tau_ms(beta):
    return -1.0 / np.log(np.clip(np.asarray(beta, dtype=np.float64), 1e-9, 1 - 1e-9))


# (group, expected_count, [(src_path, dest_name, expected_tau_or_None, expect_tau_trained)])
plan = []

# 1. Learned pairs: 20 het + 20 hom, tau trained, no fixed tau expectation
for s in SEEDS_ALL:
    plan.append(("learned_het", SRC_LEARNED / f"4class_seed{s}_het_lognormal.pt",
                 f"4class_seed{s}_het_lognormal.pt", None, True, s))
    plan.append(("learned_hom", SRC_LEARNED / f"4class_seed{s}_hom_geommean_lognormal.pt",
                 f"4class_seed{s}_hom_geommean_lognormal.pt", None, True, s))

# 2. Scaled-init homs: 10 (seed 202 lives in the local controls dir under a c-tagged name)
for s in [101, 550, 660, 710, 820, 930]:
    plan.append(("scaled_init", SRC_SHORT / f"4class_seed{s}_hom_scaledinit_inputrec_lognormal.pt",
                 f"4class_seed{s}_hom_scaledinit_inputrec_lognormal.pt", None, True, s))
for s in [210, 340, 440]:
    plan.append(("scaled_init", SRC_SMALL / f"4class_seed{s}_hom_scaledinit_inputrec_lognormal.pt",
                 f"4class_seed{s}_hom_scaledinit_inputrec_lognormal.pt", None, True, s))
plan.append(("scaled_init", SRC_CTRL / "4class_seed202_hom_scaledinit_inputrec_c6.87_lognormal.pt",
             "4class_seed202_hom_scaledinit_inputrec_lognormal.pt", None, True, 202))

# 3-6. Fixed-tau ladders: 10 each at 2, 3.5, 15, 35 ms. Tau frozen.
for tag, tau, src in [("fixed_2ms", 2.0, SRC_SHORT), ("fixed_3p5ms", 3.5, SRC_SHORT),
                      ("fixed_15ms", 15.0, SRC_FIXED), ("fixed_35ms", 35.0, SRC_FIXED)]:
    for s in SEEDS_ORIG:
        fn = f"4class_seed{s}_hom_{tag.replace('fixed_', 'fixed')}_lognormal.pt"
        plan.append((tag, src / fn, fn, tau, False, s))

# Extra (documented, outside the six expected batches)
plan.append(("extra_fixed_200ms", SRC_SMALL / "4class_seed202_hom_fixed200ms_lognormal.pt",
             "4class_seed202_hom_fixed200ms_lognormal.pt", 200.0, False, 202))

DEST.mkdir(parents=True, exist_ok=True)
rows, problems = [], []
for group, src, dest_name, exp_tau, exp_trained, seed in plan:
    rec = {"group": group, "seed": seed, "file": dest_name, "source": str(src.relative_to(RROOT))}
    if not src.exists():
        rec["status"] = "MISSING"
        problems.append(f"MISSING: {src}")
        rows.append(rec)
        continue
    try:
        ck = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as e:
        rec["status"] = f"UNREADABLE: {e}"
        problems.append(f"UNREADABLE: {src}: {e}")
        rows.append(rec)
        continue

    prms = dict(ck.get("prms", {}))
    sd = ck["model_state_dict"]
    beta = sd["network.0.beta"].cpu().numpy().ravel()
    t = tau_ms(beta)
    rec["n_hidden"] = int(prms.get("nb_recurrent", -1))
    rec["tau_unique"] = int(np.unique(np.round(t, 3)).size)
    rec["tau_median_ms"] = round(float(np.median(t)), 3)
    rec["tau_min_ms"] = round(float(t.min()), 3)
    rec["tau_max_ms"] = round(float(t.max()), 3)
    rec["train_ab"] = int(prms.get("train_ab", -1))
    rec["train_hom_ab"] = int(prms.get("train_hom_ab", -1))
    hist = ck.get("history", {}).get("test_acc", [])
    rec["test_acc"] = round(float(hist[-1]), 4) if isinstance(hist, list) and hist else None
    rec["epochs"] = len(hist) if isinstance(hist, list) else None

    issues = []
    if rec["n_hidden"] != 64:
        issues.append(f"n_hidden={rec['n_hidden']}")
    if exp_tau is not None:
        if abs(rec["tau_median_ms"] - exp_tau) > max(0.15, 0.02 * exp_tau):
            issues.append(f"tau {rec['tau_median_ms']} != claimed {exp_tau}")
        if rec["tau_unique"] != 1:
            issues.append(f"fixed net has {rec['tau_unique']} distinct taus")
    is_hom = "het" not in dest_name
    flag = rec["train_hom_ab"] if is_hom else rec["train_ab"]
    if exp_trained and flag != 1:
        issues.append(f"tau flag={flag}, expected trained(1)")
    if not exp_trained and flag == 1:
        issues.append(f"tau flag={flag}, expected frozen(0)")
    if rec["test_acc"] is None:
        issues.append("no test_acc history")

    rec["status"] = "OK" if not issues else "CHECK: " + "; ".join(issues)
    if issues:
        problems.append(f"{dest_name}: {'; '.join(issues)}")
    rows.append(rec)

    shutil.copy2(src, DEST / dest_name)

with open(DEST / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=1)

# ---- report ----
print(f"dest: {DEST}")
groups = {}
for r in rows:
    groups.setdefault(r["group"], []).append(r)
EXPECT = {"learned_het": 20, "learned_hom": 20, "scaled_init": 10, "fixed_2ms": 10,
          "fixed_3p5ms": 10, "fixed_15ms": 10, "fixed_35ms": 10, "extra_fixed_200ms": 1}
print(f"\n{'group':<20}{'found':>6}{'want':>6}{'ok':>5}  {'acc mean':>9}  tau median")
for g, want in EXPECT.items():
    rs = groups.get(g, [])
    ok = sum(1 for r in rs if r["status"] == "OK")
    accs = [r["test_acc"] for r in rs if r.get("test_acc") is not None]
    taus = [r["tau_median_ms"] for r in rs if r.get("tau_median_ms") is not None]
    am = f"{np.mean(accs):.3f}" if accs else "-"
    tm = f"{np.median(taus):.2f}" if taus else "-"
    mark = "" if (len(rs) == want and ok == want) else "  <<<"
    print(f"{g:<20}{len(rs):>6}{want:>6}{ok:>5}  {am:>9}  {tm:>10}{mark}")
print(f"\nTOTAL copied: {sum(1 for r in rows if r['status'].startswith('OK') or r['status'].startswith('CHECK'))} files")
print(f"core (six expected batches): {sum(len(groups.get(g, [])) for g in EXPECT if g != 'extra_fixed_200ms')} / 90")
if problems:
    print(f"\n!!! {len(problems)} PROBLEM(S):")
    for p in problems:
        print("   -", p)
else:
    print("\nAll files present and verified.")
