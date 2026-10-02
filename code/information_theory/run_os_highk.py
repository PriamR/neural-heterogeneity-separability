"""High-k / MAX extension for the O-S information sweep.

Adds k in {16, 24, 32} (where <= n_active) plus MAX (all active neurons, single subset) on top of the
existing {2,4,6,8,10,12} sweep. Appends to the SAME os_info_records.jsonl so Stage 3 aggregation /
the figure pick the new rows up automatically. Resumable via (seed, arch, k, subset_draw) dedup.

Memory is bounded by T (syntropy stores only observed states), so even MAX is feasible. These high-k
values are *bias-dominated* (T/2^k << 1) and are included for completeness / to show the bias wall.

Run: python run_os_highk.py
"""
import json, time, hashlib, math
from pathlib import Path
import numpy as np
from syntropy.discrete.utils import construct_joint_distribution
from syntropy.discrete import o_information, s_information

# ----------------------------- config -----------------------------
DEFAULT_BASE = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
PROJECT_ROOT = DEFAULT_BASE if DEFAULT_BASE.exists() else Path.cwd()
PROJECT_FILES = PROJECT_ROOT / "Project Files"
SPK_CACHE  = PROJECT_FILES / "Discrete Zscore Pipeline" / "outputs" / "spk_cache"
OUTPUT_DIR = PROJECT_FILES / "O-S Information" / "outputs"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_PATH = OUTPUT_DIR / "os_info_records.jsonl"

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
ARCHS = ["het", "hom"]
SUBSET_PLAN = {16: 40, 24: 25, 32: 15}   # k -> n random subsets (tapered; bias-dominated up here)
# MAX (= n_active) handled separately, single deterministic subset.

def _ss(*p):
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)

def os2(sub):
    """O, S, TC, DTC, n_states for a [k,T] binary matrix. 2 delta_k calls; derive TC/DTC."""
    jd = construct_joint_distribution(np.asarray(sub, dtype=int))
    O = float(o_information(jd)[1]); S = float(s_information(jd)[1])
    TC = (S + O) / 2.0; DTC = (S - O) / 2.0      # O = TC - DTC, S = TC + DTC
    return O, S, TC, DTC, len(jd)

# ----------------------------- resume -----------------------------
done = set()
if RECORDS_PATH.exists():
    for line in open(RECORDS_PATH):
        try:
            r = json.loads(line); done.add((r["seed"], r["arch"], r["k"], r["subset_draw"]))
        except Exception:
            pass
print(f"Resuming: {len(done)} subset records already present.")

rec_file = open(RECORDS_PATH, "a")
try:
    for seed in SEEDS:
        for arch in ARCHS:
            p = SPK_CACHE / f"4class_seed{seed}_{arch}_spk.npy"
            if not p.exists():
                print(f"seed {seed} {arch}: MISSING cache, skip"); continue
            spk = np.load(p).astype(int)
            active = np.where(spk.sum(1) > 0)[0]
            spk = spk[active]; n_active, T = spk.shape

            # build job list: sampled high-k + MAX
            jobs = []   # (k, subset_draw, neuron_tuple)
            for k, nsub in SUBSET_PLAN.items():
                if k > n_active:
                    continue
                cap = min(nsub, math.comb(n_active, k))
                rng = np.random.default_rng(_ss(seed, arch, k)); seen = []
                while len(seen) < cap:
                    s = tuple(sorted(rng.choice(n_active, size=k, replace=False).tolist()))
                    if s not in seen: seen.append(s)
                for i, s in enumerate(seen):
                    jobs.append((k, i, s))
            # MAX: all active neurons, single subset (skipped by dedup if k already covered)
            jobs.append((n_active, 0, tuple(range(n_active))))

            todo = [(k, i, s) for (k, i, s) in jobs if (seed, arch, k, i) not in done]
            if not todo:
                print(f"seed {seed} {arch}: nothing to do (n_active={n_active})"); continue
            ks = sorted({k for k, _, _ in todo})
            print(f"seed {seed} {arch}: n_active={n_active} T={T} | {len(todo)} subsets over k={ks}")
            for k, i, sub in todo:
                t0 = time.time()
                try:
                    O, S, TC, DTC, nstates = os2(spk[list(sub)])
                    rec = {"seed": seed, "arch": arch, "k": k, "subset_draw": i,
                           "O_info": O, "S_info": S, "TC": TC, "DTC": DTC,
                           "n_states": nstates, "n_active": int(n_active),
                           "is_max": (k == n_active), "status": "ok",
                           "neurons": [int(x) for x in sub]}
                except Exception as exc:
                    rec = {"seed": seed, "arch": arch, "k": k, "subset_draw": i,
                           "n_active": int(n_active), "status": f"error:{type(exc).__name__}"}
                rec_file.write(json.dumps(rec) + "\n"); rec_file.flush()
                tag = "MAX" if rec.get("is_max") else ""
                print(f"    k={k:3d} draw={i:2d} {tag:>3} -> "
                      f"{'O=%+.3f S=%.3f ns=%d' % (rec.get('O_info',0), rec.get('S_info',0), rec.get('n_states',0)) if rec['status']=='ok' else rec['status']}"
                      f"  ({time.time()-t0:.1f}s)")
finally:
    rec_file.close()
print("High-k extension complete.")
