"""O-information / S-information subset sweep for the n=20 min/max analysis.

Reproduces the original pipeline exactly so new records are comparable to the legacy ones:
  spike cache : layer_recs[0][2] (binary spikes) at 1 ms, strided by 4 -> 4 ms, first 2048 test trials,
                reshaped to [64, 512000] uint8. Identical to Project Files/Discrete Zscore Pipeline.
  subset sweep: k in {2,4,6,8,10,12}, 50 random subsets per (seed, arch, k), plus the high-k tail
                {16:40, 24:25, 32:15} and MAX. Subsets are drawn from a hash-seeded rng (_ss), the same
                scheme the legacy runner used, so a re-run of an existing seed reproduces its subsets.
  estimator   : syntropy construct_joint_distribution -> o_information, s_information (exact discrete).

Writes to data_cache/analysis_suite/os_info_records_n20.jsonl. Resumable via (seed, arch, k, draw) dedup.

Usage:
  python code/analysis_suite/run_oinfo.py --seeds new         # the 10 new seeds (1001-1010), het + hom
  python code/analysis_suite/run_oinfo.py --seeds retrain     # the 9 no-WD hom retrains (when present)
  python code/analysis_suite/run_oinfo.py --seeds 1001        # a single seed, both arches (benchmark)
  python code/analysis_suite/run_oinfo.py --seeds new --cache-only   # just build caches, no sweep
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = RROOT / "Paper Repo (Clean)"
sys.path.insert(0, str(REPO / "code" / "training"))
sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
from data_gen import open_file
from syntropy.discrete.utils import construct_joint_distribution
from syntropy.discrete import o_information, s_information

CK = REPO / "checkpoints" / "all_networks_4class_64n"
NOWD = REPO / "checkpoints" / "seeded_run_1_64n_nowd" / "4class_lognormal"
OUTDIR = REPO / "data_cache" / "analysis_suite"
CACHE = OUTDIR / "spk_cache_4ms"
CACHE.mkdir(parents=True, exist_ok=True)
RECORDS = OUTDIR / "os_info_records_n20.jsonl"
LEGACY = RROOT / "Project Files" / "O-S Information" / "outputs" / "os_info_records.jsonl"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src.DEVICE = DEV
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
for k, v in {"nb_recurrent": 64, "batch_size": 256}.items():
    src.BASE_PRMS[k] = v
torch.backends.cuda.matmul.allow_tf32 = False

STRIDE = 4
N_TRIALS = 2048                 # 8 batches x 256, matches the 512000-column legacy cache
SUBSET_SIZES = [2, 4, 6, 8, 10, 12]
N_SUBSETS = 50
MIN_SUBSETS = 5
HIGHK_PLAN = {16: 40, 24: 25, 32: 15}

SEEDS_NEW = list(range(1001, 1011))
SEEDS_RETRAIN = [202, 210, 340, 440, 550, 660, 710, 820, 930]


def _ss(*p):
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)


def os_info(sub):
    jd = construct_joint_distribution(np.asarray(sub, dtype=int))
    O = float(o_information(jd)[1]); S = float(s_information(jd)[1])
    return O, S, (S + O) / 2.0, (S - O) / 2.0, len(jd)


def ckpt_path(seed, arch):
    if arch == "hom_nowd":
        return NOWD / f"4class_seed{seed}_hom_geommean_nowd_lognormal.pt"
    name = "het" if arch == "het" else "hom_geommean"
    return CK / f"4class_seed{seed}_{name}_lognormal.pt"


@torch.no_grad()
def build_cache(seed, arch):
    """[64, N_TRIALS*250] uint8, faithful to the legacy 4 ms cache."""
    out = CACHE / f"4class_seed{seed}_{arch}_spk.npy"
    if out.exists():
        return out, "cached"
    ck = torch.load(ckpt_path(seed, arch), map_location=DEV, weights_only=False)
    prms = dict(ck["prms"]); prms.update({"dtype": torch.float, "device": DEV, "cuda": DEV.type == "cuda"})
    prms["task_label_map"] = src.TASKS["4class"]["task_label_map"]; prms["class_list"] = list(range(20))
    model = src.RSNN(prms, rec=True).to(DEV); model.load_state_dict(ck["model_state_dict"]); model.eval()
    nb_steps = int(prms["nb_steps"]); nb_hidden = int(model.network[0].output_size)
    chunks, n = [], 0
    for x, _y in src.shd_generator(UNITS, TIMES, LABELS, prms, shuffle=False, epoch=0, drop_last=True):
        spk = model(0, 0, x.to(DEV))[0][2]                       # [B,T,64] binary
        spk = spk[:, ::STRIDE, :].detach().cpu().numpy()          # [B, 250, 64]
        chunks.append(np.transpose(spk, (2, 0, 1)).reshape(nb_hidden, -1))   # [64, B*250]
        n += spk.shape[0]
        if n >= N_TRIALS:
            break
    spk = np.concatenate(chunks, axis=1)[:, :N_TRIALS * (nb_steps // STRIDE)]
    assert np.all((spk == 0) | (spk == 1)), "not binary"
    np.save(out, spk.astype(np.uint8))
    del model
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return out, f"built [{spk.shape[0]},{spk.shape[1]}]"


def sweep_net(seed, arch, done, rec_file, base_only=False):
    spk = np.load(CACHE / f"4class_seed{seed}_{arch}_spk.npy").astype(int)
    active = np.where(spk.sum(1) > 0)[0]
    spk = spk[active]; n_active, T = spk.shape
    rec_arch = "hom" if arch.startswith("hom") else "het"      # legacy label; retrains map to 'hom'
    jobs = []
    for k in SUBSET_SIZES:
        if k > n_active:
            continue
        cap = max(min(N_SUBSETS, math.comb(n_active, k)), min(MIN_SUBSETS, math.comb(n_active, k)))
        rng = np.random.default_rng(_ss(seed, rec_arch, k)); seen = set(); subs = []
        while len(subs) < cap:
            s = tuple(sorted(rng.choice(n_active, size=k, replace=False).tolist()))
            if s not in seen:
                seen.add(s); subs.append(s)
        jobs += [(k, i, s) for i, s in enumerate(subs)]
    if not base_only:      # high-k tail is bias-dominated and only feeds the appendix; min/max uses k<=12
        for k, nsub in HIGHK_PLAN.items():
            if k > n_active:
                continue
            cap = min(nsub, math.comb(n_active, k))
            rng = np.random.default_rng(_ss(seed, rec_arch, k)); seen = set(); subs = []
            while len(subs) < cap:
                s = tuple(sorted(rng.choice(n_active, size=k, replace=False).tolist()))
                if s not in seen:
                    seen.add(s); subs.append(s)
            jobs += [(k, i, s) for i, s in enumerate(subs)]
        jobs.append((n_active, 0, tuple(range(n_active))))      # MAX

    todo = [(k, i, s) for (k, i, s) in jobs if (seed, rec_arch, k, i) not in done]
    if not todo:
        print(f"  seed {seed} {arch}: nothing to do (n_active={n_active})", flush=True)
        return 0
    t0 = time.time()
    for k, i, sub in todo:
        try:
            O, S, TC, DTC, nstates = os_info(spk[list(sub)])
            rec = {"seed": seed, "arch": rec_arch, "k": k, "subset_draw": i, "O_info": O, "S_info": S,
                   "TC": TC, "DTC": DTC, "n_states": nstates, "n_active": int(n_active),
                   "is_max": (k == n_active), "status": "ok", "neurons": [int(x) for x in sub],
                   "source": arch}
        except Exception as exc:
            rec = {"seed": seed, "arch": rec_arch, "k": k, "subset_draw": i, "n_active": int(n_active),
                   "status": f"error:{type(exc).__name__}", "source": arch}
        rec_file.write(json.dumps(rec) + "\n"); rec_file.flush()
    print(f"  seed {seed} {arch}: {len(todo)} subsets (n_active={n_active}) in {time.time()-t0:.0f}s", flush=True)
    return len(todo)


def load_done():
    done = set()
    if RECORDS.exists():
        for line in open(RECORDS):
            try:
                r = json.loads(line); done.add((r["seed"], r["arch"], r["k"], r["subset_draw"]))
            except Exception:
                pass
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="'new', 'retrain', or a single seed number")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--base-only", action="store_true", help="skip the high-k tail (min/max only needs k<=12)")
    ap.add_argument("--kmax", type=int, default=12, help="cap the base subset sizes (k>8 is bias-dominated)")
    args = ap.parse_args()
    SUBSET_SIZES = [k for k in SUBSET_SIZES if k <= args.kmax]

    if args.seeds == "new":
        jobs = [(s, a) for s in SEEDS_NEW for a in ("het", "hom")]
    elif args.seeds == "retrain":
        jobs = [(s, "hom_nowd") for s in SEEDS_RETRAIN]
    else:
        s = int(args.seeds)
        jobs = [(s, "het"), (s, "hom")] if s in SEEDS_NEW else [(s, "hom_nowd")]

    print(f"device {DEV} | {len(jobs)} nets | records -> {RECORDS.name}", flush=True)
    print("loading SHD test set ...", flush=True)
    _u, _t, _l = open_file(str(src.SHD_TEST))
    UNITS, TIMES, LABELS = list(_u[:]), list(_t[:]), np.array(_l[:])
    _u._v_file.close()

    for seed, arch in jobs:
        p = ckpt_path(seed, arch)
        if not p.exists():
            print(f"  seed {seed} {arch}: checkpoint MISSING ({p.name}), skip", flush=True)
            continue
        _, status = build_cache(seed, arch)
        print(f"  cache seed {seed} {arch}: {status}", flush=True)

    if args.cache_only:
        print("cache-only done.")
        sys.exit(0)

    done = load_done()
    print(f"resume: {len(done)} subset records present", flush=True)
    rec_file = open(RECORDS, "a")
    try:
        total = 0
        for seed, arch in jobs:
            if not ckpt_path(seed, arch).exists():
                continue
            total += sweep_net(seed, arch, done, rec_file, base_only=args.base_only)
        print(f"\ndone. {total} new subset records written.", flush=True)
    finally:
        rec_file.close()
