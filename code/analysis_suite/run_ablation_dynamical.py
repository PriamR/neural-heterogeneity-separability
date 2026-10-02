"""Dynamical neuron lesion (authoritative): progressively remove hidden neurons from the recurrent network and
measure task accuracy through the real readout. A lesioned neuron is fully disconnected, its feedforward input,
recurrent in/out and readout weights zeroed, so it neither spikes nor influences anything.

Orderings:
  het   longest-tau-first and shortest-tau-first (which timescales carry the task)
  hom   random order (its taus are equal)
  fixed random order, one curve per rung (2/3.5/15/35 ms)

Eval-only forward passes on a fixed test subset for speed. Resumable: appends per-net curves to
ablation_dynamical.jsonl and skips nets already present.

Run: python code/analysis_suite/run_ablation_dynamical.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = RROOT / "Paper Repo (Clean)"
sys.path.insert(0, str(REPO / "code" / "training"))
sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
from data_gen import open_file

CK = REPO / "checkpoints" / "all_networks_4class_64n"
CACHE = REPO / "data_cache" / "analysis_suite"
OUTJL = REPO / "docs" / "analysis_suite" / "ablation_dynamical.jsonl"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src.DEVICE = DEV
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
for k, v in {"nb_recurrent": 64, "batch_size": 256}.items():
    src.BASE_PRMS[k] = v
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_ALL = SEEDS_ORIG + list(range(1001, 1011))
CONFOUNDED = [202, 210, 340, 440, 550, 660, 710, 820, 930]
RUNGS = {"fixed2ms": 2.0, "fixed3p5ms": 3.5, "fixed15ms": 15.0, "fixed35ms": 35.0}
LEVELS = [0, 8, 16, 24, 32, 40, 48, 56, 63]
MAX_TRIALS = 768


def hom_name(s):
    return (f"4class_seed{s}_hom_geommean_nowd_lognormal" if s in CONFOUNDED
            else f"4class_seed{s}_hom_geommean_lognormal")


print("loading SHD test subset ...", flush=True)
_u, _t, _l = open_file(str(src.SHD_TEST))
UNITS, TIMES, LABELS = list(_u[:]), list(_t[:]), np.array(_l[:])
_u._v_file.close()


def build_model(name):
    ck = torch.load(CK / f"{name}.pt", map_location=DEV, weights_only=False)
    prms = dict(ck["prms"]); prms.update({"dtype": torch.float, "device": DEV, "cuda": DEV.type == "cuda"})
    prms["task_label_map"] = src.TASKS["4class"]["task_label_map"]; prms["class_list"] = list(range(20))
    model = src.RSNN(prms, rec=True).to(DEV)
    return model, ck["model_state_dict"]


def masked_sd(sd, abl):
    m = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    if len(abl):
        idx = torch.tensor(list(abl), dtype=torch.long)
        m["network.0.w"][:, idx] = 0.0
        m["network.0.v"][idx, :] = 0.0
        m["network.0.v"][:, idx] = 0.0
        m["network.1.w"][idx, :] = 0.0
    return m


@torch.no_grad()
def accuracy(model, sd, abl):
    model.load_state_dict(masked_sd(sd, abl)); model.eval()
    correct = tot = 0
    for x, y in src.shd_generator(UNITS, TIMES, LABELS, PRMS, shuffle=False, epoch=0, drop_last=False):
        recs = model(0, 0, x.to(DEV))
        pred = recs[-1][1].max(1).values.argmax(1).cpu().numpy()
        correct += int((pred == y.numpy()).sum()); tot += len(y)
        if tot >= MAX_TRIALS:
            break
    return correct / tot


# the generator needs prms; grab one canonical prms
_m0, _sd0 = build_model(f"4class_seed101_het_lognormal")
PRMS = dict(torch.load(CK / "4class_seed101_het_lognormal.pt", map_location=DEV, weights_only=False)["prms"])
PRMS.update({"dtype": torch.float, "device": DEV, "cuda": DEV.type == "cuda",
             "task_label_map": src.TASKS["4class"]["task_label_map"], "class_list": list(range(20))})
del _m0


def curve(name, order):
    model, sd = build_model(name)
    return [accuracy(model, sd, order[:k]) for k in LEVELS]


def done_keys():
    if not OUTJL.exists():
        return set()
    return {json.loads(l)["key"] for l in OUTJL.read_text().splitlines() if l.strip()}


def emit(rec):
    with open(OUTJL, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    have = done_keys()
    print(f"device {DEV}; {len(have)} curves already done; levels {LEVELS}; {MAX_TRIALS} trials\n", flush=True)
    jobs = []
    for s in SEEDS_ALL:
        tau = np.array(json.load(open(CACHE / f"4class_seed{s}_het_lognormal.json"))["tau_all_ms"])
        jobs.append((f"het_long_{s}", f"4class_seed{s}_het_lognormal", np.argsort(-tau)))
        jobs.append((f"het_short_{s}", f"4class_seed{s}_het_lognormal", np.argsort(tau)))
        jobs.append((f"hom_rand_{s}", hom_name(s), np.random.default_rng(s).permutation(64)))
    for tag, tms in RUNGS.items():
        for s in SEEDS_ORIG:
            jobs.append((f"fixed{tms:g}_rand_{s}", f"4class_seed{s}_hom_{tag}_lognormal",
                         np.random.default_rng(1000 + s).permutation(64)))
    for i, (key, name, order) in enumerate(jobs, 1):
        if key in have:
            continue
        t0 = time.time()
        try:
            c = curve(name, np.asarray(order))
        except Exception as e:
            print(f"[{i}/{len(jobs)}] {key}: FAILED {type(e).__name__}: {e}", flush=True); continue
        emit({"key": key, "levels": LEVELS, "acc": c})
        print(f"[{i}/{len(jobs)}] {key}: {c[0]:.3f} -> {c[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("dynamical ablation done.")
