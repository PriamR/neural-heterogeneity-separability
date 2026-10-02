"""One forward pass per learned network over the full SHD test set, caching the tensors the geometry
analyses (CCGP, manifold capacity, memory) consume, at the clean n = 20.

The legacy representation cache in data_cache/representation was built on the ORIGINAL homogeneous nets
(seeds 202..930 carried weight decay 1e-4) and only the ten original seeds. This rebuilds it for all twenty
pairs using the CLEAN homs (the no-WD retrains for 202..930) so CCGP, capacity and memory can be recomputed
at a matched n = 20.

Per (seed, arch) it saves data_cache/representation_clean/repr_seed{seed}_{arch}.npz with:
  rate    [N, 64]        per-trial mean firing rate (the rate code)          -> CCGP, capacity PCA, geometry
  binned  [N, 20, 64]    rate in 20 time-bins (temporal-manifold points)     -> empirical manifold capacity
  hidden  [N, 50, 64]    rate in 50 x 20 ms bins                             -> memory function, temporal dim
  u       [N, 50]        input drive per 20 ms bin (sum over 700 channels)   -> memory function
  y       [N]            4-class label
  pred    [N]            predicted class (argmax of max-over-time membrane)

Reuses the proven clean-repo forward pass from extract_all.py. GPU if available. Resumable.
Run: python code/analysis_suite/extract_geometry.py
"""
import sys, time
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
OUT = REPO / "data_cache" / "representation_clean"
OUT.mkdir(parents=True, exist_ok=True)

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
CONFOUNDED = [202, 210, 340, 440, 550, 660, 710, 820, 930]   # clean hom = no-WD retrain for these
NB20, NB50 = 20, 50


def hom_name(s):
    return (f"4class_seed{s}_hom_geommean_nowd_lognormal" if s in CONFOUNDED
            else f"4class_seed{s}_hom_geommean_lognormal")


def net_name(seed, arch):
    return f"4class_seed{seed}_het_lognormal" if arch == "het" else hom_name(seed)


print("loading SHD test set ...", flush=True)
_u, _t, _l = open_file(str(src.SHD_TEST))
UNITS, TIMES, LABELS = list(_u[:]), list(_t[:]), np.array(_l[:])
_u._v_file.close()
print(f"  {len(LABELS)} trials", flush=True)


@torch.no_grad()
def extract(seed, arch):
    ck = torch.load(CK / f"{net_name(seed, arch)}.pt", map_location=DEV, weights_only=False)
    prms = dict(ck["prms"])
    prms.update({"dtype": torch.float, "device": DEV, "cuda": DEV.type == "cuda"})
    prms["task_label_map"] = src.TASKS["4class"]["task_label_map"]
    prms["class_list"] = list(range(20))
    model = src.RSNN(prms, rec=True).to(DEV)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    nb_steps = int(prms["nb_steps"])
    p20, p50 = nb_steps // NB20, nb_steps // NB50

    rate, binned, hidden, udrive, ys, preds = [], [], [], [], [], []
    for x, y in src.shd_generator(UNITS, TIMES, LABELS, prms, shuffle=False, epoch=0, drop_last=False):
        x = x.to(DEV)
        recs = model(0, 0, x)
        spk = recs[0][2]                                   # [B,T,64] hidden spikes
        mem_out = recs[-1][1]                              # [B,T,4] output membrane
        B = spk.shape[0]
        rate.append(spk.mean(1).cpu().numpy())
        binned.append(spk[:, :NB20 * p20].reshape(B, NB20, p20, 64).mean(2).cpu().numpy())
        hidden.append(spk[:, :NB50 * p50].reshape(B, NB50, p50, 64).mean(2).cpu().numpy())
        udrive.append(x[:, :NB50 * p50].sum(2).reshape(B, NB50, p50).mean(2).cpu().numpy())
        preds.append(mem_out.max(1).values.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    np.savez_compressed(
        OUT / f"repr_seed{seed}_{arch}.npz",
        rate=np.concatenate(rate).astype(np.float32),
        binned=np.concatenate(binned).astype(np.float32),
        hidden=np.concatenate(hidden).astype(np.float32),
        u=np.concatenate(udrive).astype(np.float32),
        y=np.concatenate(ys).astype(np.int64),
        pred=np.concatenate(preds).astype(np.int64),
    )
    del model
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    yv = np.concatenate(ys); pv = np.concatenate(preds)
    return len(yv), float((pv == yv).mean())


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or SEEDS_ALL
    print(f"device = {DEV}, {len(seeds)} seeds x 2 archs\n", flush=True)
    for seed in seeds:
        for arch in ("het", "hom"):
            f = OUT / f"repr_seed{seed}_{arch}.npz"
            if f.exists():
                print(f"seed {seed} {arch}: exists, skip", flush=True); continue
            t0 = time.time()
            n, acc = extract(seed, arch)
            print(f"seed {seed} {arch}: {net_name(seed, arch)}  n={n} acc={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("geometry extraction done.")
