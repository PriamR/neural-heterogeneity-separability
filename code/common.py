import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
CACHE = ROOT / "cache"
NETJSON = CACHE / "network_json"
REPR = CACHE / "representation_clean"
SPK = CACHE / "spk_cache_4ms"
RESULTS = CACHE / "results"
FIGDIR = ROOT / "figures"
TABDIR = ROOT / "tables"
SHD = ROOT / "data" / "shd"

SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_NEW = list(range(1001, 1011))
SEEDS = SEEDS_ORIG + SEEDS_NEW

NOWD_SEEDS = {202, 210, 340, 440, 550, 660, 710, 820, 930}

RUNGS = [2.0, 3.5, 15.0, 35.0]
RUNG_TAG = {2.0: "fixed2ms", 3.5: "fixed3p5ms", 15.0: "fixed15ms", 35.0: "fixed35ms"}


def net_name(seed, arch):
    if arch == "het":
        return f"4class_seed{seed}_het_lognormal"
    if arch == "hom":
        tag = "hom_geommean_nowd" if seed in NOWD_SEEDS else "hom_geommean"
        return f"4class_seed{seed}_{tag}_lognormal"
    if arch == "scaled":
        return f"4class_seed{seed}_hom_scaledinit_inputrec_lognormal"
    return f"4class_seed{seed}_hom_{arch}_lognormal"   # fixed-tau rungs, e.g. arch="fixed15ms"


def ckpt_path(seed, arch):
    name = net_name(seed, arch)
    for sub in ("final_40", "controls"):
        p = CKPT / sub / f"{name}.pt"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def load_record(seed, arch):
    return json.load(open(NETJSON / f"{net_name(seed, arch)}.json"))


def load_records(arch, seeds=None):
    return {s: load_record(s, arch) for s in (seeds or SEEDS)}


def repr_path(seed, arch):
    return REPR / f"repr_seed{seed}_{arch}.npz"


def spk_path(seed, arch):
    if arch == "hom" and seed in NOWD_SEEDS:
        return SPK / f"4class_seed{seed}_hom_nowd_spk.npy"
    return SPK / f"4class_seed{seed}_{arch}_spk.npy"


def vals(records, key, seeds=None):
    return np.array([records[s][key] for s in (seeds or sorted(records))], dtype=float)


for _d in (RESULTS, FIGDIR, TABDIR):
    _d.mkdir(parents=True, exist_ok=True)
