"""Fill in the eleven spike caches missing from data_cache/analysis_suite/spk_cache_4ms.

The n=20 clean design needs forty caches: het and hom for each of the twenty seeds. Twenty-nine were already
built by run_oinfo.py. Missing are het for seeds 101 and 202-930, and hom for seed 101. This calls the same
build_cache used for the existing twenty-nine, so every cache in the directory comes from one code path.

For seeds 202-930 the homogeneous arm of the clean design is the no-weight-decay retrain, which is already
cached as 4class_seed{s}_hom_nowd_spk.npy; nothing is rebuilt for it here.

Run: python code/supervisor_extension/build_missing_caches.py
"""
import sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "code" / "analysis_suite"))

import numpy as np
import run_oinfo as R
from data_gen import open_file

# build_cache reads these as module-level globals of run_oinfo; loaded exactly as its own main() does
_u, _t, _l = open_file(str(R.src.SHD_TEST))
R.UNITS, R.TIMES, R.LABELS = list(_u[:]), list(_t[:]), np.array(_l[:])
_u._v_file.close()

JOBS = [(101, "het"), (101, "hom")] + [(s, "het") for s in [202, 210, 340, 440, 550, 660, 710, 820, 930]]


def main():
    for seed, arch in JOBS:
        t0 = time.time()
        out, status = R.build_cache(seed, arch)
        print(f"seed {seed:>4} {arch:<3}  {status:<22} {time.time()-t0:6.1f}s  {out.name}", flush=True)


if __name__ == "__main__":
    main()
