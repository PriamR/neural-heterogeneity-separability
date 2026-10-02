"""Driver: NuMIT z-scoring, one network per fresh CPU process, with a per-network watchdog.

Uses capture_output (pipes) — NOT inherited stdout — to avoid the Windows handle-inheritance bug that
killed back-to-back children. NuMIT is slower than the shuffle null, so the per-network timeout is
larger. Resumable via the records file.
"""
import sys, subprocess, time, argparse, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_numit.py"
RECORDS = HERE / "outputs" / "numit_zscore_records.jsonl"

SEEDS = [101, 202, 210]
ARCHS = ["het", "hom"]
N_SUBSETS, N_NULL, TAU, T_LIMIT = 50, 20, 1, 128000
PER_NET_TIMEOUT = 9000  # 2.5 h; NuMIT network ~60-90 min, generous headroom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.fresh and RECORDS.exists():
        bak = RECORDS.with_suffix(f".jsonl.bak.{int(time.time())}"); RECORDS.rename(bak)
        print(f"archived old records -> {bak.name}", flush=True)

    t_all = time.time()
    for seed in SEEDS:
        for arch in ARCHS:
            cmd = [sys.executable, "-u", str(RUNNER), "--seed", str(seed), "--arch", arch,
                   "--n-subsets", str(N_SUBSETS), "--n-null", str(N_NULL),
                   "--tau", str(TAU), "--t-limit", str(T_LIMIT)]
            print(f"\n=== {seed}/{arch} (t+{time.time()-t_all:.0f}s) ===", flush=True)
            t0 = time.time(); env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            # Retry on Windows process/DLL-init startup failures (e.g. rc 0xC0000142), which happen
            # under load. The runner is resumable, so a retry just continues where it left off.
            for attempt in range(3):
                time.sleep(3)  # let the OS settle between rapid spawns
                try:
                    r = subprocess.run(cmd, timeout=PER_NET_TIMEOUT, check=False, env=env,
                                       capture_output=True, text=True)
                    if r.stdout:
                        print(r.stdout, end="", flush=True)
                    if r.stderr and r.stderr.strip():
                        print(f"[stderr] {r.stderr.strip()[-1500:]}", flush=True)
                    if r.returncode == 0:
                        print(f"--- {seed}/{arch} finished in {time.time()-t0:.0f}s (rc=0)", flush=True)
                        break
                    print(f"!!! {seed}/{arch} rc={r.returncode} (attempt {attempt+1}/3) — retrying", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"!!! {seed}/{arch} TIMED OUT after {PER_NET_TIMEOUT}s — killed, moving on", flush=True)
                    break
    print(f"\nALL NETWORKS DONE in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
