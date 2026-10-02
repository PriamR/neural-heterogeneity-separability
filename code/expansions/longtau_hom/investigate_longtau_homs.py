"""Investigate the long-tau hom anomalies (seeds 1004 at 199.5 ms and 1007 at 28.8 ms), which reach het-level
accuracy with a single tau. Two questions:
 A. Basin structure. Across the 20 learned homs, who went short (~5 ms) vs long, from the ~27 ms geommean init?
 B. Route to accuracy. Does the long-tau hom reach its accuracy via het-like dimensionality/gain (co-adaptation),
    or via a different route (long integration at LOW dimension)? The n90/PR of the 199 ms hom is the key number.
Zero training. Part B uses the GPU."""
import sys, json, os, glob
from pathlib import Path
import numpy as np, torch
REPO = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
sys.path.insert(0, str(REPO / "code" / "training")); sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); src.DEVICE = DEVICE
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd\shd_test.h5")
CK = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"
TT = CK / "tau_trajectories"
for k, v in {"nb_recurrent": 64, "batch_size": 256}.items(): src.BASE_PRMS[k] = v

# ---------------- Part A: basin structure across all 20 homs (no GPU) ----------------
print("===== A. Basin structure across the 20 learned homs =====")
print(f"{'seed':>6} {'init':>6} {'final':>7} {'max':>6} {'dir':>5} {'acc':>7}")
seeds = sorted({os.path.basename(f).split('_')[1].replace('seed','') for f in glob.glob(str(CK/'*_hom_geommean_lognormal.pt'))}, key=lambda x:int(x))
rows=[]
for s in seeds:
    tj = TT / f"4class_seed{s}_hom_learned_tautraj.json"
    if not tj.exists():
        print(f"{s:>6}   (no trajectory)"); continue
    ep = json.load(open(tj))["epochs"]; taus=[e["tau_mem_ms"][0] for e in ep]; accs=[e.get("test_acc") for e in ep]
    init,fin,mx = taus[0], taus[-1], max(taus)
    d = "LONG" if fin > 15 else "short"
    rows.append((s, fin, d))
    print(f"{s:>6} {init:>6.0f} {fin:>7.1f} {mx:>6.0f} {d:>5} {(accs[-1] or float('nan')):>7.3f}")
nlong=sum(1 for _,_,d in rows if d=="LONG")
print(f"\n  {nlong}/{len(rows)} homs settled LONG (>15 ms); the rest fell to the short (~5 ms) basin.")
print("  All start near the ~27 ms geommean init, so both basins are reachable from the same start.")

# ---------------- Part B: route to accuracy (GPU) ----------------
@torch.no_grad()
def fwd(model, prms, U, T, L, n_batches=2):
    model.eval(); nb=int(model.network[0].output_size); ns,ni=int(prms["nb_steps"]),int(prms["nb_inputs"]); bs=int(prms["batch_size"]); inv=1.0/float(prms["time_step"])
    idx=np.where(np.isin(L,prms["class_list"]))[0]; n_batches=min(n_batches,-(-len(idx)//bs)); ch=[]
    for b in range(n_batches):
        bi=idx[bs*b:min(len(idx),bs*(b+1))]; ab=len(bi)
        ta=[np.round(T[i]*inv).astype(np.int64) for i in bi]; ua=[U[i] for i in bi]; ln=np.array([len(a) for a in ta])
        ts=np.concatenate(ta); us=np.concatenate(ua); bc=np.repeat(np.arange(ab),ln); m=ts<ns
        x=torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc[m],ts[m],us[m]])),torch.ones(int(m.sum()),dtype=torch.float32),torch.Size([ab,ns,ni])).to_dense().clamp(max=1.0).to(DEVICE)
        spk=model(0,0,x)[0][2].detach().cpu().numpy(); ch.append(np.transpose(spk,(2,0,1)).reshape(nb,-1).astype(np.uint8))
    return np.concatenate(ch,axis=1)

def geom(spk, stride=4):
    act=np.where(spk.sum(1)>0)[0]
    X=spk[act][:,::stride].T.astype(np.float64); X=X-X.mean(0,keepdims=True)
    ev=np.linalg.eigvalsh((X.T@X)/(X.shape[0]-1))[::-1].clip(min=0); tot=ev.sum()
    n90=int(np.searchsorted(np.cumsum(ev)/tot,0.90)+1) if tot>0 else float("nan")
    pr=(ev.sum()**2)/(np.square(ev).sum()) if tot>0 else float("nan")
    return len(act), float(spk.mean()), n90, float(pr)

def load(seed, role):
    fn=f"4class_seed{seed}_{'het' if role=='het' else 'hom_geommean'}_lognormal.pt"
    ck=torch.load(CK/fn, map_location=DEVICE, weights_only=False); prms=dict(ck["prms"]); prms.update({"dtype":torch.float,"device":DEVICE,"cuda":DEVICE.type=="cuda"})
    m=src.RSNN(prms,rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval()
    sr=float(np.max(np.abs(np.linalg.eigvals(ck["model_state_dict"]["network.0.v"].cpu().numpy()))))
    acc=ck.get("history",{}).get("test_acc",[float("nan")]); acc=acc[-1] if isinstance(acc,list) and acc else float("nan")
    return m, prms, sr, acc

print("\n===== B. Route to accuracy: geometry + gain (seeds 1004, 1007 vs a normal hom 1001, and 202) =====")
print(f"{'seed/role':>14} {'acc':>6} {'active':>7} {'rate':>7} {'n90':>5} {'PR':>6} {'spec_radius':>11}")
print("loading SHD test..."); cache=src.SHDCache(src.SHD_TEST); U,T,L=cache.units,cache.times,cache.labels
for s in ("1004","1007","1001","202"):
    for role in ("het","hom"):
        m,p,sr,acc=load(s,role); a,r,n90,pr=geom(fwd(m,p,U,T,L))
        tag = "  <== LONG-TAU HOM" if (role=="hom" and s in ("1004","1007")) else ""
        print(f"{s+' '+role:>14} {acc:>6.3f} {a:>7} {r:>7.4f} {n90:>5} {pr:>6.2f} {sr:>11.2f}{tag}")
        del m
        if DEVICE.type=="cuda": torch.cuda.empty_cache()
print("\n  READ: if the long-tau hom has LOW n90/PR but high accuracy, it wins by long integration, a different")
print("        route than het's high-dimension route. If it has het-like n90/PR/gain, it found het's regime with")
print("        one tau (co-adaptation), which challenges 'a single tau is rank-limited' more directly.")
