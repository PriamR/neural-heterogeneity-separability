"""Is inhibition why hom does worse? Two zero-training tests.
 #1 Silence-vs-drive: are hom's SILENT neurons the net-INHIBITED ones, or just weakly driven?
 #2 Inhibition lesion: attenuate hom's NEGATIVE weights (lambda: 1->0) and re-measure active-count + accuracy.
    If removing inhibition wakes neurons AND raises accuracy -> inhibition was suppressing hom."""
import sys, json
from pathlib import Path
import numpy as np, torch
REPO = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
sys.path.insert(0, str(REPO / "code" / "training")); sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); src.DEVICE = DEVICE
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd\shd_test.h5")
CK = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"
for k, v in {"nb_recurrent": 64, "batch_size": 256}.items(): src.BASE_PRMS[k] = v
W_HID, V_HID = "network.0.w", "network.0.v"
FN = {"het": "4class_seed{}_het_lognormal.pt", "hom": "4class_seed{}_hom_geommean_lognormal.pt"}

def load(seed, role):
    ck = torch.load(CK / FN[role].format(seed), map_location=DEVICE, weights_only=False)
    prms = dict(ck["prms"]); prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, prms

@torch.no_grad()
def fwd(model, prms, U, T, L, n_batches=2, want_input=False):
    model.eval(); nb_h = int(model.network[0].output_size)
    ns, ni = int(prms["nb_steps"]), int(prms["nb_inputs"]); bs = int(prms["batch_size"]); inv = 1.0/float(prms["time_step"])
    idx = np.where(np.isin(L, prms["class_list"]))[0]; n_batches = min(n_batches, -(-len(idx)//bs)); ch = []; xin = np.zeros(ni)
    nx = 0
    for b in range(n_batches):
        bi = idx[bs*b:min(len(idx), bs*(b+1))]; ab = len(bi)
        ta = [np.round(T[i]*inv).astype(np.int64) for i in bi]; ua = [U[i] for i in bi]; ln = np.array([len(a) for a in ta])
        ts = np.concatenate(ta); us = np.concatenate(ua); bc = np.repeat(np.arange(ab), ln); m = ts < ns
        x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc[m], ts[m], us[m]])), torch.ones(int(m.sum()), dtype=torch.float32),
                                    torch.Size([ab, ns, ni])).to_dense().clamp(max=1.0).to(DEVICE)
        if want_input: xin += x.mean((0,1)).cpu().numpy()*ab; nx += ab
        spk = model(0, 0, x)[0][2].detach().cpu().numpy(); ch.append(np.transpose(spk, (2, 0, 1)).reshape(nb_h, -1).astype(np.uint8))
    spk = np.concatenate(ch, axis=1)
    return (spk, xin/max(nx,1)) if want_input else spk

print("loading SHD test..."); c = src.SHDCache(src.SHD_TEST); U, T, L = c.units, c.times, c.labels

# ---------- #1 silence vs drive ----------
print("\n===== #1  Is silence caused by net INHIBITION, or just weak drive? =====")
for seed in (202, 340):
    for role in ("het", "hom"):
        m, p = load(seed, role); spk, inrate = fwd(m, p, U, T, L, want_input=True)
        Win = m.state_dict()[W_HID].cpu().numpy()                 # 700 x 64
        ff_drive = Win.T @ inrate                                  # expected feedforward current per hidden neuron
        rate = spk.mean(1); active = spk.sum(1) > 0
        sil = ~active
        # net drive of silent vs active neurons
        ds, da = ff_drive[sil], ff_drive[active]
        # correlation of per-neuron drive with output rate
        r = np.corrcoef(ff_drive, rate)[0,1] if rate.std() > 0 else np.nan
        neg_frac_sil = float((ds < 0).mean()) if ds.size else float("nan")
        print(f"  seed{seed} {role}: active {active.sum():2d}/64 | silent-neuron ff-drive: mean {ds.mean():+6.2f} ({neg_frac_sil*100:3.0f}% net-neg) "
              f"| active-neuron ff-drive mean {da.mean():+6.2f} | corr(drive,rate)={r:+.2f}")
        del m
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
print("  READ: if silent neurons are strongly NET-NEGATIVE (inhibited) -> inhibition silences them;")
print("        if they're just near-ZERO (weak) while active ones are positive -> it's a GAIN/drive story, not inhibition.")

# ---------- #2 inhibition lesion on trained hom ----------
print("\n===== #2  Inhibition lesion on trained hom (attenuate NEGATIVE weights, lambda 1->0) =====")
print(f"  {'lambda':>7} {'active':>7} {'rate':>7} {'acc':>7}")
for seed in (202,):
    for lam in [1.0, 0.75, 0.5, 0.25, 0.0]:
        m, p = load(seed, "hom"); sd = m.state_dict()
        for key in (W_HID, V_HID):
            w = sd[key].clone(); w[w < 0] *= lam; sd[key] = w
        m.load_state_dict(sd)
        spk = fwd(m, p, U, T, L)
        acc = src.evaluate_batches(m, p, U, T, L)["acc"]
        print(f"  {lam:>7.2f} {int((spk.sum(1)>0).sum()):>7} {spk.mean():>7.4f} {acc:>7.3f}")
        del m
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
print("  READ: if accuracy RISES and neurons WAKE as inhibition->0, inhibition was suppressing hom; if flat/worse, it wasn't.")
