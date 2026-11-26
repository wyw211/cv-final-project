"""Training-free post-processing of ReID features, evaluated on VALIDATION protocols only.
Never touches the CARGO test set (refuses dataset names without _VAL).

Step 1 (extract): features of a run's model_final.pth, plain and horizontally flipped, plus the
mean luminance of every image (computed from pixels, not from file names).
Step 2 (evaluate): variants
  base          cosine distance of plain features
  flip          mean of plain + flipped features
  illum-center  subtract the mean feature of the image's illumination group (dark / bright by
                luminance threshold), computed over the query+gallery set, re-normalise
  cam-center    same with camera id groups
  aqe           alpha query expansion on the gallery (k neighbours, weight sim^alpha)
usage: python tools/posthoc_val.py RUN_DIR [DATASET ...]
"""
import sys, os, os.path as osp, json, itertools
sys.path.insert(0, '.')
import numpy as np, torch
from PIL import Image

run = sys.argv[1]
names = sys.argv[2:] or ['CARGO_VAL', 'CARGO_VAL_AG', 'CARGO_VAL_AA', 'CARGO_VAL_GG']
assert all('_VAL' in n for n in names), 'validation protocols only'
tag = osp.basename(run.rstrip('/'))
os.makedirs('logs/posthoc', exist_ok=True)


def extract():
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultTrainer
    from fastreid.utils.checkpoint import Checkpointer
    cfg = get_cfg(); cfg.merge_from_file(osp.join(run, 'config.yaml'))
    cfg.MODEL.BACKBONE.PRETRAIN = False; cfg.TEST.NUM_WORKERS = 0; cfg.TEST.IMS_PER_BATCH = 64
    model = DefaultTrainer.build_model(cfg); Checkpointer(model).load(osp.join(run, 'model_final.pth')); model.eval()
    for name in names:
        out = osp.join('logs/posthoc', f'{tag}_{name}.npz')
        if osp.exists(out): continue
        loader, nq = DefaultTrainer.build_test_loader(cfg, name)
        F, G, P, C, paths = [], [], [], [], []
        with torch.no_grad():
            for b in loader:
                img = b['images'].clone()
                F.append(torch.nn.functional.normalize(model(b).float(), dim=1).cpu())
                b['images'] = img.flip(3)  # preprocess normalises in place, so give it a fresh tensor
                G.append(torch.nn.functional.normalize(model(b).float(), dim=1).cpu())
                P.append(b['targets'].cpu()); C.append(b['camids'].cpu()); paths += list(b['img_paths'])
        lum = np.array([np.asarray(Image.open(p).convert('L'), dtype=np.float32).mean() / 255 for p in paths])
        np.savez(out, F=torch.cat(F).numpy(), G=torch.cat(G).numpy(), P=torch.cat(P).numpy(),
                 C=torch.cat(C).numpy(), L=lum, paths=np.array(paths), nq=nq)
        print('saved', out, flush=True)


def l2n(x): return x / np.linalg.norm(x, axis=1, keepdims=True)


def evaluate(F, P, C, nq, night):
    q, g = F[:nq], F[nq:]
    S = q @ g.T
    aps, r1s, nights = [], [], []
    for i in range(nq):
        keep = ~((P[nq:] == P[i]) & (C[nq:] == C[i]))
        s, m = S[i][keep], (P[nq:][keep] == P[i])
        if m.sum() == 0: continue
        m = m[np.argsort(-s)]; hits = np.where(m)[0]
        aps.append(np.mean((np.arange(len(hits)) + 1) / (hits + 1))); r1s.append(hits[0] == 0); nights.append(night[i])
    aps, r1s, nights = map(np.array, (aps, r1s, nights))
    return dict(mAP=100 * aps.mean(), R1=100 * r1s.mean(),
                night_mAP=100 * aps[nights].mean(), night_R1=100 * r1s[nights].mean(),
                day_mAP=100 * aps[~nights].mean(), n_night=int(nights.sum()))


def center(F, groups, a=1.0):
    F = F.copy()
    for gid in np.unique(groups):
        m = groups == gid
        F[m] -= a * F[m].mean(0)
    return l2n(F)


def aqe(F, nq, k=2, alpha=3.0):
    g = F[nq:]
    S = g @ g.T
    idx = np.argsort(-S, axis=1)[:, :k + 1]
    w = np.take_along_axis(S, idx, 1).clip(0) ** alpha
    g2 = l2n((w[..., None] * g[idx]).sum(1))
    S2 = F[:nq] @ g2.T  # query side expanded against expanded gallery
    qi = np.argsort(-S2, axis=1)[:, :k]
    wq = np.take_along_axis(S2, qi, 1).clip(0) ** alpha
    q2 = l2n(F[:nq] + (wq[..., None] * g2[qi]).sum(1))
    return np.concatenate([q2, g2])


def main():
    extract()
    res = {}
    for name in names:
        d = np.load(osp.join('logs/posthoc', f'{tag}_{name}.npz'))
        F, G, P, C, L, nq = d['F'], d['G'], d['P'], d['C'], d['L'], int(d['nq'])
        night_label = np.array(['night' in osp.basename(p) for p in d['paths']])  # used only for reporting
        V = {'base': F, 'flip': l2n(F + G)}
        for t in (0.20, 0.25, 0.30):
            dark = L < t
            agree = (dark == night_label).mean()
            for a in (0.5, 1.0):
                V[f'illum-center t={t} a={a} (agree {100*agree:.0f}%)'] = center(F, dark.astype(int), a)
        for a in (0.5, 1.0):
            V[f'cam-center a={a}'] = center(F, C, a)
        for k, al in itertools.product((1, 2, 3), (1.0, 3.0)):
            V[f'aqe k={k} alpha={al}'] = aqe(F, nq, k, al)
        res[name] = {}
        print(f'\n== {name}  (nq={nq})')
        print(f"{'variant':44s} {'mAP':>6s} {'R1':>6s} {'nightmAP':>8s} {'nightR1':>7s} {'daymAP':>6s}")
        for v, X in V.items():
            r = evaluate(X, P, C, nq, night_label); res[name][v] = r
            print(f"{v:44s} {r['mAP']:6.2f} {r['R1']:6.2f} {r['night_mAP']:8.2f} {r['night_R1']:7.2f} {r['day_mAP']:6.2f}")
    json.dump(res, open(osp.join('logs/posthoc', f'{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
