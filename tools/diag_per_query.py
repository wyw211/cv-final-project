"""Per-query AP on a VALIDATION protocol, bucketed by query image shape / size / lighting.
Never touches the CARGO test set (refuses dataset names without _VAL).
usage: python tools/diag_per_query.py RUN_DIR DATASET [DATASET ...]"""
import sys, os.path as osp, json
sys.path.insert(0, '.')
import numpy as np, torch
from PIL import Image
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

run, names = sys.argv[1], sys.argv[2:]
assert all('_VAL' in n for n in names), 'validation protocols only'
cfg = get_cfg(); cfg.merge_from_file(osp.join(run, 'config.yaml'))
cfg.MODEL.BACKBONE.PRETRAIN = False; cfg.TEST.NUM_WORKERS = 0; cfg.TEST.IMS_PER_BATCH = 64
model = DefaultTrainer.build_model(cfg); Checkpointer(model).load(osp.join(run, 'model_final.pth')); model.eval()

def ap_of(q, qp, qc, gf, gp, gc):
    d = 1 - gf @ q
    keep = ~((gp == qp) & (gc == qc)); d, m = d[keep], (gp[keep] == qp)
    if m.sum() == 0: return None, None
    o = np.argsort(d); m = m[o]; hits = np.where(m)[0]
    ap = np.mean([(k + 1) / (r + 1) for k, r in enumerate(hits)])
    return ap, float(hits[0] == 0)

out = {}
for name in names:
    loader, nq = DefaultTrainer.build_test_loader(cfg, name)
    F, P, C, paths = [], [], [], []
    with torch.no_grad():
        for b in loader:
            f = model(b); F.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
            P.append(b['targets'].cpu()); C.append(b['camids'].cpu()); paths += list(b['img_paths'])
    F = torch.cat(F).numpy(); P = torch.cat(P).numpy(); C = torch.cat(C).numpy()
    np.savez(osp.join('logs/aa_diag', osp.basename(run.rstrip('/')) + f'_{name}.npz'), F=F, P=P, C=C, paths=np.array(paths), nq=nq)
    rows = []
    for i in range(nq):
        ap, r1 = ap_of(F[i], P[i], C[i], F[nq:], P[nq:], C[nq:])
        if ap is None: continue
        w, h = Image.open(paths[i]).size
        rows.append(dict(path=osp.basename(paths[i]), ap=ap, r1=r1, w=w, h=h,
                         night='night' in osp.basename(paths[i])))
    out[name] = rows
    R = rows; ap = np.array([r['ap'] for r in R]); r1 = np.array([r['r1'] for r in R])
    def show(lbl, m):
        m = np.array(m)
        print(f'  {lbl:28s} n={m.sum():4d}  mAP={100*ap[m].mean():6.2f}  R1={100*r1[m].mean():6.2f}')
    print(f'{name}: {len(R)} queries, mAP {100*ap.mean():.2f}, R1 {100*r1.mean():.2f}')
    show('landscape (w>h)', [r['w'] > r['h'] for r in R])
    show('portrait h/w>=1.5', [r['h'] >= 1.5 * r['w'] for r in R])
    show('between', [r['w'] <= r['h'] < 1.5 * r['w'] for r in R])
    show('short side < 48px', [min(r['w'], r['h']) < 48 for r in R])
    show('short side >= 48px', [min(r['w'], r['h']) >= 48 for r in R])
    show('night', [r['night'] for r in R])
    show('day', [not r['night'] for r in R])
json.dump(out, open(osp.join('logs/aa_diag', osp.basename(run.rstrip('/')) + '.json'), 'w'))
