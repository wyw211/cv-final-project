"""FINAL test evaluation with the post-processing setting locked on validation (NEXT_IMPROVEMENTS.md §14):
alpha query expansion, k=2, alpha=3. No other variant is computed on test, nothing is tuned here.
Reports plain (sanity check vs. the earlier test numbers) and alpha-QE for each run.
usage: python tools/eval_test_aqe.py RUN_DIR [RUN_DIR ...]
"""
import sys, os, os.path as osp, json, datetime
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import numpy as np, torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

K, ALPHA = 2, 3.0  # locked on validation
NAMES = ['CARGO', 'CARGO_AG', 'CARGO_AA', 'CARGO_GG']
OUT = 'logs/CARGO/aqe_test'; os.makedirs(OUT, exist_ok=True)


def l2n(x): return x / np.linalg.norm(x, axis=1, keepdims=True)


def aqe(F, nq, k=K, alpha=ALPHA):  # identical to tools/posthoc_val.py
    g = F[nq:]; S = g @ g.T
    idx = np.argsort(-S, axis=1)[:, :k + 1]
    w = np.take_along_axis(S, idx, 1).clip(0) ** alpha
    g2 = l2n((w[..., None] * g[idx]).sum(1))
    S2 = F[:nq] @ g2.T
    qi = np.argsort(-S2, axis=1)[:, :k]
    wq = np.take_along_axis(S2, qi, 1).clip(0) ** alpha
    q2 = l2n(F[:nq] + (wq[..., None] * g2[qi]).sum(1))
    return np.concatenate([q2, g2])


def evaluate(F, P, C, nq):
    S = F[:nq] @ F[nq:].T; aps, r1s = [], []
    for i in range(nq):
        keep = ~((P[nq:] == P[i]) & (C[nq:] == C[i]))
        s, m = S[i][keep], (P[nq:][keep] == P[i])
        if m.sum() == 0: continue
        m = m[np.argsort(-s)]; hits = np.where(m)[0]
        aps.append(np.mean((np.arange(len(hits)) + 1) / (hits + 1))); r1s.append(hits[0] == 0)
    return dict(mAP=100 * float(np.mean(aps)), R1=100 * float(np.mean(r1s)), n_valid_queries=len(aps))


res = {}
for run in sys.argv[1:]:
    tag = osp.basename(run.rstrip('/'))
    cfg = get_cfg(); cfg.merge_from_file(osp.join(run, 'config.yaml'))
    cfg.MODEL.BACKBONE.PRETRAIN = False; cfg.TEST.NUM_WORKERS = 0; cfg.TEST.IMS_PER_BATCH = 64
    model = DefaultTrainer.build_model(cfg); Checkpointer(model).load(osp.join(run, 'model_final.pth')); model.eval()
    res[tag] = {}
    for name in NAMES:
        loader, nq = DefaultTrainer.build_test_loader(cfg, name)
        F, P, C = [], [], []
        with torch.no_grad():
            for b in loader:
                F.append(torch.nn.functional.normalize(model(b).float(), dim=1).cpu())
                P.append(b['targets'].cpu()); C.append(b['camids'].cpu())
        F = torch.cat(F).numpy(); P = torch.cat(P).numpy(); C = torch.cat(C).numpy()
        r = dict(plain=evaluate(F, P, C, nq), aqe=evaluate(aqe(F, nq), P, C, nq))
        res[tag][name] = r
        print(f"{tag:32s} {name:9s} plain {r['plain']['mAP']:6.2f} ({r['plain']['R1']:6.2f})  "
              f"aQE {r['aqe']['mAP']:6.2f} ({r['aqe']['R1']:6.2f})  n={r['plain']['n_valid_queries']}", flush=True)
res['_meta'] = dict(k=K, alpha=ALPHA, time=datetime.datetime.now().isoformat(timespec='seconds'))
json.dump(res, open(osp.join(OUT, 'results.json'), 'w'), indent=1)
