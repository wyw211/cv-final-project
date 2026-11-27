"""alpha-QE combined with k-reciprocal re-ranking, evaluated on VALIDATION protocols only.
Never touches the CARGO test set (refuses dataset names without _VAL).

Reads the cached validation features written by tools/posthoc_val.py (logs/posthoc/<tag>_<name>.npz).
Variants (fixed before running, see NEXT_IMPROVEMENTS.md §17):
  base                         cosine distance of plain features
  aqe k2 a3                    alpha-QE as in tools/posthoc_val.py (sanity check vs §14)
  rerank k1=20 k2=6 l=0.3      FastReID k-reciprocal re-ranking, default parameters
  aqe -> rerank k1=20          alpha-QE features, then re-ranking with default parameters
  aqe -> rerank k1=10          same with k1=10 (k2=6, lambda=0.3)
Re-ranking reuses fastreid.utils.compute_dist.compute_jaccard_distance (the code the test-set
re-ranking run used) and mixes distances like ReidEvaluator: (1 - lambda) * jaccard + lambda * cosine.
usage: python tools/posthoc_rerank_val.py [TAG] [DATASET ...]   (TAG default personvit_experts4)
"""
import sys, os, os.path as osp, json, time
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import numpy as np, torch
from fastreid.utils.compute_dist import compute_jaccard_distance


def l2n(x): return x / np.linalg.norm(x, axis=1, keepdims=True)


def aqe(F, nq, k=2, alpha=3.0):  # identical to tools/posthoc_val.py and tools/eval_test_aqe.py
    g = F[nq:]
    S = g @ g.T
    idx = np.argsort(-S, axis=1)[:, :k + 1]
    w = np.take_along_axis(S, idx, 1).clip(0) ** alpha
    g2 = l2n((w[..., None] * g[idx]).sum(1))
    S2 = F[:nq] @ g2.T
    qi = np.argsort(-S2, axis=1)[:, :k]
    wq = np.take_along_axis(S2, qi, 1).clip(0) ** alpha
    q2 = l2n(F[:nq] + (wq[..., None] * g2[qi]).sum(1))
    return np.concatenate([q2, g2])

tag = sys.argv[1] if len(sys.argv) > 1 else 'personvit_experts4'
names = sys.argv[2:] or ['CARGO_VAL', 'CARGO_VAL_AG', 'CARGO_VAL_AA', 'CARGO_VAL_GG']
assert all('_VAL' in n for n in names), 'validation protocols only'
OUT = f'results/val/posthoc_rerank_{tag}.json'


def evaluate_dist(D, P, C, nq, night):
    """Same rules as posthoc_val.evaluate, but on a query x gallery distance matrix."""
    aps, r1s, nights = [], [], []
    for i in range(nq):
        keep = ~((P[nq:] == P[i]) & (C[nq:] == C[i]))
        d, m = D[i][keep], (P[nq:][keep] == P[i])
        if m.sum() == 0: continue
        m = m[np.argsort(d, kind='stable')]; hits = np.where(m)[0]
        aps.append(np.mean((np.arange(len(hits)) + 1) / (hits + 1))); r1s.append(hits[0] == 0); nights.append(night[i])
    aps, r1s, nights = map(np.array, (aps, r1s, nights))
    return dict(mAP=100 * aps.mean(), R1=100 * r1s.mean(),
                night_mAP=100 * aps[nights].mean(), night_R1=100 * r1s[nights].mean(),
                day_mAP=100 * aps[~nights].mean(), n_valid=int(len(aps)), n_night=int(nights.sum()))


def cosine_dist(X, nq): return 1 - X[:nq] @ X[nq:].T


def rerank(X, nq, k1, k2=6, lam=0.3):
    jac = compute_jaccard_distance(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                                   k1=k1, k2=k2, search_option=3)[:nq, nq:]
    return jac * (1 - lam) + cosine_dist(X, nq) * lam


res = {}
for name in names:
    d = np.load(osp.join('logs/posthoc', f'{tag}_{name}.npz'))
    F, P, C, nq = d['F'].astype(np.float32), d['P'], d['C'], int(d['nq'])
    night = np.array(['night' in osp.basename(p) for p in d['paths']])  # reporting only
    Fq = aqe(F, nq, 2, 3.0).astype(np.float32)
    V = {'base': lambda: cosine_dist(F, nq),
         'aqe k2 a3': lambda: cosine_dist(Fq, nq),
         'rerank k1=20 k2=6 l=0.3': lambda: rerank(F, nq, 20),
         'aqe -> rerank k1=20 k2=6 l=0.3': lambda: rerank(Fq, nq, 20),
         'aqe -> rerank k1=10 k2=6 l=0.3': lambda: rerank(Fq, nq, 10)}
    res[name] = {}
    print(f'\n== {name}  (nq={nq}, N={len(F)})', flush=True)
    print(f"{'variant':32s} {'mAP':>6s} {'R1':>6s} {'nightmAP':>8s} {'nightR1':>7s} {'daymAP':>6s} {'sec':>5s}")
    for v, fn in V.items():
        t = time.time(); r = evaluate_dist(fn(), P, C, nq, night); r['sec'] = round(time.time() - t, 1)
        res[name][v] = r
        print(f"{v:32s} {r['mAP']:6.2f} {r['R1']:6.2f} {r['night_mAP']:8.2f} {r['night_R1']:7.2f} {r['day_mAP']:6.2f} {r['sec']:5.1f}", flush=True)

os.makedirs(osp.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('saved', OUT)
