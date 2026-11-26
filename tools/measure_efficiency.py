"""Params / FLOPs / inference latency of trained ViSA models. Uses random input tensors only,
so no dataset image is read (the αQE timing uses random features of the test-set sizes).
FLOPs: torch.utils.flop_counter at batch 1 (executed ops; MoE dispatch is sparse, so this is the
active top-2 compute). Reported as GFLOPs (= 2 x GMACs) and GMACs (fvcore-style "FLOPs").
Latency: fp32, eval, no grad, 50 warm-up + 200 timed runs, CUDA-synchronised, median.
usage: python tools/measure_efficiency.py RUN_DIR [RUN_DIR ...]
"""
import sys, os, os.path as osp, json, time, platform
sys.path.insert(0, '.')
import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.utils.checkpoint import Checkpointer

dev = 'cuda'
torch.backends.cudnn.benchmark = True


def batch(n, view, H, W):
    return {'images': torch.rand(n, 3, H, W, device=dev) * 255,
            'camids': torch.zeros(n, dtype=torch.long, device=dev),
            'viewids': [view] * n}


def timed(fn, warm=50, reps=200):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


res = {'_env': dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version())}
for run in sys.argv[1:]:
    tag = osp.basename(run.rstrip('/'))
    cfg = get_cfg(); cfg.merge_from_file(osp.join(run, 'config.yaml')); cfg.MODEL.BACKBONE.PRETRAIN = False
    model = DefaultTrainer.build_model(cfg); Checkpointer(model).load(osp.join(run, 'model_final.pth'))
    model.eval().to(dev)
    H, W = cfg.INPUT.SIZE_TEST
    r = {'params_M': sum(p.numel() for p in model.parameters()) / 1e6}
    with torch.no_grad():
        for view in ('Aerial', 'Ground'):
            fc = FlopCounterMode(display=False)
            with fc: model(batch(1, view, H, W))
            r[f'GFLOPs_{view}'] = fc.get_total_flops() / 1e9
            r[f'GMACs_{view}'] = fc.get_total_flops() / 2e9
            r[f'latency_bs1_ms_{view}'] = 1e3 * timed(lambda: model(batch(1, view, H, W)))
        t = timed(lambda: model({**batch(64, 'Aerial', H, W), 'viewids': ['Aerial'] * 32 + ['Ground'] * 32}), 10, 50)
        r['throughput_bs64_img_per_s'] = 64 / t
    res[tag] = r
    print(tag, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    del model; torch.cuda.empty_cache()

# αQE cost (numpy, CPU) at the CARGO test-set size (ALL protocol); only the image COUNT is read
def l2n(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
def aqe(F, nq, k=2, alpha=3.0):  # identical to tools/eval_test_aqe.py
    g = F[nq:]; S = g @ g.T
    idx = np.argsort(-S, axis=1)[:, :k + 1]
    w = np.take_along_axis(S, idx, 1).clip(0) ** alpha
    g2 = l2n((w[..., None] * g[idx]).sum(1))
    S2 = F[:nq] @ g2.T
    qi = np.argsort(-S2, axis=1)[:, :k]
    wq = np.take_along_axis(S2, qi, 1).clip(0) ** alpha
    return np.concatenate([l2n(F[:nq] + (wq[..., None] * g2[qi]).sum(1)), g2])
loader, nq = DefaultTrainer.build_test_loader(cfg, 'CARGO'); ng = len(loader.dataset) - nq
with torch.no_grad():
    m2 = DefaultTrainer.build_model(cfg).eval().to(dev); d = m2(batch(1, 'Aerial', H, W)).shape[1]; del m2
F = l2n(np.random.randn(nq + ng, d).astype(np.float32)); ts = []
for _ in range(5):
    t = time.perf_counter(); aqe(F, nq); ts.append(time.perf_counter() - t)
res['_aqe'] = dict(nq=nq, ng=ng, dim=int(d), seconds_median=float(np.median(ts)))
print('aQE', res['_aqe'])
os.makedirs('logs/efficiency', exist_ok=True)
json.dump(res, open('logs/efficiency/results.json', 'w'), indent=1)
