"""Timing of a GPU (torch) implementation of the locked αQE (k=2, α=3), equivalent to the numpy one in
tools/eval_test_aqe.py, at the CARGO test-set size with random unit features. Checks equality on a small set."""
import sys, time; sys.path.insert(0, '.')
import numpy as np, torch
def l2n(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
def aqe_np(F, nq, k=2, alpha=3.0):
    g = F[nq:]; S = g @ g.T
    idx = np.argsort(-S, axis=1)[:, :k + 1]
    w = np.take_along_axis(S, idx, 1).clip(0) ** alpha
    g2 = l2n((w[..., None] * g[idx]).sum(1))
    S2 = F[:nq] @ g2.T
    qi = np.argsort(-S2, axis=1)[:, :k]
    wq = np.take_along_axis(S2, qi, 1).clip(0) ** alpha
    return np.concatenate([l2n(F[:nq] + (wq[..., None] * g2[qi]).sum(1)), g2])
def aqe_t(F, nq, k=2, alpha=3.0, chunk=4096):
    n = torch.nn.functional.normalize; g = F[nq:]; outs = []
    for i in range(0, g.shape[0], chunk):
        s, idx = (g[i:i + chunk] @ g.T).topk(k + 1, dim=1)
        outs.append(((s.clamp(min=0) ** alpha)[..., None] * g[idx]).sum(1))
    g2 = n(torch.cat(outs), dim=1)
    s, qi = (F[:nq] @ g2.T).topk(k, dim=1)
    q2 = n(F[:nq] + ((s.clamp(min=0) ** alpha)[..., None] * g2[qi]).sum(1), dim=1)
    return torch.cat([q2, g2])
rng = np.random.default_rng(0)
X = l2n(rng.standard_normal((2000, 64)).astype(np.float32))
diff = np.abs(aqe_np(X, 100) - aqe_t(torch.from_numpy(X).cuda(), 100).cpu().numpy()).max()
print('max |numpy - torch| on 2000x64:', diff)
nq, ng, d = 312, 50712, 1536
F = torch.nn.functional.normalize(torch.randn(nq + ng, d, device='cuda'), dim=1)
aqe_t(F, nq); torch.cuda.synchronize(); ts = []
for _ in range(5):
    t = time.perf_counter(); aqe_t(F, nq); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
print(f'GPU αQE nq={nq} ng={ng} d={d}: median {np.median(ts):.3f} s')
