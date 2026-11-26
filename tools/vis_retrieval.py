"""Retrieval visualisation for the report (final evaluation output only: nothing here is used to choose
or tune anything). Examples are drawn at RANDOM with a fixed seed, not hand-picked.

Phase 1: extract test features of the given runs once and cache them in logs/CARGO/vis_feats/.
Phase 2: figures in logs/CARGO/vis/
  <proto>_final_aqe.png      final model + αQE (k=2, α=3): 3 random R1 successes + 3 random R1 failures
  compare_<proto>.png        baseline (plain) vs final + αQE on the same queries:
                             2 random "baseline wrong -> final right" + 2 random "both wrong"
Each row: query (blue border) | top-10 gallery (green = same ID, red = different ID). Gallery images with
the same ID and camera as the query are removed, as in the evaluation.
usage: python tools/vis_retrieval.py BASELINE_RUN FINAL_RUN
"""
import sys, os, os.path as osp
sys.path.insert(0, '.')
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

BASE, FINAL = sys.argv[1], sys.argv[2]
PROTOS = ['CARGO_AG', 'CARGO_AA', 'CARGO_GG']
FEAT = 'logs/CARGO/vis_feats'; OUT = 'logs/CARGO/vis'
os.makedirs(FEAT, exist_ok=True); os.makedirs(OUT, exist_ok=True)
SEED = 0; TOPK = 10; TW, TH = 64, 128; PAD = 6


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


def extract(run):
    tag = osp.basename(run.rstrip('/'))
    todo = [p for p in PROTOS if not osp.exists(osp.join(FEAT, f'{tag}_{p}.npz'))]
    if not todo: return
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultTrainer
    from fastreid.utils.checkpoint import Checkpointer
    cfg = get_cfg(); cfg.merge_from_file(osp.join(run, 'config.yaml'))
    cfg.MODEL.BACKBONE.PRETRAIN = False; cfg.TEST.NUM_WORKERS = 0; cfg.TEST.IMS_PER_BATCH = 64
    model = DefaultTrainer.build_model(cfg); Checkpointer(model).load(osp.join(run, 'model_final.pth')); model.eval()
    for name in todo:
        loader, nq = DefaultTrainer.build_test_loader(cfg, name)
        F, P, C, paths = [], [], [], []
        with torch.no_grad():
            for b in loader:
                F.append(torch.nn.functional.normalize(model(b).float(), dim=1).cpu())
                P.append(b['targets'].cpu()); C.append(b['camids'].cpu()); paths += list(b['img_paths'])
        np.savez(osp.join(FEAT, f'{tag}_{name}.npz'), F=torch.cat(F).numpy(), P=torch.cat(P).numpy(),
                 C=torch.cat(C).numpy(), paths=np.array(paths), nq=nq)
        print('saved', tag, name, flush=True)
    del model; torch.cuda.empty_cache()


def rank(F, P, C, nq):
    """per query: (top-K gallery indices, their same-ID flags, AP, R1) after junk removal"""
    S = F[:nq] @ F[nq:].T; out = []
    for i in range(nq):
        keep = np.where(~((P[nq:] == P[i]) & (C[nq:] == C[i])))[0]
        o = keep[np.argsort(-S[i][keep])]; m = P[nq:][o] == P[i]
        if m.sum() == 0: out.append(None); continue
        hits = np.where(m)[0]
        out.append(dict(top=o[:TOPK], ok=m[:TOPK], ap=float(np.mean((np.arange(len(hits)) + 1) / (hits + 1))),
                        r1=bool(m[0])))
    return out


def tile(path, color, w=3):
    im = Image.open(path).convert('RGB').resize((TW, TH))
    d = ImageDraw.Draw(im); d.rectangle([0, 0, TW - 1, TH - 1], outline=color, width=w)
    return im


def light(p): return 'night' if 'night' in osp.basename(p).lower() else 'day'


def draw(rows, title, fn):
    """rows: list of (label, query_path, gallery_paths, ok_flags)"""
    font = ImageFont.load_default()
    LW = 150; RH = TH + 2 * PAD + 14
    W = LW + (TOPK + 1) * (TW + PAD) + 3 * PAD; Hh = 24 + RH * len(rows)
    canvas = Image.new('RGB', (W, Hh), 'white'); d = ImageDraw.Draw(canvas)
    d.text((PAD, 6), title, fill='black', font=font)
    for r, (label, q, gs, oks) in enumerate(rows):
        y = 24 + r * RH
        for j, line in enumerate(label.split('\n')): d.text((PAD, y + 20 + 14 * j), line, fill='black', font=font)
        canvas.paste(tile(q, (40, 90, 230), 4), (LW, y + PAD)); d.text((LW, y + PAD + TH + 1), 'query', fill='black', font=font)
        for k, (g, ok) in enumerate(zip(gs, oks)):
            x = LW + (k + 1) * (TW + PAD) + 2 * PAD
            canvas.paste(tile(g, (20, 170, 60) if ok else (220, 30, 30)), (x, y + PAD))
            d.text((x, y + PAD + TH + 1), str(k + 1), fill='black', font=font)
    canvas.save(fn); print('wrote', fn)


def load(run, p):
    d = np.load(osp.join(FEAT, f"{osp.basename(run.rstrip('/'))}_{p}.npz"))
    return d['F'], d['P'], d['C'], d['paths'], int(d['nq'])


def main():
    extract(BASE); extract(FINAL)
    rng = np.random.default_rng(SEED)
    for p in PROTOS:
        Ff, P, C, paths, nq = load(FINAL, p); Fb, Pb, _, pathsb, _ = load(BASE, p)
        assert (P == Pb).all() and (paths == pathsb).all()
        rf = rank(aqe(Ff, nq), P, C, nq); rb = rank(Fb, P, C, nq)
        gp = paths[nq:]
        valid = [i for i in range(nq) if rf[i] is not None]
        def row(i, r, tag):
            return (f'{tag}\nID {P[i]}  {light(paths[i])}\nAP {100 * r["ap"]:.1f}', paths[i], gp[r['top']], r['ok'])
        succ = [i for i in valid if rf[i]['r1']]; fail = [i for i in valid if not rf[i]['r1']]
        pick = lambda L, n: list(rng.choice(L, size=min(n, len(L)), replace=False)) if L else []
        rows = [row(i, rf[i], 'success') for i in pick(succ, 3)] + [row(i, rf[i], 'failure') for i in pick(fail, 3)]
        draw(rows, f'{p}: final model + aQE (k=2, a=3), random R1 successes / failures (seed {SEED}); '
                   f'R1 success {len(succ)}/{len(valid)}', osp.join(OUT, f'{p}_final_aqe.png'))
        fixed = [i for i in valid if not rb[i]['r1'] and rf[i]['r1']]
        both = [i for i in valid if not rb[i]['r1'] and not rf[i]['r1']]
        rows = []
        for i in pick(fixed, 2): rows += [row(i, rb[i], 'baseline'), row(i, rf[i], 'final+aQE')]
        for i in pick(both, 2): rows += [row(i, rb[i], 'baseline'), row(i, rf[i], 'final+aQE')]
        draw(rows, f'{p}: baseline vs final+aQE on the same queries. First {2 * min(2, len(fixed))} rows: '
                   f'fixed by ours ({len(fixed)} queries); rest: both wrong ({len(both)} queries)',
             osp.join(OUT, f'compare_{p}.png'))
        n_night_fail = sum(light(paths[i]) == 'night' for i in fail)
        print(f'{p}: final+aQE R1 failures {len(fail)} (night {n_night_fail}); fixed {len(fixed)}, both wrong {len(both)}, '
              f'broken by ours {sum(rb[i]["r1"] and not rf[i]["r1"] for i in valid)}')


if __name__ == '__main__':
    main()
