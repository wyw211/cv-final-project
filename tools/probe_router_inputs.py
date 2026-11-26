"""Check the v3 hypothesis: are the aerial/ground router inputs nearly constant?

For a trained checkpoint, on CARGO validation images only, reports per view:
  * mean pairwise cosine among images of the router input candidates
    (view token = official input, inv_features, mean of b1 patches);
  * per-dimension std of each candidate relative to its mean norm;
  * router logit std across images and experts, and the top-2 vs 3rd margin
    (used to size ROUTER_NOISE_STD).

usage: python tools/probe_router_inputs.py --config-file CFG --weights PTH [--max-images 1024] [--out JSON]
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--max-images', type=int, default=1024)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = get_cfg(); cfg.merge_from_file(args.config_file)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = build_model(cfg).eval()
    Checkpointer(model).load(args.weights)
    bb = model.backbone
    feats = {'view_token': [], 'inv_features': [], 'patch_mean': []}
    views = []

    def b1_hook(_m, _inp, out):  # b1 output: [B, 2 + N, D]
        g, v, p = out[:, 0], out[:, 1], out[:, 2:]
        feats['view_token'].append(v.float().cpu())
        feats['inv_features'].append((g - v).float().cpu())
        feats['patch_mean'].append(p.mean(1).float().cpu())
    bb.b1.register_forward_hook(b1_hook)

    loader, _ = build_reid_test_loader(cfg, dataset_name='CARGO_VAL')
    seen = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for batch in loader:
            model(batch)
            views += list(batch['viewids'])
            seen += len(batch['viewids'])
            if seen >= args.max_images:
                break
    X = {k: torch.cat(v) for k, v in feats.items()}
    view = torch.tensor([0 if v == 'Aerial' else 1 for v in views])

    def stats(x):
        t = F.normalize(x, dim=-1); s = t @ t.T; n = len(x)
        return {'mean_pairwise_cos': ((s.sum() - n) / (n * (n - 1))).item(),
                'relative_std': (x.std(0).norm() / x.norm(dim=-1).mean()).item()}

    report = {'weights': args.weights, 'images': seen, 'per_view': {}}
    for vid, name, router in ((0, 'Aerial', bb.sky_router), (1, 'Ground', bb.ground_router)):
        m = view == vid
        entry = {k: stats(x[m]) for k, x in X.items()}
        with torch.no_grad():
            for k in ('view_token', 'patch_mean'):
                logits = router.gate.float().cpu()(X[k][m])  # what this router would see
                top = logits.topk(3, dim=-1).values
                entry[k]['router_logit_std_over_images'] = logits.std(0).mean().item()
                entry[k]['router_logit_std_over_experts'] = logits.std(1).mean().item()
                entry[k]['top2_minus_3rd_margin'] = (top[:, 1] - top[:, 2]).mean().item()
                entry[k]['distinct_top2_sets'] = len({tuple(sorted(r)) for r in logits.topk(2, -1).indices.tolist()})
        report['per_view'][name] = {'images': int(m.sum()), **entry}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
