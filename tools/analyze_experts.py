"""Diagnose whether ETGM experts differentiate, on CARGO validation images only.

Measures, for a trained checkpoint:
  * query diversity: mean pairwise cosine similarity among the 16 expert query
    tokens of one image (1.0 = all identical), for the shared and view branches;
  * cross-attention spread: same statistic right after the experts' cross-attn
    on the first selected expert (official keys -> exactly 1.0);
  * router usage: how often each expert is in the top-k, per router, and its
    normalised entropy (1.0 = perfectly uniform);
  * patch-set overlap: mean Jaccard overlap between the top-k patch sets chosen
    by different queries (official shared set -> 1.0).

usage: python tools/analyze_experts.py --config-file CFG --weights PTH [--max-images 1024] [--out JSON]
"""
import argparse
import json
import math
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


def pairwise_cos(tokens):  # [B, M, D] -> mean off-diagonal cosine
    with torch.autocast('cuda', enabled=False):
        return _pairwise_cos(tokens)


def _pairwise_cos(tokens):
    t = F.normalize(tokens.float(), dim=-1)
    sim = t @ t.transpose(1, 2)
    m = sim.shape[-1]
    return ((sim.sum((1, 2)) - m) / (m * (m - 1))).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--dataset', default='CARGO_VAL')
    parser.add_argument('--max-images', type=int, default=1024)
    parser.add_argument('--out', default=None)
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help='config overrides; with --weights init the model is analysed at initialisation')
    args = parser.parse_args()
    assert args.dataset.startswith('CARGO_VAL'), 'diagnostics must use the validation split'

    cfg = get_cfg(); cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.weights == 'init':
        # ImageNet ViT + freshly initialised ViSA head, as at training start.
        torch.manual_seed(0)
        model = build_model(cfg).eval()
    else:
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS = args.weights
        model = build_model(cfg).eval()
        Checkpointer(model).load(args.weights)
    bb = model.backbone
    E, k_graph = bb.num_experts_per_type, bb.gtop_k

    rec = {'share_q': [], 'view_q': [], 'ca_spread': [], 'patch_overlap_share': [], 'patch_overlap_view': []}
    usage = {name: torch.zeros(E) for name in ('share', 'sky', 'ground')}

    def router_hook(name):
        def hook(_m, _inp, out):
            usage[name] += torch.bincount(out[1].flatten().cpu(), minlength=E).float()
        return hook
    bb.share_router.register_forward_hook(router_hook('share'))
    bb.sky_router.register_forward_hook(router_hook('sky'))
    bb.ground_router.register_forward_hook(router_hook('ground'))

    # Cross-attention spread inside every expert block call.
    rec.update(attn_mass_on_routing=[], attn_entropy_norm=[], attn_l1_between_queries=[])

    def ca_hook(mod, _args, kwargs, out):
        rec['ca_spread'].append(pairwise_cos(out))
        with torch.autocast('cuda', enabled=False):
            qh = mod._separate_heads(mod.q_proj.float()(kwargs['q'].float()), mod.num_heads)
            kh = mod._separate_heads(mod.k_proj.float()(kwargs['k'].float()), mod.num_heads)
            if getattr(mod, 'qk_norm', False):  # mirror CrossAttention.forward
                logits = F.normalize(qh, dim=-1) @ F.normalize(kh, dim=-1).transpose(-1, -2)
                logits = logits * mod.logit_scale.float().clamp(max=math.log(100.0)).exp()
            else:
                logits = qh @ kh.transpose(-1, -2) / math.sqrt(qh.shape[-1])
            a = torch.softmax(logits, -1)  # B,H,M,N
            rec['attn_mass_on_routing'].append(a[..., 0].mean().item())
            if a.shape[-1] > 1:
                ent = -(a.clamp_min(1e-12).log() * a).sum(-1) / math.log(a.shape[-1])
                rec['attn_entropy_norm'].append(ent.mean().item())
                rec['attn_l1_between_queries'].append((a - a.mean(2, keepdim=True)).abs().sum(-1).mean().item())
    for group in (bb.share_experts, bb.sky_experts, bb.ground_experts):
        for expert in group:
            expert.prm.cross_attn.register_forward_hook(ca_hook, with_kwargs=True)

    # Expert outputs = what the GCN consumes; gcn inputs reveal patch sets.
    def gcn_hook(branch):
        def hook(_m, inp, _out):
          with torch.autocast('cuda', enabled=False):
            feat = inp[0]  # [B, M, 1+k, D]: query token + its selected patches
            queries = feat[:, :, 0]
            rec[f'{branch}_q'].append(pairwise_cos(queries))
            patches = feat[:, :, 1:].float()  # [B, M, k, D]
            # Same patch -> identical feature vector. Fraction of query 0's
            # patches that also appear in each other query's set.
            a, b = patches[:, 0], patches[:, 1:]
            dist = torch.cdist(a.unsqueeze(1).expand(-1, b.shape[1], -1, -1).reshape(-1, a.shape[1], a.shape[2]),
                               b.reshape(-1, b.shape[2], b.shape[3]),
                               compute_mode='donot_use_mm_for_euclid_dist')  # [B*(M-1), k, k]
            same = (dist < 1e-3).any(-1).float().mean()
            rec[f'patch_overlap_{branch}'].append(same.item())
        return hook
    bb.share_gcn.register_forward_hook(gcn_hook('share'))
    gcn_view = bb.view_gcn if bb.separate_view_gcn else None
    if gcn_view is not None:
        gcn_view.register_forward_hook(gcn_hook('view'))
    else:  # both branches call share_gcn: hook fires twice, alternate branches
        state = {'n': 0}
        bb.share_gcn._forward_hooks.clear()
        def alternating(m, inp, out):
            gcn_hook('share' if state['n'] % 2 == 0 else 'view')(m, inp, out)
            state['n'] += 1
        bb.share_gcn.register_forward_hook(alternating)

    loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset)
    seen = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for batch in loader:
            model(batch)
            seen += batch['images'].shape[0]
            if seen >= args.max_images:
                break

    def entropy(c):
        p = c / c.sum()
        return float(-(p[p > 0] * p[p > 0].log()).sum() / math.log(len(c)))

    mean = lambda xs: sum(xs) / len(xs) if xs else None
    report = {
        'config': args.config_file, 'weights': args.weights, 'opts': args.opts, 'dataset': args.dataset, 'images': seen,
        'expert_kv': bb.expert_kv, 'graph_per_query': bb.graph_per_query,
        'separate_view_gcn': bb.separate_view_gcn,
        'query_mean_pairwise_cos_share': mean(rec['share_q']),
        'query_mean_pairwise_cos_view': mean(rec['view_q']),
        'cross_attn_mean_pairwise_cos': mean(rec['ca_spread']),
        'attn_mass_on_routing_token': mean(rec['attn_mass_on_routing']),
        'attn_entropy_normalised': mean(rec['attn_entropy_norm']),
        'attn_l1_between_queries': mean(rec['attn_l1_between_queries']),
        'patch_set_overlap_share': mean(rec['patch_overlap_share']),
        'patch_set_overlap_view': mean(rec['patch_overlap_view']),
        'router_usage': {n: c.tolist() for n, c in usage.items()},
        'router_entropy': {n: entropy(c) for n, c in usage.items() if c.sum() > 0},
        'graph_top_k': k_graph,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
