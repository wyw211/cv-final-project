"""Check CUDA, CARGO training images, and ViSA forward/backward; no test evaluation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from fastreid.config import get_cfg
from fastreid.data.common import CommDataset
from fastreid.data.datasets import CARGO
from fastreid.data.transforms import build_transforms
from fastreid.modeling import build_model
from fastreid.utils.env import seed_all_rng
from fastreid.utils.events import EventStorage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    assert args.steps > 0 and args.batch_size >= 8 and args.batch_size % 4 == 0
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    seed_all_rng(42)
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}', flush=True)
    probe = torch.randn(256, 256, device='cuda')
    assert torch.isfinite(probe @ probe.T).all()
    cfg = get_cfg()
    cfg.merge_from_file(str(ROOT / 'configs/CARGO/visa_windows.yml'))
    weight = ROOT / cfg.MODEL.BACKBONE.PRETRAIN_PATH
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest.startswith('80ecf9dd'), f'Unexpected pretrained weight hash: {digest}'
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = str(weight)
    dataset = CARGO(root=str(ROOT / 'datasets'))
    assert dataset.train and dataset.query and dataset.gallery, 'Missing CARGO split'
    groups = defaultdict(list)
    for item in sorted(dataset.train):
        groups[item[1]].append(item)
    # Choose four images per identity, with both aerial and ground views.
    selected = []
    for items in groups.values():
        aerial = [item for item in items if item[3] == 'Aerial']
        ground = [item for item in items if item[3] == 'Ground']
        if len(aerial) >= 2 and len(ground) >= 2:
            selected.extend(aerial[:2] + ground[:2])
        if len(selected) == args.batch_size:
            break
    assert len(selected) == args.batch_size, 'Insufficient training identities'
    # Keep the full training classifier; only the smoke-test minibatch is small.
    train_set = CommDataset(dataset.train, build_transforms(cfg, is_train=True))
    indices = {item[0]: i for i, item in enumerate(dataset.train)}
    cfg.MODEL.HEADS.NUM_CLASSES = train_set.num_classes
    model = build_model(cfg)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=0.9)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
    reference = model.heads.weight if hasattr(model.heads, 'weight') else next(model.parameters())
    before = reference.detach().clone()
    results = []
    with EventStorage() as storage:
        for step in range(args.steps):
            samples = [train_set[indices[item[0]]] for item in selected]
            batch = {
                'images': torch.stack([s['images'] for s in samples]).cuda(),
                'targets': torch.tensor([s['targets'] for s in samples], device='cuda'),
                'camids': torch.tensor([s['camids'] for s in samples], device='cuda'),
                'viewids': [s['viewids'] for s in samples],
            }
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                losses = model(batch)
                loss = sum(losses.values())
            assert torch.isfinite(loss), f'Nonfinite loss: {losses}'
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all().item() for g in grads), 'Nonfinite gradients'
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            row = {'step': step + 1, 'seconds': time.perf_counter() - started,
                   'losses': {name: value.detach().item() for name, value in losses.items()}}
            results.append(row)
            print(json.dumps(row), flush=True)
            storage.step()
    assert not torch.equal(before, reference.detach()), 'Optimizer did not update model'
    # Check the inference branch on these same TRAIN images only.
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        batch['images'] = torch.stack([s['images'] for s in samples]).cuda()
        features = model(batch)
    assert features.shape[0] == args.batch_size and torch.isfinite(features).all()
    report = {
        'purpose': 'Environment verification only; not benchmark results',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'seed': 42,
        'weight_sha256': digest, 'training_images': len(dataset.train),
        'training_identities': train_set.num_classes,
        'batch_size': args.batch_size, 'parameters': sum(p.numel() for p in model.parameters()),
        'peak_cuda_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
        'inference_feature_shape': list(features.shape), 'steps': results,
        'optimizer_updated': True,
    }
    output = ROOT / 'output' / 'smoke_test.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'PASS: {output}', flush=True)


if __name__ == '__main__':
    main()
