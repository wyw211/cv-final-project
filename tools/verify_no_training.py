"""Read-only verification of assets, splits, loader, metrics, and model inference.

No optimizer, backward pass, checkpoint selection, or benchmark evaluation.
Only generated verification reports are written.
"""
from collections import Counter, defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from fastreid.config import get_cfg
from fastreid.data import build_reid_train_loader
from fastreid.data.datasets import CARGO
from fastreid.evaluation.rank import evaluate_rank, IS_CYTHON_AVAI
from fastreid.evaluation import ReidEvaluator
from fastreid.modeling import build_model
from fastreid.utils.compute_dist import build_dist
from fastreid.utils.env import seed_all_rng


def verify_assets():
    weight = ROOT / 'weights/jx_vit_base_p16_224-80ecf9dd.pth'
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest == '80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c'
    base = ROOT / 'datasets/CARGO'
    with zipfile.ZipFile(ROOT / 'downloads/CARGO.zip') as archive:
        entries = {i.filename: (i.file_size, i.CRC) for i in archive.infolist() if not i.is_dir()}
    actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
    assert actual == set(entries), f'Archive mismatch: missing {len(set(entries)-actual)}, extra {len(actual-set(entries))}'
    pattern = re.compile(r'^(train|query|gallery)/Cam(\d+)/Cam(\d+)_(day|night)_(\d+)_(\d+)\.jpg$')

    def inspect(entry):
        name, (size, crc) = entry
        try:
            match = pattern.fullmatch(name)
            assert match is not None, 'Invalid filename layout'
            split, folder_cam, file_cam, lighting, pid, index = match.groups()
            assert folder_cam == file_cam and 1 <= int(file_cam) <= 13, 'Camera mismatch'
            raw = (base / name).read_bytes()
            assert len(raw) == size and zlib.crc32(raw) == crc, 'Size/CRC mismatch'
            with Image.open(io.BytesIO(raw)) as image:
                image.load()  # Decode every image, including test images, only for integrity.
                assert image.width > 0 and image.height > 0 and image.mode == 'RGB'
                shape = f'{image.width}x{image.height}'
            return (split, int(pid), int(file_cam), hashlib.sha256(raw).hexdigest(), shape, None)
        except Exception as error:
            return (None, None, None, None, None, f'{name}: {error}')

    ids = defaultdict(set)
    counts = Counter()
    cameras = defaultdict(Counter)
    shapes = Counter()
    hashes = defaultdict(set)
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, row in enumerate(pool.map(inspect, entries.items()), 1):
            split, pid, cam, digest_image, shape, error = row
            if error:
                errors.append(error)
            else:
                ids[split].add(pid)
                counts[split] += 1
                cameras[split][cam] += 1
                shapes[shape] += 1
                hashes[digest_image].add(split)
            if number % 10000 == 0:
                print(f'Images verified: {number}/{len(entries)}', flush=True)
    overlap = ids['train'] & (ids['query'] | ids['gallery'])
    leaked_hashes = [h for h, splits in hashes.items() if 'train' in splits and len(splits) > 1]
    assert not errors, f'{len(errors)} invalid files: {errors[:10]}'
    assert not overlap, f'Train/test identity overlap: {sorted(overlap)[:10]}'
    assert not leaked_hashes, f'{len(leaked_hashes)} byte-identical train/test images'
    assert counts['train'] == 51451 and len(ids['train']) == 2500
    assert counts['query'] + counts['gallery'] == 51024
    assert ids['query'] <= ids['gallery'], 'Query identity absent from gallery'
    return {'weight_sha256': digest, 'images_verified': len(entries), 'split_images': dict(counts),
            'split_identities': {k: len(v) for k, v in ids.items()},
            'split_camera_images': {k: dict(v) for k, v in cameras.items()},
            'image_dimensions': dict(shapes), 'invalid_files': 0,
            'train_test_identity_overlap': 0, 'train_test_identical_image_hashes': 0,
            'integrity_only_no_test_inference': True}


def verify_metrics():
    # Independently hand-calculated AP: positive matches at ranks 1 and 3 -> (1 + 2/3)/2.
    cases = [
        ('perfect_first', [[.1,.2,.3]], [1], [1,2,3], [0], [1,1,1], 1., 1., 1.),
        ('positive_second', [[.1,.2,.3]], [1], [2,1,3], [0], [1,1,1], 0., .5, .5),
        ('two_positives', [[.1,.2,.3,.4]], [1], [1,2,1,3], [0], [1,1,2,1], 1., 5/6, 2/3),
        ('same_camera_exclusion', [[.1,.2,.3,.4]], [1], [1,2,1,3], [0], [0,1,1,1], 0., .5, .5),
        ('invalid_query_skipped', [[.1,.2,.3],[.2,.1,.3]], [1,99], [1,2,3], [0,0], [1,1,1], 1., 1., 1.),
        ('two_query_average', [[.1,.2,.3,.4],[.1,.2,.3,.4]], [1,2], [1,3,2,4], [0,0], [1,1,1,1], .5, 2/3, 2/3),
    ]
    results = []
    for name, dist, qp, gp, qc, gc, r1, ap, inp in cases:
        cmc, aps, inps = evaluate_rank(*[np.array(v) for v in (dist,qp,gp,qc,gc)], max_rank=3, use_cython=False)
        got = [float(cmc[0]), float(np.mean(aps)), float(np.mean(inps))]
        np.testing.assert_allclose(got, [r1,ap,inp], rtol=1e-6, atol=1e-6, err_msg=name)
        results.append({'case': name, 'rank1': got[0], 'mAP': got[1], 'mINP': got[2], 'passed': True})
    try:
        evaluate_rank(np.array([[.1,.2,.3]]), np.array([99]), np.array([1,2,3]),
                      np.array([0]), np.array([1,1,1]), max_rank=3, use_cython=False)
    except AssertionError:
        results.append({'case': 'all_queries_invalid_rejected', 'passed': True})
    else:
        raise AssertionError('All-invalid queries must be rejected')
    feat = torch.tensor([[1.,0.],[0.,1.]])
    np.testing.assert_allclose(build_dist(feat,feat,'cosine'), [[0.,1.],[1.,0.]], atol=1e-6)
    np.testing.assert_allclose(build_dist(feat,feat,'euclidean'), [[0.,2.],[2.,0.]], atol=1e-6)
    cfg = get_cfg()
    cfg.TEST.METRIC = 'cosine'
    evaluator = ReidEvaluator(cfg, num_query=2)
    evaluator.reset()
    # Two perfect queries, 12 gallery entries; enough for the wrapper's Rank-10 output.
    features = torch.cat((feat, feat, torch.tensor([[-1.,-1.]]).repeat(10,1)))
    evaluator.process({'targets': torch.tensor([1,2,1,2] + list(range(3,13))),
                       'camids': torch.tensor([0,0] + [1]*12)}, features)
    wrapper_result = evaluator.evaluate()
    for metric in ('Rank-1','Rank-5','Rank-10','mAP','mINP'):
        np.testing.assert_allclose(wrapper_result[metric],100.,atol=1e-5,err_msg=metric)
    return {'python_rank_cases': results, 'cython_available': IS_CYTHON_AVAI,
            'cosine_and_squared_euclidean_distance': 'passed', 'synthetic_fixtures_only': True,
            'synthetic_full_evaluator': {k:float(v) for k,v in wrapper_result.items()}}


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_pipeline():
    assert torch.cuda.is_available()
    seed_all_rng(42)
    torch.set_num_threads(8)
    cfg = get_cfg()
    cfg.merge_from_file(str(ROOT / 'configs/CARGO/visa_windows.yml'))
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = str(ROOT / cfg.MODEL.BACKBONE.PRETRAIN_PATH)
    data = CARGO(root=str(ROOT / 'datasets'))
    by_path = {str(Path(row[0]).resolve()): row for row in data.train}
    loader = build_reid_train_loader(cfg)
    batches = []
    sample = None
    try:
        iterator = iter(loader)
        for _ in range(3):
            batch = next(iterator)
            assert tuple(batch['images'].shape) == (16,3,256,128)
            assert batch['images'].device.type == 'cuda' and batch['images'].dtype == torch.float32
            assert torch.isfinite(batch['images']).all() and 0 <= batch['images'].min() <= batch['images'].max() <= 255
            assert sorted(Counter(batch['targets'].tolist()).values()) == [4,4,4,4]
            for i, path in enumerate(batch['img_paths']):
                raw = by_path[str(Path(path).resolve())]
                assert batch['targets'][i].item() == loader.dataset.pid_dict[raw[1]]
                assert batch['camids'][i].item() == loader.dataset.cam_dict[raw[2]]
                assert batch['viewids'][i] == raw[3]
            batches.append({'shape': list(batch['images'].shape), 'device': str(batch['images'].device),
                            'identity_counts': sorted(Counter(batch['targets'].tolist()).values()),
                            'views': dict(Counter(batch['viewids']))})
            if sample is None:
                sample = batch
    finally:
        loader.shutdown()
    cfg.MODEL.HEADS.NUM_CLASSES = loader.dataset.num_classes
    model = build_model(cfg).eval()
    before = state_hash(model)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        features = model(sample)
    assert tuple(features.shape) == (16,1536) and torch.isfinite(features).all()
    after = state_hash(model)
    assert before == after, 'Model parameters or buffers changed during inference'
    assert all(p.grad is None for p in model.parameters())
    return {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'official_loader_batches': batches,
            'feature_shape': list(features.shape), 'features_finite': True,
            'all_parameters_and_buffers_unchanged': before == after,
            'model_state_sha256': after, 'gradients_created': False,
            'optimizer_created': False, 'test_images_sent_to_model': 0}


def main():
    checks = {'assets_and_splits':verify_assets, 'synthetic_metrics':verify_metrics,
              'official_loader_and_inference':verify_pipeline}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', nargs='+', choices=list(checks), default=list(checks))
    args = parser.parse_args()
    out = ROOT / 'output/verification_no_training.json'
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Verification only; no training or test-set benchmark evaluation', 'checks': {}}
    if set(args.checks) != set(checks) and out.exists():
        report = json.loads(out.read_text(encoding='utf-8'))
        for previous in report['checks'].values():
            previous.setdefault('checked_at_utc', report['timestamp_utc'])
    for name in args.checks:
        check = checks[name]
        print(f'Starting {name}', flush=True)
        try:
            report['checks'][name] = {'status':'passed', 'details':check()}
        except Exception as error:
            import traceback
            traceback.print_exc()
            report['checks'][name] = {'status':'failed', 'error':str(error)}
        report['checks'][name]['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
        print(f"{name}: {report['checks'][name]['status']}", flush=True)
    report['status'] = 'passed' if set(report['checks']) == set(checks) and all(x['status']=='passed' for x in report['checks'].values()) else 'failed_or_incomplete'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"{report['status'].upper()}: {out}", flush=True)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    sys.exit(main())
