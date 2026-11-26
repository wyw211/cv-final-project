"""Convert a PersonViT (hustvl/PersonViT, LUPerson self-supervised) ViT-B/16 checkpoint
into a plain state dict that ViSA's VisionTransformer_multiview.load_param accepts.

* Loads with weights_only=True (only numpy scalar / dtype / argparse.Namespace are
  allow-listed, needed for the stored training args), so no pickled code runs.
* Keeps the teacher backbone (the official PersonViT fine-tuning uses the teacher),
  strips the 'backbone.' prefix and drops the DINO/iBOT projection heads.
* pos_embed: PersonViT was pretrained at 256x128 with stride 16 (16x8 grid + CLS).
  ViSA uses stride 12 (21x10 grid) and two leading tokens (CLS, view). The grid is
  bilinearly resized to 21x10; the CLS position embedding is used for both leading
  tokens. The result has ViSA's exact shape, so load_param does not resize again
  (its own resizer assumes a square grid, which a 16x8 grid is not).

usage: python -I tools/convert_personvit.py SRC.pth DST.pth [--stride 12] [--size 256 128]
"""
import argparse
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from numpy._core.multiarray import scalar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--stride', type=int, default=12)
    ap.add_argument('--size', type=int, nargs=2, default=[256, 128])
    ap.add_argument('--src-grid', type=int, nargs=2, default=[16, 8])
    a = ap.parse_args()

    safe = [(scalar, 'numpy.core.multiarray.scalar'), np.dtype, argparse.Namespace]
    safe += [type(np.dtype(t)) for t in ('float64', 'float32', 'int64', 'int32', 'bool')]
    with torch.serialization.safe_globals(safe):
        ck = torch.load(a.src, map_location='cpu', weights_only=True)
    teacher = ck['teacher']
    sd = {k[len('backbone.'):]: v.float().contiguous() for k, v in teacher.items() if k.startswith('backbone.')}

    pos = sd['pos_embed']                               # [1, 1 + gh*gw, D]
    gh, gw = a.src_grid
    assert pos.shape[1] == 1 + gh * gw, pos.shape
    H, W = a.size
    ny, nx = (H - 16) // a.stride + 1, (W - 16) // a.stride + 1
    grid = pos[:, 1:].reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(ny, nx), mode='bilinear', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, ny * nx, -1)
    cls = pos[:, :1]
    sd['pos_embed'] = torch.cat([cls, cls, grid], dim=1)  # [1, 2 + ny*nx, D]

    torch.save(sd, a.dst)
    h = hashlib.sha256(open(a.dst, 'rb').read()).hexdigest()
    print(f"epoch {ck.get('epoch')} | {len(sd)} tensors | pos_embed {tuple(sd['pos_embed'].shape)} "
          f"(grid {gh}x{gw} -> {ny}x{nx}) | saved {a.dst} sha256 {h}")


if __name__ == '__main__':
    main()
