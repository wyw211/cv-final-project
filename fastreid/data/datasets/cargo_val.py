# encoding: utf-8
"""Validation split carved out of the CARGO *training* set.

Used for design decisions and pilot runs only. The official CARGO query/gallery
(test) images are never read here, so test data stays reserved for the final
evaluation, as the course rules require.

Split (deterministic, no files needed):
  * identities of datasets/CARGO/train are sorted; random.Random(SEED) samples
    NUM_VAL_IDS of them as validation identities, the rest stay for training;
  * for every validation identity, one aerial and one ground image (when the
    identity has them) are drawn as queries, all its other images form the
    gallery. Query images are never in the gallery.

Protocols mirror fastreid/data/datasets/cargo.py:
  CARGO_VAL     ALL   all queries vs all gallery, per-camera camids
  CARGO_VAL_AG  A->G  aerial queries vs ground gallery, camid = view
  CARGO_VAL_AA  A<->A aerial only
  CARGO_VAL_GG  G<->G ground only
"""

import glob
import os.path as osp
import random
from collections import defaultdict

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['CARGO_VAL', 'CARGO_VAL_AG', 'CARGO_VAL_AA', 'CARGO_VAL_GG', 'cargo_val_split']

SEED = 0
NUM_VAL_IDS = 250


def _parse(img_path):
    name = osp.basename(img_path)
    pid = int(name.split('_')[2])
    cam = int(name.split('_')[0][3:])  # 1..13
    view = 'Aerial' if cam <= 5 else 'Ground'
    return pid, cam, view


def cargo_val_split(train_dir, seed=SEED, num_val_ids=NUM_VAL_IDS):
    """Return (train_paths, query_paths, gallery_paths, val_ids), all sorted."""
    paths = []
    for cam_index in range(13):
        paths += glob.glob(osp.join(train_dir, f'Cam{cam_index + 1}', '*.jpg'))
    paths = sorted(paths)
    assert paths, f'no images under {train_dir}'
    by_pid = defaultdict(list)
    for p in paths:
        by_pid[_parse(p)[0]].append(p)
    rng = random.Random(seed)
    val_ids = sorted(rng.sample(sorted(by_pid), num_val_ids))
    val_set = set(val_ids)
    train_paths = [p for p in paths if _parse(p)[0] not in val_set]
    query, gallery = [], []
    for pid in val_ids:
        imgs = by_pid[pid]
        chosen = []
        for view in ('Aerial', 'Ground'):
            candidates = [p for p in imgs if _parse(p)[2] == view]
            if candidates:
                chosen.append(rng.choice(candidates))
        query += chosen
        gallery += [p for p in imgs if p not in chosen]
    return train_paths, sorted(query), sorted(gallery), val_ids


class _CARGOValBase(ImageDataset):
    dataset_dir = 'CARGO'
    dataset_name = 'cargo'
    query_view = 'all'
    gallery_view = 'all'
    camid_is_view = False

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.data_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.data_dir, 'train')
        train_paths, query_paths, gallery_paths, _ = cargo_val_split(self.train_dir)
        train = self._items(train_paths, is_train=True)
        query = self._items(query_paths, is_train=False, view=self.query_view)
        gallery = self._items(gallery_paths, is_train=False, view=self.gallery_view)
        super().__init__(train, query, gallery, **kwargs)

    def _items(self, paths, is_train, view='all'):
        data = []
        for img_path in paths:
            pid, cam, viewid = _parse(img_path)
            if view != 'all' and viewid != view:
                continue
            camid = (0 if viewid == 'Aerial' else 1) if self.camid_is_view else cam - 1
            if is_train:
                pid = self.dataset_name + '_' + str(pid)
                camid = self.dataset_name + '_' + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data


@DATASET_REGISTRY.register()
class CARGO_VAL(_CARGOValBase):
    """ALL protocol on the validation split; its .train is the reduced train set."""


@DATASET_REGISTRY.register()
class CARGO_VAL_AG(_CARGOValBase):
    query_view = 'Aerial'
    gallery_view = 'Ground'
    camid_is_view = True


@DATASET_REGISTRY.register()
class CARGO_VAL_AA(_CARGOValBase):
    query_view = 'Aerial'
    gallery_view = 'Aerial'


@DATASET_REGISTRY.register()
class CARGO_VAL_GG(_CARGOValBase):
    query_view = 'Ground'
    gallery_view = 'Ground'
