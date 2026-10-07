'''
EXP-17 data protocol: train / test = 80 : 20, no validation split.

All images of the author's release (train + val folders of data/data_isic1718/<ds>) are pooled and divided once,
randomly with a fixed seed (123), stratified by lesion-area tertile (computed on the pooled set, mask >= 128 at the
original resolution, as make_split_701020.py), round(n / 5) of every tertile going to test. Images with identical
pixel content (a duplicate group) always stay on the same side, forced into train: ISIC2017 holds one such pair,
train/ISIC_0012970 and val/ISIC_0012127, with two different masks. ISIC2018 names its files 0.png, 1.png, ... in both
author folders (same names, different images), so its files are renamed tr_<name> / va_<name> by source folder; ISIC2017
keeps its names. Image / mask pairs follow NPY_datasets exactly (sorted images[i] with sorted masks[i]).

  python scripts/make_split_8020.py --create          # once, locally: writes analysis/lists/split_8020_<ds>.json
  python scripts/make_split_8020.py                   # anywhere (also on a rented box): builds the folders from the lists

Output: data/data_isic1718_8020/<isic2017|isic2018>/{train,test}/{images,masks} (copies of the original files).
'''
import os
import sys
import json
import shutil
import hashlib
import argparse

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'data', 'data_isic1718')
DST = os.path.join(REPO, 'data', 'data_isic1718_8020')
LISTS = os.path.join(REPO, 'analysis', 'lists')
DATASETS = {'isic17': 'isic2017', 'isic18': 'isic2018'}
RENAME = {'isic17': False, 'isic18': True}
PREFIX = {'train': 'tr_', 'val': 'va_'}


def pairs(folder):
    imgs = sorted(os.listdir(os.path.join(folder, 'images')))
    msks = sorted(os.listdir(os.path.join(folder, 'masks')))
    if len(imgs) != len(msks):
        sys.exit(f'{folder}: {len(imgs)} images but {len(msks)} masks')
    return list(zip(imgs, msks))


def pooled(ds, sub):
    '''[(source folder, image, mask, destination image, destination mask)] over the author train + val folders.'''
    out = []
    for split in ('train', 'val'):
        for img, msk in pairs(os.path.join(SRC, sub, split)):
            pre = PREFIX[split] if RENAME[ds] else ''
            out.append((split, img, msk, pre + img, pre + msk))
    names = [o[3] for o in out]
    if len(set(names)) != len(names):
        sys.exit(f'{ds}: destination image names collide; enable renaming')
    return out


def pixel_hash(path):
    a = np.array(Image.open(path).convert('RGB'))
    return hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()


def create(ds, sub, seed):
    items = pooled(ds, sub)
    area = np.array([(np.array(Image.open(os.path.join(SRC, sub, s, 'masks', m)).convert('L')) >= 128).mean() for s, _, m, _, _ in items])
    groups = {}
    for i, (s, img, _, _, _) in enumerate(items):
        groups.setdefault(pixel_hash(os.path.join(SRC, sub, s, 'images', img)), []).append(i)
    dup = [g for g in groups.values() if len(g) > 1]
    forced = {i for g in dup for i in g}
    tert = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
    rng = np.random.default_rng(seed)
    test = []
    per_tert = []
    for k in range(3):
        idx = rng.permutation(np.flatnonzero(tert == k))
        n_test = int(round(len(idx) / 5))
        pick = [int(i) for i in idx if int(i) not in forced][:n_test]
        test += pick
        per_tert.append({'tertile': k, 'n': int(len(idx)), 'test': len(pick)})
    test_set = set(test)
    rec = lambda i: list(items[i])
    out = {'dataset': ds, 'seed': seed,
           'rule': 'author train + val pooled -> 80 % train / 20 % test (round(n/5) per lesion-area tertile), '
                   'duplicate-pixel groups forced into train' + ('; files renamed tr_/va_ by source folder' if RENAME[ds] else ''),
           'fields': ['source_folder', 'image', 'mask', 'dest_image', 'dest_mask'],
           'tertile_edges_area_frac': np.quantile(area, [1 / 3, 2 / 3]).tolist(), 'per_tertile': per_tert,
           'duplicate_groups': [[f'{items[i][0]}/{items[i][1]}' for i in g] for g in dup],
           'train': sorted((rec(i) for i in range(len(items)) if i not in test_set), key=lambda r: r[3]),
           'test': sorted((rec(i) for i in test), key=lambda r: r[3])}
    os.makedirs(LISTS, exist_ok=True)
    path = os.path.join(LISTS, f'split_8020_{ds}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=0)
    print(f'{ds}: {len(items)} images -> train {len(out["train"])} | test {len(out["test"])}; per tertile {per_tert}; '
          f'duplicate groups {out["duplicate_groups"]}  -> {path}')


def build(ds, sub):
    spec = json.load(open(os.path.join(LISTS, f'split_8020_{ds}.json')))
    for split in ('train', 'test'):
        for kind in ('images', 'masks'):
            d = os.path.join(DST, sub, split, kind)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        for src, img, msk, dimg, dmsk in spec[split]:
            shutil.copy2(os.path.join(SRC, sub, src, 'images', img), os.path.join(DST, sub, split, 'images', dimg))
            shutil.copy2(os.path.join(SRC, sub, src, 'masks', msk), os.path.join(DST, sub, split, 'masks', dmsk))
        # the pairing must survive the copy: NPY_datasets pairs sorted images with sorted masks
        got = pairs(os.path.join(DST, sub, split))
        want = sorted((r[3], r[4]) for r in spec[split])
        if got != want:
            sys.exit(f'{ds} {split}: image / mask pairing changed after the copy')
    print(f'{ds}: built {DST}/{sub}: train {len(spec["train"])} | test {len(spec["test"])}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--create', action='store_true', help='draw the split and write the list files (once)')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--datasets', default='isic17 isic18')
    a = p.parse_args()
    for ds in a.datasets.split():
        sub = DATASETS[ds]
        if not (os.path.isdir(os.path.join(SRC, sub, 'train', 'images')) and os.path.isdir(os.path.join(SRC, sub, 'val', 'images'))):
            print(f'skip {ds}: {SRC}/{sub}/train or val missing')
            continue
        if a.create:
            create(ds, sub, a.seed)
        build(ds, sub)
    return 0


if __name__ == '__main__':
    sys.exit(main())
