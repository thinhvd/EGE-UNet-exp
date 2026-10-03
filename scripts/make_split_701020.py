'''
EXP-14 data protocol: train / val / test = 70 : 10 : 20 instead of the author's train / val = 70 : 30 with the val
split doubling as the test set.

The train split is kept as it is. The author's val split is divided into a new val (1/3, used for checkpoint
selection and for every choice made before training) and a test split (2/3, never looked at before the final
evaluation). The division is random with a fixed seed, stratified by lesion-area tertile so both parts cover small
and large lesions alike. Image / mask pairs follow NPY_datasets exactly (sorted images[i] with sorted masks[i]).

  python scripts/make_split_701020.py --create          # once, locally: writes analysis/lists/split_701020_<ds>.json
  python scripts/make_split_701020.py                   # anywhere (also on a rented box): builds the folders
                                                        # from the committed lists

Output: data/data_isic1718_701020/<isic2017|isic2018>/{train,val,test}/{images,masks} (copies of the original files).
'''
import os
import sys
import json
import shutil
import argparse

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'data', 'data_isic1718')
DST = os.path.join(REPO, 'data', 'data_isic1718_701020')
LISTS = os.path.join(REPO, 'analysis', 'lists')
DATASETS = {'isic17': 'isic2017', 'isic18': 'isic2018'}


def pairs(folder):
    imgs = sorted(os.listdir(os.path.join(folder, 'images')))
    msks = sorted(os.listdir(os.path.join(folder, 'masks')))
    if len(imgs) != len(msks):
        sys.exit(f'{folder}: {len(imgs)} images but {len(msks)} masks')
    return list(zip(imgs, msks))


def create(ds, sub, seed):
    val = pairs(os.path.join(SRC, sub, 'val'))
    area = np.array([(np.array(Image.open(os.path.join(SRC, sub, 'val', 'masks', m)).convert('L')) >= 128).mean() for _, m in val])
    tert = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
    rng = np.random.default_rng(seed)
    new_val, test = [], []
    for k in range(3):
        idx = rng.permutation(np.flatnonzero(tert == k))
        n_val = int(round(len(idx) / 3))
        new_val += idx[:n_val].tolist()
        test += idx[n_val:].tolist()
    out = {'dataset': ds, 'seed': seed, 'rule': 'author val split -> 1/3 val + 2/3 test, stratified by lesion-area tertile',
           'n_train': len(pairs(os.path.join(SRC, sub, 'train'))),
           'val': [list(val[i]) for i in sorted(new_val)], 'test': [list(val[i]) for i in sorted(test)]}
    os.makedirs(LISTS, exist_ok=True)
    path = os.path.join(LISTS, f'split_701020_{ds}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=0)
    print(f'{ds}: train {out["n_train"]} | val {len(out["val"])} | test {len(out["test"])}  -> {path}')


def build(ds, sub):
    spec = json.load(open(os.path.join(LISTS, f'split_701020_{ds}.json')))
    src_val = os.path.join(SRC, sub, 'val')
    plan = {'train': [(os.path.join(SRC, sub, 'train'), p) for p in pairs(os.path.join(SRC, sub, 'train'))],
            'val': [(src_val, tuple(p)) for p in spec['val']], 'test': [(src_val, tuple(p)) for p in spec['test']]}
    for split, items in plan.items():
        for kind in ('images', 'masks'):
            d = os.path.join(DST, sub, split, kind)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        for folder, (img, msk) in items:
            shutil.copy2(os.path.join(folder, 'images', img), os.path.join(DST, sub, split, 'images', img))
            shutil.copy2(os.path.join(folder, 'masks', msk), os.path.join(DST, sub, split, 'masks', msk))
        # the pairing must survive the copy: NPY_datasets pairs sorted images with sorted masks
        got = pairs(os.path.join(DST, sub, split))
        want = sorted((i, m) for _, (i, m) in items)
        if got != want:
            sys.exit(f'{ds} {split}: image / mask pairing changed after the copy')
    print(f'{ds}: built {DST}/{sub}: train {len(plan["train"])} | val {len(plan["val"])} | test {len(plan["test"])}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--create', action='store_true', help='draw the split and write the list files (once)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--datasets', default='isic17 isic18')
    a = p.parse_args()
    for ds in a.datasets.split():
        sub = DATASETS[ds]
        if not os.path.isdir(os.path.join(SRC, sub, 'val', 'images')):
            print(f'skip {ds}: {SRC}/{sub} missing')
            continue
        if a.create:
            create(ds, sub, a.seed)
        build(ds, sub)
    return 0


if __name__ == '__main__':
    sys.exit(main())
