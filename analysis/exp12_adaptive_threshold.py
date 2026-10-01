'''
EXP-12 supplement (post hoc, not pre-registered): how much pooled DSC a per-image decision threshold could add.

EXP-12 found that inside each test image the model's own probability ranks the missed pale rim far above the
nearby skin (per-image AUC ~0.98) while one global threshold separates them much worse. This measures what that
is worth, on the test images, for each model:
  global 0.5    the model as trained
  oracle        per image, the threshold in --grid that maximizes that image's DSC (uses the ground truth: an
                upper bound, not a usable method)
  otsu          per image, Otsu's threshold on the image's own probability histogram (no ground truth, no
                parameter tuned on the test set); images whose probability map has almost no spread (max < 0.5)
                keep 0.5
Pooled DSC over all images and per size tertile, interior miss of the large tertile and FP.

  python analysis/exp12_adaptive_threshold.py --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --run D0=results/EGE-UNet-results-exp11/egeunet_isic17_learnable_s42 --out results/exp12_info/isic17/adaptive_threshold.csv
'''
import os
import sys
import csv
import argparse

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from datasets.dataset import NPY_datasets                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--grid', default=','.join(f'{t:g}' for t in [0.01, 0.02, 0.05] + [round(0.05 * k, 2) for k in range(2, 20)]))
    p.add_argument('--out', required=True)
    return p.parse_args()


def otsu(prob, bins=256):
    h, edges = np.histogram(prob, bins=bins, range=(0, 1))
    w = h.astype(np.float64) / h.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(w)
    m0 = np.cumsum(w * mids)
    mt = m0[-1]
    var = np.where((w0 > 0) & (w0 < 1), (mt * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12), 0)
    return float(edges[int(np.argmax(var)) + 1])


def counts(pred, g, far_in):
    return np.array([(pred & g).sum(), (pred & ~g).sum(), (~pred & g).sum(), (~pred & far_in).sum()], np.int64)


def dsc(c):
    return 100 * 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)


def main():
    a = parse_args()
    grid = [float(t) for t in a.grid.split(',')]
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=False)
    gts, far, area = [], [], []
    for i in range(len(ds)):
        g = ds[i][1].float()[0].numpy() >= 0.5
        gts.append(g)
        far.append(g & (distance_transform_edt(g) > 10) if g.any() else g)
        area.append(g.mean())
    tert = np.digitize(np.array(area), np.quantile(np.array(area), [1 / 3, 2 / 3]))
    rows = []
    for spec in a.run:
        label, path = spec.split('=', 1)
        sd, _ = C.load_state_dict(C.resolve_checkpoint(path))
        model, _ = C.build_model(sd, 'cpu', None, strict=True)
        per = {k: [] for k in ('global', 'oracle', 'otsu')}
        thr_used = {'oracle': [], 'otsu': []}
        with torch.no_grad():
            for i in range(len(ds)):
                prob = model(ds[i][0].float()[None])[1][0, 0].numpy()
                g, fi = gts[i], far[i]
                per['global'].append(counts(prob >= 0.5, g, fi))
                cs = [counts(prob >= t, g, fi) for t in grid]
                k = int(np.argmax([dsc(c) for c in cs]))
                per['oracle'].append(cs[k])
                thr_used['oracle'].append(grid[k])
                t = otsu(prob) if prob.max() >= 0.5 else 0.5
                per['otsu'].append(counts(prob >= t, g, fi))
                thr_used['otsu'].append(t)
        for rule, cs in per.items():
            cs = np.array(cs)
            row = {'model': label, 'rule': rule, 'dsc_pooled': dsc(cs.sum(0))}
            for k, name in enumerate(('small', 'mid', 'large')):
                row[f'dsc_{name}'] = dsc(cs[tert == k].sum(0))
            row['interior_miss_large'] = int(cs[tert == 2, 3].sum())
            row['fp'] = int(cs[:, 1].sum())
            row['fn'] = int(cs[:, 2].sum())
            if rule in thr_used:
                row['threshold_median'] = float(np.median(thr_used[rule]))
                row['threshold_share_below_0.5'] = float(np.mean(np.array(thr_used[rule]) < 0.5))
            rows.append(row)
            print(f'{a.dataset} {label} {rule:6s}: pooled DSC {row["dsc_pooled"]:.2f} | small {row["dsc_small"]:.2f} mid {row["dsc_mid"]:.2f} '
                  f'large {row["dsc_large"]:.2f} | interior miss (large) {row["interior_miss_large"]} | FP {row["fp"]} FN {row["fn"]}'
                  + (f' | median thr {row["threshold_median"]:.2f}, share < 0.5: {100 * row["threshold_share_below_0.5"]:.0f} %' if rule in thr_used else ''))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + [k for k in rows[-1] if k not in rows[0]], restval='')
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
