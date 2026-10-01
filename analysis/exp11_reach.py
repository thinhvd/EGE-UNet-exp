'''
EXP-11 step 0: how many growth steps T the region completion needs (models/completion.py).

On a fusion-only checkpoint (EXP-10 B1) and the TRAIN images (test transform; the val split is the test set
and is not used for the choice), for every missed deep lesion pixel (lesion, distance to the background
> 10 px, probability < 0.5: the "interior miss" of the readouts) measure how many cells of the 128 grid
separate it from the nearest seed cell (sigmoid(z_base) >= --seed, z_base = final(D5) at 128). One growth
step moves one cell in any of 8 directions, so this is the chessboard distance, and T steps can reach a
pixel only if its distance is <= T. Pixels of an image with no seed at all are never reachable.

Prints and writes the share of interior-miss pixels within reach for T = 0..--max-steps, over all images and
over the large tertile, and the smallest T reaching --target (0.85) of them.

  python analysis/exp11_reach.py --checkpoint results/EGE-UNet-results-exp10/egeunet_isic17_learnable_fuse-sum-shallow3-d8_s42 \
      --dataset isic17 --data-path data/data_isic1718/isic2017 --out results/exp11_step0/reach_isic17.json
'''
import os
import sys
import json
import argparse

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from datasets.dataset import NPY_datasets                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--split', default='train', choices=['train', 'val'])
    p.add_argument('--seed', type=float, default=0.9)
    p.add_argument('--radius', type=float, default=10.0)
    p.add_argument('--max-steps', type=int, default=64)
    p.add_argument('--target', type=float, default=0.85)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--out', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    device = C.pick_device(a.device)
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=(a.split == 'train'))
    ds.transformer = config.test_transformer
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    ckpt = C.resolve_checkpoint(a.checkpoint)
    sd, _ = C.load_state_dict(ckpt)
    model, _ = C.build_model(sd, device, None, strict=True)
    rec = {}
    model.final.register_forward_hook(lambda m, i, o: rec.__setitem__('z', o.detach()))
    big = 10 ** 6
    dists, areas, per_img = [], [], []
    with torch.no_grad():
        for i in range(n):
            img, msk = ds[i]
            _, out = model(img.float()[None].to(device))
            prob = out[0, 0].cpu().numpy()
            g = msk[0].numpy() >= 0.5
            areas.append(float(g.mean()))
            seed = (torch.sigmoid(rec['z'])[0, 0] >= a.seed).cpu().numpy()
            miss = g & (distance_transform_edt(g) > a.radius) & (prob < 0.5) if g.any() else np.zeros_like(g)
            ys, xs = np.nonzero(miss)
            if seed.any():
                d128 = distance_transform_cdt(~seed, metric='chessboard')
                s = seed.shape[0]
                cy = np.rint(ys * (s - 1) / (g.shape[0] - 1)).astype(int)
                cx = np.rint(xs * (s - 1) / (g.shape[1] - 1)).astype(int)
                d = d128[cy, cx]
            else:
                d = np.full(len(ys), big)
            dists.append(d)
            per_img.append({'image': os.path.splitext(os.path.basename(ds.data[i][0]))[0], 'miss_px': int(len(ys)),
                            'has_seed': bool(seed.any()), 'median_steps': float(np.median(d)) if len(d) else float('nan')})
    tert = np.digitize(np.array(areas), np.quantile(np.array(areas), [1 / 3, 2 / 3]))
    out = {'checkpoint': ckpt, 'dataset': a.dataset, 'split': a.split, 'n_images': n, 'seed': a.seed, 'radius': a.radius,
           'target': a.target, 'coverage': {}}
    for group, keep in (('all', np.ones(n, bool)), ('large', tert == 2)):
        d = np.concatenate([dists[i] for i in range(n) if keep[i]] or [np.zeros(0)])
        cov = [float((d <= t).mean()) if len(d) else float('nan') for t in range(a.max_steps + 1)]
        t_star = next((t for t, c in enumerate(cov) if c >= a.target), None)
        no_seed = float((d >= big).mean()) if len(d) else float('nan')
        fin = d[d < big]
        out['coverage'][group] = {'miss_px': int(len(d)), 'share_without_seed': no_seed, 'coverage_by_T': cov, 'T_target': t_star,
                                  'median_steps': float(np.median(fin)) if len(fin) else float('nan'),
                                  'p90_steps': float(np.quantile(fin, 0.9)) if len(fin) else float('nan')}
        print(f'{a.dataset} {a.split} {group}: {len(d)} interior-miss px, {100 * no_seed:.1f} % in images without a seed; '
              f'steps median {out["coverage"][group]["median_steps"]:.0f}, p90 {out["coverage"][group]["p90_steps"]:.0f}; '
              f'within reach at T = 8/16/24/32/40/48: ' + ' / '.join(f'{100 * cov[t]:.1f}' for t in (8, 16, 24, 32, 40, 48)
                                                                   if t <= a.max_steps)
              + f' %; smallest T reaching {100 * a.target:.0f} %: {t_star}')
    out['per_image'] = per_img
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
