'''
EXP-6: fix the weight of each extra loss term BEFORE training, without touching the test set.

On an already trained baseline checkpoint, run the TRAIN images (test transform, no augmentation) in
batches. For each batch take the gradient, with respect to the final probability map u, of
  - BceDice(u, g), the part of the original loss that acts on the final output, and
  - each extra term of contour_losses.py.
A term's weight is set so that its median (over batches) gradient norm is `--ratio` (default 0.25) times
BceDice's: at the point where training ends up, the new term pushes a quarter as hard as the loss it is
added to. The median, not the mean: BCE on probabilities gives a huge gradient (-g/u) on the few pixels
where the model is confidently wrong, so the mean of the BceDice norm is set by a handful of batches
(ISIC18: mean 95 vs ISIC17 0.075 on the EXP-5 baselines).

  python analysis/calibrate_loss_weights.py --checkpoint results/EGE-UNet-results-exp5/egeunet_isic17_learnable_s42 \
      --dataset isic17 --data-path data/data_isic1718/isic2017 --out results/.../loss_weights_isic17.json
'''
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TERMS = ['tv', 'tv_match', 'area', 'bl', 'snbl']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='baseline run dir or .pth')
    p.add_argument('--dataset', default='isic17', choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--ratio', type=float, default=0.25)
    p.add_argument('--snbl-tau', type=float, default=3.0)
    p.add_argument('--area-delta', type=float, default=0.05)
    p.add_argument('--limit', type=int, default=0, help='use only the first N train images (0 = all)')
    p.add_argument('--out', default=None, help='json path for the table')
    return p.parse_args()


def grad_norm(loss_fn, u, g):
    u = u.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(loss_fn(u, g), u)
    return float(grad.norm())


def main():
    a = parse_args()
    from torch.utils.data import DataLoader
    from analysis import common as C
    from datasets.dataset import NPY_datasets
    from utils import BceDiceLoss
    import contour_losses as L

    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=True)
    ds.transformer = config.test_transformer            # no augmentation
    if a.limit:
        ds.data = ds.data[:a.limit]
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    ckpt = C.resolve_checkpoint(a.checkpoint)
    sd, _ = C.load_state_dict(ckpt)
    model, _ = C.build_model(sd, 'cpu', None, strict=True)
    model.eval()

    terms = {'tv': L.TVLength(), 'tv_match': L.TVMatch(), 'area': L.LogArea(a.area_delta),
             'bl': L.BoundaryLoss(), 'snbl': L.ScaleNormBoundaryLoss(a.snbl_tau)}
    base = BceDiceLoss(wb=1, wd=1)
    norms = {k: [] for k in ['bcedice'] + TERMS}
    values = {k: [] for k in ['bcedice'] + TERMS}
    for img, msk in loader:
        with torch.no_grad():
            _, u = model(img.float())
        g = msk.float()
        for name, fn in [('bcedice', base)] + list(terms.items()):
            norms[name].append(grad_norm(fn, u, g))
            with torch.no_grad():
                values[name].append(float(fn(u, g)))
    ref = float(np.median(norms['bcedice']))
    table = {'checkpoint': ckpt, 'dataset': a.dataset, 'n_images': len(ds), 'ratio': a.ratio,
             'snbl_tau': a.snbl_tau, 'area_delta': a.area_delta,
             'bcedice': {'median_grad_norm': ref, 'mean_grad_norm': float(np.mean(norms['bcedice'])),
                         'mean_value': float(np.mean(values['bcedice']))}, 'terms': {}}
    print(f'{a.dataset}: {len(ds)} train images, checkpoint {ckpt}')
    print(f'  BceDice(final output): median |grad| {ref:.4g} (mean {np.mean(norms["bcedice"]):.4g}), '
          f'mean value {np.mean(values["bcedice"]):.4g}')
    print(f'  {"term":9s} {"median |grad|":>13s} {"mean |grad|":>12s} {"mean value":>11s} {"weight":>10s}')
    for name in TERMS:
        n = float(np.median(norms[name]))
        w = a.ratio * ref / n
        table['terms'][name] = {'median_grad_norm': n, 'mean_grad_norm': float(np.mean(norms[name])),
                                'mean_value': float(np.mean(values[name])), 'weight': w}
        print(f'  {name:9s} {n:13.4g} {np.mean(norms[name]):12.4g} {np.mean(values[name]):11.4g} {w:10.4g}')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(table, f, indent=2)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    sys.exit(main())
