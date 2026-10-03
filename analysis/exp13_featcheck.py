'''
EXP-13 CPU check (TRAIN images only): can the network learn edges that separate the dark-core edge from the weak GT
boundary when the edge loss is allowed to reach its features?

From the fusion-only checkpoint of EXP-11 (D1), fine-tune the whole network plus a fresh propagation module
('pair', tau / kappa / beta of step 0) with the original segmentation loss + the edge-affinity loss
(contour_losses.WithPropagationAux) for --epochs on 80 % of the train images (train transform, i.e. with the
augmentation of training), then read on the other 20 % (test transform):
  - cut-AUC of 1 - S for X_weak (edges crossing the GT contour whose lesion cell has p_base < 0.5) vs LL_inner
    (both cells lesion, crossing the base contour, > 3 cells inside the GT) and the per-image ranking accuracy;
  - pooled DSC with the propagation on and off, growth precision, coverage of the interior miss.
--feature-grad lets the edge loss reach D5 and the fusion projections (z_base stays detached); without it the edge
head reads detached features, as in the planned arms. Same split, order, seed and steps for both.

  python analysis/exp13_featcheck.py --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --checkpoint results/EGE-UNet-results-exp11/egeunet_isic17_learnable_fuse-sum-shallow3-d8_s42 \
      --feature-grad --out results/exp13_featcheck/isic17_feat.json
'''
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from analysis.exp13_mechanism import edge_groups                   # noqa: E402
from analysis.exp11_mechanism import auc                           # noqa: E402
from datasets.dataset import NPY_datasets                          # noqa: E402
from models.egeunet import EGEUNet                                 # noqa: E402
import contour_losses as CL                                        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--feature-grad', action='store_true')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--tau', type=float, default=0.3)
    p.add_argument('--kappa', type=float, default=5.0)
    p.add_argument('--beta', type=float, default=0.5)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def evaluate(model, ds, idx):
    '''Edge readouts and DSC with the propagation on / off on the held-out train images.'''
    model.eval()
    rec = {}
    h1 = model.final.register_forward_hook(lambda m, i, o: rec.__setitem__('z', o.detach()))
    h2 = model.propagate.register_forward_hook(lambda m, i, o: rec.__setitem__('s', torch.sigmoid(o['s_logit'].detach())))
    xw, li, rank = [], [], []
    on, off = np.zeros(3), np.zeros(3)
    flip = flip_in = miss = miss_cov = 0
    with torch.no_grad():
        for i in idx:
            img, msk = ds[i]
            _, p = model(img.float()[None])
            g = msk[0].numpy() >= 0.5
            z = rec['z']
            pb = torch.sigmoid(F.interpolate(z, scale_factor=(2, 2), mode='bilinear', align_corners=True))[0, 0].numpy()
            pf = p[0, 0].numpy()
            s = rec['s'][0]
            groups = edge_groups(g, (torch.sigmoid(z) >= 0.5)[0, 0])
            xw.append(s[groups['X_weak']].numpy())
            li.append(s[groups['LL_inner']].numpy())
            if groups['X_weak'].any() and groups['LL_inner'].any():
                rank.append(float(s[groups['LL_inner']].mean() > s[groups['X_weak']].mean()))
            for acc, pr in ((on, pf >= 0.5), (off, pb >= 0.5)):
                acc += [(pr & g).sum(), (pr & ~g).sum(), (~pr & g).sum()]
            fl = (pf >= 0.5) & (pb < 0.5)
            ms = g & ((distance_transform_edt(g) > 10) if g.any() else g) & (pb < 0.5)
            flip += int(fl.sum())
            flip_in += int((fl & g).sum())
            miss += int(ms.sum())
            miss_cov += int((fl & ms).sum())
    h1.remove()
    h2.remove()
    xw, li = np.concatenate(xw), np.concatenate(li)
    dsc = lambda c: 100 * 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)
    return {'cut_auc': auc(1 - xw, 1 - li), 'rank_acc': float(np.mean(rank)) if rank else float('nan'), 'rank_images': len(rank),
            'mean_S_x_weak': float(xw.mean()), 'mean_S_ll_inner': float(li.mean()),
            'dsc_propagation_on': dsc(on), 'dsc_propagation_off': dsc(off), 'delta_dsc': dsc(on) - dsc(off),
            'flipped_px': flip, 'growth_precision': flip_in / flip if flip else float('nan'),
            'interior_miss_covered': miss_cov / max(miss, 1)}


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    config = C.make_config(a.dataset, 256)
    ds_tr = NPY_datasets(a.data_path, config, train=True)                  # train transform (augmentation)
    ds_ev = NPY_datasets(a.data_path, config, train=True)
    ds_ev.transformer = config.test_transformer
    n = len(ds_tr) if a.max_images is None else min(a.max_images, len(ds_tr))
    perm = np.random.default_rng(a.seed).permutation(n)
    n_fit = int(round(0.8 * n))
    fit_idx, hold_idx = sorted(perm[:n_fit].tolist()), sorted(perm[n_fit:].tolist())
    sd, _ = C.load_state_dict(C.resolve_checkpoint(a.checkpoint))
    cfg = C.infer_model_config(sd)
    model = EGEUNet(**{**cfg, 'propagate_mode': 'pair', 'propagate_tau': a.tau, 'propagate_kappa': a.kappa, 'propagate_beta': a.beta})
    missing = model.load_state_dict(sd, strict=False).missing_keys
    assert all(k.startswith('propagate.') for k in missing)
    model.propagate.feature_grad = a.feature_grad
    crit = CL.build_criterion(aff_weight=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    tag = 'feature-grad' if a.feature_grad else 'detached'
    print(f'{a.dataset} [{tag}]: fine-tune on {len(fit_idx)} train images, read on {len(hold_idx)} held-out train images')
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        model.train()
        crit.reset_stats()
        order = np.random.default_rng(a.seed + 1 + ep).permutation(fit_idx).tolist()
        tot, nb = 0.0, 0
        for k in range(0, len(order), a.batch_size):
            items = [ds_tr[i] for i in order[k:k + a.batch_size]]
            x = torch.stack([u for u, _ in items]).float()
            y = torch.stack([v for _, v in items]).float()
            gp, out = model(x)
            loss = crit(gp, out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        hist.append({'loss': tot / nb, **crit.epoch_means()})
        ev = evaluate(model, ds_ev, hold_idx)
        hist[-1].update({f'heldout_{k}': v for k, v in ev.items()})
        print(f'  epoch {ep + 1}: loss {tot / nb:.4f} aff_X {hist[-1]["aff_X"]:.3f} aff_hard {hist[-1]["aff_hard"]:.3f} | held-out: '
              f'cut-AUC {ev["cut_auc"]:.3f}, ranking {ev["rank_acc"]:.2f}, S LL_inner {ev["mean_S_ll_inner"]:.3f} vs X_weak '
              f'{ev["mean_S_x_weak"]:.3f}, dDSC {ev["delta_dsc"]:+.2f} (on {ev["dsc_propagation_on"]:.2f}), growth precision '
              f'{ev["growth_precision"]:.2f}, interior miss covered {100 * ev["interior_miss_covered"]:.1f} %  ({time.time() - t0:.0f} s)')
    out = {'dataset': a.dataset, 'variant': tag, 'checkpoint': C.resolve_checkpoint(a.checkpoint), 'epochs': a.epochs, 'lr': a.lr,
           'tau': a.tau, 'kappa': a.kappa, 'beta': a.beta, 'n_fit': len(fit_idx), 'n_heldout': len(hold_idx), 'history': hist,
           'final': hist[-1]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
