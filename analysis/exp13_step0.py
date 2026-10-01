'''
EXP-13 step 0 (CPU, TRAIN images only): fix the cap and edge parameters tau, kappa, beta of the core-to-region
propagation (models/propagation.py) before any GPU run, and preview what the propagation can do.

On the fusion-only checkpoint of EXP-11 (D1), frozen:
  1. fit the edge head of a fresh CorePropagation ('pair', same architecture and init as in training) with the
     edge-affinity loss (contour_losses.affinity_terms) on 80 % of the train images (test transform);
  2. on the other 20 % ("held-out train"), run the propagation offline for every (tau, beta, kappa) of the grid and
     report pooled DSC of the OR-fused output against the frozen base, growth precision (share of the pixels that
     the propagation turns positive which are lesion), coverage of the interior miss (lesion > 10 px inside the
     contour with p_base < 0.5), and the S = 1 control (only the relative cap, i.e. image-relative hysteresis);
  3. edge readouts on held-out train: cut-AUC of 1 - S for X_weak (edges crossing the GT contour whose lesion cell
     has p_base < 0.5) vs LL_inner (both cells lesion, crossing the base contour, > 3 cells inside the GT), and the
     per-image ranking accuracy (mean S on LL_inner > mean S on X_weak);
  4. gradient balance for the coupled arm at the best grid point: norms of the gradients of the segmentation loss,
     the completion loss and the affinity loss with respect to the edge-head parameters.
The test split is never read.

  python analysis/exp13_step0.py --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --checkpoint results/EGE-UNet-results-exp11/egeunet_isic17_learnable_fuse-sum-shallow3-d8_s42 \
      --out results/exp13_step0/isic17.json
'''
import os
import sys
import json
import time
import argparse
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from datasets.dataset import NPY_datasets                          # noqa: E402
from models.egeunet import EGEUNet                                 # noqa: E402
from models.propagation import propagate, full_weights, image_relative, shift, OFFS_HALF   # noqa: E402
import contour_losses as CL                                        # noqa: E402
from utils import GT_BceDiceLoss                                   # noqa: E402

TAUS = (0.1, 0.2, 0.3, 0.4, 0.5)
BETAS = (0.5, 1.0, 2.0)
KAPPAS = (5.0, 10.0, 20.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def batches(ds, idx, bs):
    for k in range(0, len(idx), bs):
        items = [ds[i] for i in idx[k:k + bs]]
        yield torch.stack([a for a, _ in items]).float(), torch.stack([b for _, b in items]).float()


def backbone(model, x):
    '''Frozen forward: (z_base at 128, D5, projections, rgb at 128, p_base at 256).'''
    rec = {}
    h1 = model.final.register_forward_hook(lambda m, i, o: rec.update(d5=i[0], z=o))
    h2 = [model.fusion.proj[k].register_forward_hook((lambda k: lambda m, i, o: rec.__setitem__(f'p{k}', o))(k)) for k in range(5)]
    with torch.no_grad():
        _, out = model(x)
    h1.remove()
    for h in h2:
        h.remove()
    return rec['z'], rec['d5'], [rec[f'p{k}'] for k in range(5)], F.avg_pool2d(x, 2), out


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=True)
    ds.transformer = config.test_transformer
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    perm = np.random.default_rng(a.seed).permutation(n)
    n_fit = int(round(0.8 * n))
    fit_idx, hold_idx = sorted(perm[:n_fit].tolist()), sorted(perm[n_fit:].tolist())
    sd, _ = C.load_state_dict(C.resolve_checkpoint(a.checkpoint))
    cfg = C.infer_model_config(sd)
    base = EGEUNet(**cfg)
    base.load_state_dict(sd, strict=True)
    base.eval()
    # a fresh propagation module with exactly the training init (built through EGEUNet, trunk from the checkpoint)
    holder = EGEUNet(**{**cfg, 'propagate_mode': 'pair', 'propagate_steps': a.steps, 'propagate_tau': 0.3,
                        'propagate_kappa': 10.0, 'propagate_beta': 1.0})
    missing = holder.load_state_dict(sd, strict=False).missing_keys
    assert all(k.startswith('propagate.') for k in missing)
    prop = holder.propagate
    opt = torch.optim.AdamW(prop.parameters(), lr=a.lr, weight_decay=1e-2)
    print(f'{a.dataset}: fit on {len(fit_idx)} train images, select on {len(hold_idx)} held-out train images')

    # ---------------------------------------------------------------- 1. fit the edge head
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        rng = np.random.default_rng(a.seed + ep)
        order = rng.permutation(fit_idx).tolist()
        tot, nb = {}, 0
        for x, y in batches(ds, order, a.batch_size):
            z, d5, projs, rgb, _ = backbone(base, x)
            res = prop(z, d5, projs, rgb)
            masks = CL.edge_masks(y, z.shape[-2:], res['contrast'], 0.2)
            loss, terms = CL.affinity_terms(res['s_logit'], masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in terms.items():
                tot[k] = tot.get(k, 0.0) + float(v)
            tot['loss'] = tot.get('loss', 0.0) + float(loss)
            nb += 1
        hist.append({k: v / nb for k, v in tot.items()})
        print(f'  epoch {ep + 1}: ' + ' '.join(f'{k} {v:.4f}' for k, v in hist[-1].items()) + f'  ({time.time() - t0:.0f} s)')

    # ---------------------------------------------------------------- 2-3. held-out train: grid and edge readouts
    prop.eval()
    combos = list(itertools.product(TAUS, BETAS, KAPPAS))
    acc = {c: np.zeros(3) for c in combos}                 # TP, FP, FN of the OR-fused output
    acc_s1 = {(t, k): np.zeros(3) for t in TAUS for k in KAPPAS}
    grow = {c: np.zeros(3) for c in combos}                # flipped px, flipped lesion px, recovered interior miss
    base_cnt, miss_total = np.zeros(3), 0
    xw, li, rank = [], [], []
    with torch.no_grad():
        for x, y in batches(ds, hold_idx, a.batch_size):
            z, d5, projs, rgb, pb = backbone(base, x)
            g, t_z, valid, p1 = prop.cell_features(z, d5, projs, rgb / 255.0)
            s = torch.sigmoid(prop.edge_logits(g))
            seed = (torch.sigmoid(z) >= prop.seed_thr).float()
            gt = (y >= 0.5)
            gtn = gt.numpy()[:, 0]
            pbn = pb.numpy()[:, 0]
            pred0 = pbn >= 0.5
            base_cnt += [(pred0 & gtn).sum(), (pred0 & ~gtn).sum(), (~pred0 & gtn).sum()]
            deep = np.stack([(distance_transform_edt(m) > 10) if m.any() else np.zeros_like(m) for m in gtn])
            miss = deep & gtn & ~pred0
            miss_total += int(miss.sum())
            # edge readouts at 128
            lab = F.adaptive_avg_pool2d(gt.float(), z.shape[-2:]) >= 0.5
            pb128 = torch.sigmoid(z) >= 0.5
            d_in128 = torch.from_numpy(np.stack([distance_transform_edt(m) for m in lab[:, 0].numpy()]))[:, None]
            for i, (dy, dx) in enumerate(OFFS_HALF):
                valid_e = shift(torch.ones_like(lab, dtype=torch.float32), dy, dx) > 0.5
                ly = shift(lab.float(), dy, dx) > 0.5
                cross = valid_e & (lab != ly)
                inner_p = torch.where(lab, ~pb128, ~(shift(pb128.float(), dy, dx) > 0.5))      # p_base < 0.5 at the lesion cell
                xweak = cross & inner_p
                pby = shift(pb128.float(), dy, dx) > 0.5
                dy_in = shift(d_in128.float(), dy, dx)
                linner = valid_e & lab & ly & (pb128 != pby) & (d_in128 > 3) & (dy_in > 3)
                se = s[:, i:i + 1]
                xw.append(se[xweak].numpy())
                li.append(se[linner].numpy())
                for b in range(x.shape[0]):
                    if xweak[b].any() and linner[b].any():
                        rank.append(float(se[b][linner[b]].mean() > se[b][xweak[b]].mean()))
            # grid
            tcap = torch.where(valid.view(-1, 1, 1, 1), t_z, torch.ones_like(t_z))
            for tau in TAUS:
                for kappa in KAPPAS:
                    u = torch.sigmoid(kappa * (tcap - tau))
                    variants = [((tau, beta, kappa), full_weights(s, u, beta)) for beta in BETAS]
                    variants.append(((tau, None, kappa), full_weights(torch.ones_like(s), u)))
                    for key, w8 in variants:
                        r = propagate(seed, w8, a.steps)
                        gg = F.interpolate(r - seed, scale_factor=(2, 2), mode='bilinear', align_corners=True).numpy()[:, 0]
                        pf = 1 - (1 - pbn) * (1 - gg)
                        pred = pf >= 0.5
                        cnt = [(pred & gtn).sum(), (pred & ~gtn).sum(), (~pred & gtn).sum()]
                        if key[1] is None:
                            acc_s1[(tau, kappa)] += cnt
                        else:
                            acc[key] += cnt
                            flip = pred & ~pred0
                            grow[key] += [flip.sum(), (flip & gtn).sum(), (flip & miss).sum()]
    dsc = lambda c: 100 * 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)
    d0 = dsc(base_cnt)
    grid = []
    for (tau, beta, kappa), c in acc.items():
        gr = grow[(tau, beta, kappa)]
        grid.append({'tau': tau, 'beta': beta, 'kappa': kappa, 'dsc': dsc(c), 'delta_dsc': dsc(c) - d0,
                     'flipped_px': int(gr[0]), 'growth_precision': float(gr[1] / gr[0]) if gr[0] else float('nan'),
                     'interior_miss_covered': float(gr[2] / max(miss_total, 1)),
                     's_one_delta_dsc': dsc(acc_s1[(tau, kappa)]) - d0})
    grid.sort(key=lambda r: -r['delta_dsc'])
    xw, li = np.concatenate(xw), np.concatenate(li)
    edge = {'n_x_weak': int(len(xw)), 'n_ll_inner': int(len(li)), 'mean_S_x_weak': float(xw.mean()) if len(xw) else float('nan'),
            'mean_S_ll_inner': float(li.mean()) if len(li) else float('nan'),
            'cut_auc_xweak_vs_llinner': auc(1 - xw, 1 - li), 'rank_acc': float(np.mean(rank)) if rank else float('nan'),
            'n_rank_images': len(rank)}
    print(f'  held-out base pooled DSC {d0:.2f}; interior-miss px {miss_total}')
    print(f'  edges: mean S on X_weak {edge["mean_S_x_weak"]:.3f} vs LL_inner {edge["mean_S_ll_inner"]:.3f}; '
          f'cut-AUC {edge["cut_auc_xweak_vs_llinner"]:.3f}; ranking accuracy {edge["rank_acc"]:.2f} over {len(rank)} images')
    for r in grid[:5]:
        print(f'  tau {r["tau"]} beta {r["beta"]} kappa {r["kappa"]}: dDSC {r["delta_dsc"]:+.2f} (S=1: {r["s_one_delta_dsc"]:+.2f}) '
              f'growth precision {r["growth_precision"]:.2f}, interior miss covered {100 * r["interior_miss_covered"]:.1f} %')

    # ---------------------------------------------------------------- 4. gradient balance for the coupled arm
    best = grid[0]
    prop.train()
    prop.tau, prop.kappa, prop.beta = best['tau'], best['kappa'], best['beta']
    prop.mode = 'pair_coupled'
    params = [p for p in prop.parameters()]
    norms = {'seg': [], 'comp': [], 'aff': []}
    for k, (x, y) in enumerate(batches(ds, hold_idx, a.batch_size)):
        if k >= 6:
            break
        z, d5, projs, rgb, pb = backbone(base, x)
        res = prop(z, d5, projs, rgb)
        gg = F.interpolate(res['g'], scale_factor=(2, 2), mode='bilinear', align_corners=True)
        pf = 1 - (1 - pb) * (1 - gg)
        masks = CL.edge_masks(y, z.shape[-2:], res['contrast'], 0.2)
        terms = {'seg': CL.BceDiceLoss(1, 1)(pf, y), 'comp': CL.completion_term(res['r'], masks['lab'], res['seed']),
                 'aff': CL.affinity_terms(res['s_logit'], masks)[0]}
        for name, t in terms.items():
            gr = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            norms[name].append(float(torch.sqrt(sum((g ** 2).sum() for g in gr if g is not None))))
    gb = {k: float(np.median(v)) for k, v in norms.items()}
    gb['comp_over_seg'] = gb['comp'] / max(gb['seg'], 1e-12)
    print(f'  gradient norms on the edge head (median of 6 batches, coupled, best grid point): seg {gb["seg"]:.4g}, '
          f'comp {gb["comp"]:.4g}, aff {gb["aff"]:.4g}; comp / seg = {gb["comp_over_seg"]:.2f}')
    out = {'dataset': a.dataset, 'checkpoint': C.resolve_checkpoint(a.checkpoint), 'n_fit': len(fit_idx), 'n_heldout': len(hold_idx),
           'fit_history': hist, 'heldout_base_dsc': d0, 'heldout_interior_miss_px': miss_total, 'edges': edge,
           'grid': grid, 'grad_balance_at_best': gb}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
