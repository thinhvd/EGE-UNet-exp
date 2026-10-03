'''
EXP-14 step 0, part D (CPU): choose the weight of each added loss term on the VAL split of the 70:10:20 protocol.

One invocation = one (term, lambda): fine-tune the fusion-only checkpoint of EXP-11 (D1) with the original loss +
lambda * term for --epochs on the train split (train transform, i.e. with augmentation; AdamW, constant --lr) and
evaluate on the val split after every epoch: per-image mean DSC (the EXP-14 headline metric), per-image mean
precision and recall, pooled DSC. --term none is the control (same fine-tune, original loss only). The score used
for the choice is the mean over the last --score-epochs epochs. Same seed, image order and augmentation draws for
every invocation of a dataset.

  python analysis/exp14_lambda_sweep.py --dataset isic17 --term ihp --lam 5.2 --tau 0.15 \
      --out results/exp14_step0/sweep/isic17_ihp_5.2.json
'''
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from datasets.dataset import NPY_datasets, split_dataset           # noqa: E402
from models.egeunet import EGEUNet                                 # noqa: E402
import contour_losses as CL                                        # noqa: E402

ROOT = 'data/data_isic1718_701020'
SUB = {'isic17': 'isic2017', 'isic18': 'isic2018'}
CSF_RUN = 'results/EGE-UNet-results-exp11/egeunet_{ds}_learnable_fuse-sum-shallow3-d8_s42'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--term', required=True, choices=['none', 'ihp', 'focal', 'tversky'])
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--score-epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out', required=True)
    return p.parse_args()


def evaluate(model, ds):
    model.eval()
    dsc, prec, rec = [], [], []
    TP = FP = FN = 0
    with torch.no_grad():
        for i in range(len(ds)):
            img, msk = ds[i]
            p = model(img.float()[None])[1][0, 0].numpy() >= 0.5
            g = msk[0].numpy() >= 0.5
            tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            dsc.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
            if tp + fp:
                prec.append(tp / (tp + fp))
            if tp + fn:
                rec.append(tp / (tp + fn))
    return {'dsc_img': 100 * float(np.mean(dsc)), 'precision_img': 100 * float(np.mean(prec)), 'recall_img': 100 * float(np.mean(rec)),
            'dsc_pooled': 100 * 2 * TP / (2 * TP + FP + FN)}


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    config = C.make_config(a.dataset, 256)
    root = os.path.join(ROOT, SUB[a.dataset])
    ds_tr = NPY_datasets(root, config, train=True)
    if a.max_images:
        ds_tr.data = ds_tr.data[:a.max_images]
    ds_val = split_dataset(root, config, 'val')
    sd, _ = C.load_state_dict(C.resolve_checkpoint(CSF_RUN.format(ds=a.dataset)))
    model = EGEUNet(**C.infer_model_config(sd))
    model.load_state_dict(sd, strict=True)
    if a.term == 'none':
        crit = CL.build_criterion()
        desc = 'bcedice'
    else:
        kw = {'ihp': dict(ihp_tau=a.tau, ihp_gamma=2.0, ihp_plow=0.5), 'focal': dict(focal_alpha=0.75, focal_gamma=2.0),
              'tversky': dict(tversky_alpha=0.3, tversky_beta=0.7)}[a.term]
        crit = CL.build_criterion(extra_term=a.term, extra_weight=str(a.lam), **kw)
        desc = CL.describe('bcedice', a.term, str(a.lam), **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    hist = [{'epoch': 0, **evaluate(model, ds_val)}]
    print(f'{a.dataset} {desc}: start val {hist[0]}')
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        order = np.random.default_rng(a.seed + ep).permutation(len(ds_tr)).tolist()
        losses = []
        for k in range(0, len(order), a.batch_size):
            items = [ds_tr[i] for i in order[k:k + a.batch_size]]
            x = torch.stack([u for u, _ in items]).float()
            y = torch.stack([v for _, v in items]).float()
            gp, out = model(x)
            loss = crit(gp, out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        hist.append({'epoch': ep, 'train_loss': float(np.mean(losses)), **evaluate(model, ds_val)})
        print(f'  epoch {ep}: train loss {hist[-1]["train_loss"]:.4f} | val per-image DSC {hist[-1]["dsc_img"]:.2f}, precision '
              f'{hist[-1]["precision_img"]:.2f}, recall {hist[-1]["recall_img"]:.2f}, pooled {hist[-1]["dsc_pooled"]:.2f}  ({time.time() - t0:.0f} s)')
    last = hist[-a.score_epochs:]
    score = {k: float(np.mean([h[k] for h in last])) for k in ('dsc_img', 'precision_img', 'recall_img', 'dsc_pooled')}
    out = {'dataset': a.dataset, 'term': a.term, 'lam': a.lam, 'tau': a.tau, 'loss': desc, 'epochs': a.epochs, 'lr': a.lr,
           'score_epochs': a.score_epochs, 'history': hist, 'score': score}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'score (mean of last {a.score_epochs} epochs): {score}\nwrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
