'''
EXP-14 step 0 (CPU), on the 70:10:20 protocol (data/data_isic1718_701020, scripts/make_split_701020.py). Every choice
uses the train or the new val split; the test split is read only by the noise ruler (part C), which evaluates OLD
checkpoints to measure how much the per-image metric moves between retrainings, before any EXP-14 run exists.

A. Interior threshold tau (val). On the fusion-only checkpoint (EXP-11 D1) take the val images the model segments
   well (per-image DSC >= --good-dsc, 0.9) and the normalized depth (contour_losses.normalized_depth) of their missed
   lesion pixels: these are the ordinary boundary-placement misses. tau = the --pct (90th) percentile of that depth:
   misses deeper than tau are rare on well-segmented images, i.e. they are interior misses. Reported per dataset;
   the shared tau is their mean rounded to 0.05.
B. Weight unit per term (train). At the same checkpoint, on the train images (test transform, batch 8): the RMS over
   batches of the gradient norm with respect to the final logit of BceDice(final output) divided by the same for the
   term (ihp with the shared tau, focal, tversky). lambda_unit makes the term push as hard as the original loss on
   the final output; the shared unit is the geometric mean of the two datasets.
C. Noise ruler (test). Per-image DSC / precision / recall on the test split of every original-model checkpoint of
   EXP-5..11 (7 retrainings of the same configuration) and of the fusion-only checkpoints (EXP-10 B1, EXP-11 D1):
   spread of the per-image mean DSC, and every pairwise paired t-test between two retrainings (how often p < 0.05
   arises from retraining alone). Caveat: these checkpoints were selected on the author's val split, which contains
   the new test images, so their absolute level is slightly optimistic; the spread is what matters here.

  python analysis/exp14_step0.py --dataset isic17 --out results/exp14_step0/isic17.json
'''
import os
import sys
import json
import argparse
import itertools

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from datasets.dataset import split_dataset                         # noqa: E402
import contour_losses as CL                                        # noqa: E402
from utils import BceDiceLoss                                      # noqa: E402

ROOT = 'data/data_isic1718_701020'
SUB = {'isic17': 'isic2017', 'isic18': 'isic2018'}
CSF_RUN = 'results/EGE-UNet-results-exp11/egeunet_{ds}_learnable_fuse-sum-shallow3-d8_s42'
BASELINES = [f'results/EGE-UNet-results-exp{e}/egeunet_{{ds}}_learnable_s42' for e in (5, 6, 7, 8, 9, 10, 11)]
CSF_DRAWS = ['results/EGE-UNet-results-exp10/egeunet_{ds}_learnable_fuse-sum-shallow3-d8_s42', CSF_RUN]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--good-dsc', type=float, default=0.9)
    p.add_argument('--pct', type=float, default=90.0)
    p.add_argument('--tau', type=float, default=None, help='part B: use this tau (default: the one found in part A)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--parts', default='A,B,C')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out', required=True)
    return p.parse_args()


def load(run):
    sd, _ = C.load_state_dict(C.resolve_checkpoint(run))
    return C.build_model(sd, 'cpu', None, strict=True)[0]


def per_image(model, ds, n):
    rows = []
    with torch.no_grad():
        for i in range(n):
            img, msk = ds[i]
            p = model(img.float()[None])[1][0, 0].numpy() >= 0.5
            g = msk[0].numpy() >= 0.5
            tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
            rows.append({'dsc': 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else 1.0,
                         'precision': tp / (tp + fp) if tp + fp else float('nan'), 'recall': tp / (tp + fn) if tp + fn else float('nan')})
    return rows


def logit_grad_norm(fn, u, g):
    u = u.detach().clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(fn(u, g), u)
    return float((gr * u.detach() * (1 - u.detach())).norm())


def main():
    a = parse_args()
    parts = set(a.parts.split(','))
    config = C.make_config(a.dataset, 256)
    root = os.path.join(ROOT, SUB[a.dataset])
    out = {'dataset': a.dataset, 'data_root': root}
    if os.path.isfile(a.out):
        out.update(json.load(open(a.out)))
    csf = load(CSF_RUN.format(ds=a.dataset))

    if 'A' in parts:
        ds = split_dataset(root, config, 'val')
        n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
        depths, n_good, fn_all, fn_deep = [], 0, [], []
        with torch.no_grad():
            for i in range(n):
                img, msk = ds[i]
                p = csf(img.float()[None])[1][0, 0].numpy() >= 0.5
                g = msk[0].numpy() >= 0.5
                dn = CL.normalized_depth(msk.float()[None])[0, 0].numpy()
                fn = g & ~p
                fn_all.append(dn[fn])
                tp = int((p & g).sum())
                dsc = 2 * tp / max(2 * tp + int((p & ~g).sum()) + int(fn.sum()), 1)
                if dsc >= a.good_dsc and fn.any():
                    depths.append(dn[fn])
                    n_good += 1
        dep = np.concatenate(depths)
        tau = float(np.percentile(dep, a.pct))
        allfn = np.concatenate(fn_all)
        out['A'] = {'n_val': n, 'n_good_images': n_good, 'good_dsc': a.good_dsc, 'pct': a.pct, 'tau': tau,
                    'good_fn_depth_quantiles': {q: float(np.percentile(dep, q)) for q in (50, 75, 90, 95)},
                    'share_of_all_val_fn_at_depth_ge_tau': float((allfn >= tau).mean()), 'n_val_fn_px': int(len(allfn))}
        print(f'A {a.dataset}: {n_good} well-segmented val images; their missed-pixel depth quantiles 50/75/90/95 % = '
              + ' / '.join(f'{v:.3f}' for v in out['A']['good_fn_depth_quantiles'].values())
              + f'; tau = {tau:.3f}; {100 * out["A"]["share_of_all_val_fn_at_depth_ge_tau"]:.1f} % of all val misses are deeper')

    if 'B' in parts:
        tau = a.tau if a.tau is not None else out['A']['tau']
        ds = split_dataset(root, config, 'train')
        n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
        terms = {'ihp': CL.InteriorHardPositive(tau, 2.0, 0.5), 'focal': CL.FocalLoss(0.75, 2.0), 'tversky': CL.TverskyLoss(0.3, 0.7)}
        base = BceDiceLoss(1, 1)
        sq = {k: 0.0 for k in ['base', *terms]}
        nb = 0
        for k in range(0, n, a.batch_size):
            items = [ds[i] for i in range(k, min(k + a.batch_size, n))]
            x = torch.stack([u for u, _ in items]).float()
            y = torch.stack([v for _, v in items]).float()
            with torch.no_grad():
                u = csf(x)[1]
            sq['base'] += logit_grad_norm(base, u, y) ** 2
            for name, t in terms.items():
                sq[name] += logit_grad_norm(t, u, y) ** 2
            nb += 1
        rms = {k: (v / nb) ** 0.5 for k, v in sq.items()}
        out['B'] = {'tau_used': tau, 'n_train': n, 'rms_grad_norm': rms,
                    'lambda_unit': {k: rms['base'] / rms[k] for k in terms}}
        print(f'B {a.dataset} (tau {tau:.3f}): RMS logit-gradient norm base {rms["base"]:.4g}; lambda unit '
              + ', '.join(f'{k} {v:.4g}' for k, v in out['B']['lambda_unit'].items()))

    if 'C' in parts:
        ds = split_dataset(root, config, 'test')
        n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
        draws = {}
        for run in BASELINES + CSF_DRAWS:
            path = run.format(ds=a.dataset)
            if os.path.isdir(path):
                draws[path] = per_image(load(path), ds, n)
                print(f'C {a.dataset}: {path}: per-image mean DSC {100 * np.mean([r["dsc"] for r in draws[path]]):.2f}')
        base = [p for p in draws if p.endswith('_learnable_s42')]
        csfs = [p for p in draws if 'fuse-sum-shallow3-d8_s42' in p]
        arr = {p: np.array([r['dsc'] for r in draws[p]]) for p in draws}
        means = np.array([100 * arr[p].mean() for p in base])
        pairs = []
        for p, q in itertools.combinations(base, 2):
            t = stats.ttest_rel(arr[p], arr[q])
            pairs.append({'a': p, 'b': q, 'diff': float(100 * (arr[p] - arr[q]).mean()), 'p': float(t.pvalue)})
        mean_rec = {p: 100 * np.nanmean([r['recall'] for r in draws[p]]) for p in base}
        mean_prec = {p: 100 * np.nanmean([r['precision'] for r in draws[p]]) for p in base}
        out['C'] = {'n_test': n, 'baseline_per_image_mean_dsc': {p: float(m) for p, m in zip(base, means)},
                    'baseline_mean': float(means.mean()), 'baseline_sd': float(means.std(ddof=1)),
                    'baseline_pair_abs_diff_max': float(max(abs(x['diff']) for x in pairs)),
                    'baseline_pairs_p_lt_0.05': int(sum(x['p'] < 0.05 for x in pairs)), 'baseline_pairs': len(pairs),
                    'baseline_pair_tests': pairs,
                    'baseline_recall_sd': float(np.std(list(mean_rec.values()), ddof=1)),
                    'baseline_precision_sd': float(np.std(list(mean_prec.values()), ddof=1)),
                    'csf_per_image_mean_dsc': {p: float(100 * arr[p].mean()) for p in csfs}}
        if len(csfs) == 2:
            t = stats.ttest_rel(arr[csfs[0]], arr[csfs[1]])
            out['C']['csf_pair'] = {'diff': float(100 * (arr[csfs[0]] - arr[csfs[1]]).mean()), 'p': float(t.pvalue)}
        print(f'C {a.dataset}: 7 baseline retrainings, per-image mean DSC {means.mean():.2f} +- {means.std(ddof=1):.2f} (SD); '
              f'largest pairwise gap {out["C"]["baseline_pair_abs_diff_max"]:.2f}; pairs with p < 0.05: '
              f'{out["C"]["baseline_pairs_p_lt_0.05"]} / {len(pairs)}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
