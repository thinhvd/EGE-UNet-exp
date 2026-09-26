'''
Size-dependent extent bias: does the model draw small lesions too big and large lesions too small,
and what does the excess look like?

Two subcommands.

  csv    reads the per-image CSVs written by analysis/eval_per_image.py (no model, no torch) and
         prints, per run: mean log(pred area / GT area) on each area tertile, the size-bias contrast
         C = logratio(small) - logratio(large), the same split into the easy / hard half of each
         tertile (by oracle DSC, i.e. by how well the model ranks the pixels once area is fixed) and
         the oracle gap (oracle DSC - DSC), and the slope beta of log(pred area) on log(GT area).
         C > 0 / beta < 1 mean small lesions are over-drawn and large ones under-drawn. Measured on
         the 6 retrainings of sum_attn-all5 (EXP-4 section 11): C sd 0.050, beta 0.910 sd 0.017,
         small-tertile oracle gap sd 0.27 DSC points.

  masks  runs one checkpoint over a split (val, or train with the test transform) and stores per
         image: FP pixels split into near the GT contour (<= 3 px), far but in a predicted component
         that touches the lesion, and in components detached from it; perimeter and compactness
         (P^2/A) of prediction and GT; the ACL length term (TV) of the probability map and of the GT;
         the mean probability on the GT inner contour and in the 1-3 px outer ring.
  report prints the tertile summary of one or more npz files written by `masks`.

Usage:
  python analysis/extent_bias_diag.py csv --run base=results/.../egeunet_isic17_learnable_s42 [--run ...]
  python analysis/extent_bias_diag.py masks --checkpoint <run_dir> --dataset isic17 \
      --data-path data/data_isic1718/isic2017 --split val --out <file.npz>
  python analysis/extent_bias_diag.py report <file.npz> [<file.npz> ...]
'''
import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEAR_PX = 3
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def tertiles(area):
    return np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))


def logratio(tp, fp, fn):
    '''log(pred area / GT area), +1 on both sides so an empty prediction stays finite.'''
    return np.log((tp + fp + 1) / (tp + fn + 1))


# ----------------------------------------------------------------------------- csv

def cmd_csv(a):
    from scipy.stats import spearmanr
    for spec in a.run:
        label, run = spec.split('=', 1)
        path = os.path.join(run, 'analysis', a.csv_name)
        rows = list(csv.DictReader(open(path)))
        col = lambda k: np.array([float(r[k]) for r in rows])
        area, tp, fp, fn = col('area_frac'), col('tp'), col('fp'), col('fn')
        dsc, orc = col('dsc'), col('oracle_dsc')
        t, lr = tertiles(area), logratio(tp, fp, fn)
        c = lr[t == 0].mean() - lr[t == 2].mean()
        rho = spearmanr(np.log(area), lr)[0]
        # OLS slope of log(pred area) on log(GT area): 1 = no size bias, < 1 = pulled toward typical size
        beta = np.polyfit(np.log(tp + fn + 1), np.log(tp + fp + 1), 1)[0]
        print(f'\n{label}: C = {c:+.3f}   slope beta = {beta:.3f}   '
              f'spearman(log area, logratio) = {rho:+.3f}   n = {len(rows)}')
        for k, name in enumerate(['small', 'mid', 'large']):
            m = t == k
            hard = m & (orc < np.median(orc[m]))
            easy = m & ~hard
            fpfn = fp[m].sum() / max(fn[m].sum(), 1)
            print(f'  {name:5s} logratio {lr[m].mean():+.3f} (easy half {lr[easy].mean():+.3f}, '
                  f'hard half {lr[hard].mean():+.3f})  FP/FN px {fpfn:.2f}  '
                  f'DSC {100 * dsc[m].mean():.2f}  oracle {100 * orc[m].mean():.2f}  '
                  f'gap {100 * (orc[m] - dsc[m]).mean():.2f}')


# ----------------------------------------------------------------------------- masks

def perimeter(m):
    from scipy.ndimage import binary_erosion
    return int((m & ~binary_erosion(m, CROSS, border_value=0)).sum())


def total_variation(u):
    '''ACL length term (Chen et al. 2019): sum of sqrt(dx^2 + dy^2 + eps) over the image.'''
    dx = np.diff(u, axis=0)[:, :-1]
    dy = np.diff(u, axis=1)[:-1, :]
    return float(np.sqrt(dx ** 2 + dy ** 2 + 1e-8).sum())


def image_stats(u, g):
    from scipy.ndimage import distance_transform_edt, label, binary_erosion
    p = u >= 0.5
    fp = p & ~g
    d_out = distance_transform_edt(~g)
    lab, n = label(p, structure=np.ones((3, 3)))
    touching = set(np.unique(lab[p & g])) - {0}
    detached = fp & (lab > 0) & ~np.isin(lab, list(touching))
    attached = fp & ~detached
    g_contour = g & ~binary_erosion(g, CROSS, border_value=0)
    ring = ~g & (d_out <= NEAR_PX)
    return dict(
        area=int(g.sum()), pred_area=int(p.sum()), tp=int((p & g).sum()), fp=int(fp.sum()),
        fn=int((~p & g).sum()),
        fp_near=int((attached & (d_out <= NEAR_PX)).sum()),
        fp_far_att=int((attached & (d_out > NEAR_PX)).sum()),
        fp_detached=int(detached.sum()),
        n_detached_cc=len(set(np.unique(lab[detached])) - {0}),
        n_gt_cc=int(label(g, structure=np.ones((3, 3)))[1]),
        perim_pred=perimeter(p), perim_gt=perimeter(g),
        tv_u=total_variation(u), tv_g=total_variation(g.astype(np.float64)),
        u_contour=float(u[g_contour].mean()),
        u_ring=float(u[ring].mean()) if ring.any() else np.nan)


def cmd_masks(a):
    import torch
    from analysis import common as C
    from datasets.dataset import NPY_datasets
    config = C.make_config(a.dataset, a.input_size)
    ds = NPY_datasets(a.data_path, config, train=(a.split == 'train'))
    ds.transformer = config.test_transformer          # no augmentation on the train split either
    sd, _ = C.load_state_dict(C.resolve_checkpoint(a.checkpoint))
    model, _ = C.build_model(sd, 'cpu', None, strict=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, msk = ds[i]
            _, pred = model(img.float()[None])
            rows.append(image_stats(pred[0, 0].numpy(), msk.float()[0].numpy() >= 0.5))
    np.savez(a.out, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f'wrote {a.out} ({len(rows)} images)')


def cmd_report(a):
    for f in a.npz:
        z = np.load(f)
        area = z['area'].astype(float)
        t = tertiles(area)
        tp, fp, fn = z['tp'], z['fp'], z['fn']
        lr, dsc = logratio(tp, fp, fn), 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        print(f'\n{os.path.basename(f)}: n = {len(area)}, mean DSC {100 * dsc.mean():.2f}, '
              f'C = {lr[t == 0].mean() - lr[t == 2].mean():+.3f}')
        for k, name in enumerate(['small', 'mid', 'large']):
            m = t == k
            share = lambda key: 100 * z[key][m].sum() / max(fp[m].sum(), 1)
            compact = (z['perim_pred'][m] ** 2 / np.maximum(z['pred_area'][m], 1)) / \
                      (z['perim_gt'][m] ** 2 / z['area'][m])
            print(f'  {name:5s} DSC {100 * dsc[m].mean():.2f} logratio {lr[m].mean():+.3f} | FP near '
                  f'{share("fp_near"):.0f}% far-attached {share("fp_far_att"):.0f}% detached '
                  f'{share("fp_detached"):.0f}% | worst 10% imgs hold '
                  f'{100 * np.sort(fp[m])[::-1][:m.sum() // 10].sum() / max(fp[m].sum(), 1):.0f}% of FP | '
                  f'perimeter pred/GT {np.median(z["perim_pred"][m] / np.maximum(z["perim_gt"][m], 1)):.2f} '
                  f'TV u/GT {np.median(z["tv_u"][m] / z["tv_g"][m]):.2f} compactness pred/GT '
                  f'{np.median(compact):.2f} | u on GT contour {z["u_contour"][m].mean():.2f}, '
                  f'outer ring {np.nanmean(z["u_ring"][m]):.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    pc = sub.add_parser('csv')
    pc.add_argument('--run', action='append', required=True, help='label=run_dir')
    pc.add_argument('--csv-name', default='per_image_metrics_full.csv')
    pm = sub.add_parser('masks')
    pm.add_argument('--checkpoint', required=True)
    pm.add_argument('--dataset', default='isic17')
    pm.add_argument('--data-path', required=True)
    pm.add_argument('--split', choices=['val', 'train'], default='val')
    pm.add_argument('--input-size', type=int, default=256)
    pm.add_argument('--out', required=True)
    pr = sub.add_parser('report')
    pr.add_argument('npz', nargs='+')
    a = p.parse_args()
    return {'csv': cmd_csv, 'masks': cmd_masks, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
