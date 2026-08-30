'''
Per-image evaluation of an EGE-UNet checkpoint on the val split, for stratified comparisons.

For every val image it records DSC / IoU plus GT descriptors (lesion area fraction, centroid offset from
the image center, touches-border flag, bbox) and also the dataset-level pooled metrics computed exactly
like engine.test_one_epoch (so they reconcile with test_results.json).

Options for hypothesis tests without retraining:
  --shift DX DY            translate image+mask jointly (reflect padding) before inference
  --gate-override MODE     knock out the GHPA gate structure of a trained model post hoc:
                           ones | spatial_mean | shuffle (see analysis/common.py GateOverride)

Usage:
  python analysis/eval_per_image.py --checkpoint <work_dir or .pth> --data-path data/data_isic1718/isic2017
Output: <work_dir>/analysis/per_image_metrics[_shift{dx}_{dy}][_override-{mode}].csv (+ .json sidecar)
'''
import os
import sys
import csv
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C
from datasets.dataset import NPY_datasets

CSV_FIELDS = ['filename', 'dsc', 'iou', 'oracle_dsc', 'tp', 'fp', 'fn', 'tn', 'area_frac', 'pred_area_frac',
              'cy', 'cx', 'centroid_offset', 'touches_border', 'bbox_y0', 'bbox_x0', 'bbox_y1', 'bbox_x1']


def oracle_dsc(prob, gt):
    '''
    DSC of the size-matched prediction: exactly k highest-probability pixels, k = number of GT pixels.
    Removes any area/calibration bias by construction — what remains is pure ranking/localization quality.
    '''
    k = int(gt.sum())
    if k == 0:
        return 1.0
    flat = prob.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    pred = np.zeros(flat.shape[0], dtype=bool)
    pred[idx] = True
    inter = int(np.logical_and(pred, gt.ravel()).sum())
    return float(2 * inter / (2 * k))  # |pred| == |gt| == k


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='.pth file or experiment work_dir')
    p.add_argument('--data-path', required=True, help='dataset root containing val/images and val/masks')
    p.add_argument('--dataset', default='isic17', choices=['isic17', 'isic18'])
    p.add_argument('--input-size', type=int, default=256)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--thresholds', type=str, default=None,
                   help='comma-separated sweep, e.g. 0.3,0.4,0.5,0.6,0.7 — one forward pass, per-threshold '
                        'pooled/mean/tertile metrics go to the sidecar json (csv columns stay at --threshold)')
    p.add_argument('--out-csv', default=None, help='default: <work_dir>/analysis/per_image_metrics<suffix>.csv')
    p.add_argument('--shift', type=int, nargs=2, default=None, metavar=('DX', 'DY'),
                   help='translate image+mask by (DX right, DY down) pixels with reflect padding')
    p.add_argument('--gate-override', default=None, choices=['ones', 'spatial_mean', 'shuffle'])
    p.add_argument('--override-seed', type=int, default=0)
    p.add_argument('--hpa-mode', default='auto', choices=['auto', 'learnable', 'none'])
    p.add_argument('--c-list', type=int, nargs=6, default=None)
    p.add_argument('--non-strict', action='store_true')
    p.add_argument('--max-images', type=int, default=None, help='debug: only evaluate the first N images')
    return p.parse_args()


def main():
    args = parse_args()
    device = C.pick_device(args.device)
    ckpt = C.resolve_checkpoint(args.checkpoint)
    work_dir = C.infer_work_dir(ckpt)
    suffix = ''
    if args.shift is not None:
        suffix += f'_shift{args.shift[0]}_{args.shift[1]}'
    if args.gate_override:
        suffix += f'_override-{args.gate_override}'
    out_csv = args.out_csv or os.path.join(work_dir, 'analysis', f'per_image_metrics{suffix}.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    print(f'checkpoint: {ckpt}\nout_csv:    {out_csv}')

    sd, meta = C.load_state_dict(ckpt)
    overrides = {'c_list': args.c_list, 'hpa_mode': None if args.hpa_mode == 'auto' else args.hpa_mode}
    model, cfg = C.build_model(sd, device, overrides, strict=not args.non_strict)
    if args.gate_override:
        n = C.apply_gate_override(model, args.gate_override, args.override_seed)
        print(f'gate override "{args.gate_override}" applied to {n} GHPA modules')

    config = C.make_config(args.dataset, args.input_size)
    ds = NPY_datasets(args.data_path, config, train=False)
    n = len(ds) if args.max_images is None else min(args.max_images, len(ds))
    print(f'val images: {n}  (dataset={args.dataset}, input={args.input_size}, threshold={args.threshold})')

    thresholds = [args.threshold]
    if args.thresholds:
        thresholds = sorted({float(t) for t in args.thresholds.split(',')} | {args.threshold})

    rows = []
    pooled_thr = {t: [0, 0, 0, 0] for t in thresholds}   # TP, FP, FN, TN per threshold
    dsc_thr = {t: [] for t in thresholds}
    dx, dy = (args.shift if args.shift is not None else (0, 0))
    with torch.no_grad():
        for i in range(n):
            img, msk = ds[i]
            img, msk = img.float(), msk.float()
            if dx or dy:
                img, msk = C.shift_tensor(img, dx, dy), C.shift_tensor(msk, dx, dy)
            gt_pre, out = model(img[None].to(device))
            prob = out[0, 0].cpu().numpy()
            gt = msk[0].numpy() >= 0.5
            for t in thresholds:
                tp_t, fp_t, fn_t, tn_t = C.binary_counts(prob >= t, gt)
                acc = pooled_thr[t]
                acc[0] += tp_t; acc[1] += fp_t; acc[2] += fn_t; acc[3] += tn_t
                dsc_thr[t].append(C.dsc_from_counts(tp_t, fp_t, fn_t))
            pred = prob >= args.threshold
            tp, fp, fn, tn = C.binary_counts(pred, gt)
            row = {'filename': os.path.basename(ds.data[i][0]),
                   'dsc': C.dsc_from_counts(tp, fp, fn), 'iou': C.iou_from_counts(tp, fp, fn),
                   'oracle_dsc': oracle_dsc(prob, gt),
                   'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                   'pred_area_frac': float(pred.mean())}
            row.update(C.mask_descriptors(gt))
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{n}')
    TP, FP, FN, TN = pooled_thr[args.threshold]

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval='')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in CSV_FIELDS})

    pooled = C.pooled_metrics(TP, FP, FN, TN)
    dscs = np.array([r['dsc'] for r in rows]); ious = np.array([r['iou'] for r in rows])
    areas = np.array([r['area_frac'] for r in rows])
    edges = np.quantile(areas, [1 / 3, 2 / 3])
    tert = np.digitize(areas, edges)  # 0=small, 1=medium, 2=large
    odscs = np.array([r['oracle_dsc'] for r in rows])
    summary = {
        'checkpoint': ckpt, 'checkpoint_meta': meta, 'model_config': cfg, 'dataset': args.dataset,
        'data_path': args.data_path, 'input_size': args.input_size, 'threshold': args.threshold,
        'shift': [dx, dy], 'gate_override': args.gate_override, 'n_images': n,
        'pooled': pooled,
        'per_image': {'mean_dsc': float(dscs.mean()), 'median_dsc': float(np.median(dscs)),
                      'mean_iou': float(ious.mean()), 'median_iou': float(np.median(ious))},
        'oracle': {'mean_oracle_dsc': float(odscs.mean()), 'median_oracle_dsc': float(np.median(odscs)),
                   'tertile_mean_oracle_dsc': [float(odscs[tert == t].mean()) for t in range(3)]},
        'tertile_edges_area_frac': edges.tolist(),
    }
    if len(thresholds) > 1:
        summary['sweep'] = {}
        for t in thresholds:
            tp_t, fp_t, fn_t, tn_t = pooled_thr[t]
            d = np.array(dsc_thr[t])
            summary['sweep'][f'{t:g}'] = {
                'pooled': C.pooled_metrics(tp_t, fp_t, fn_t, tn_t),
                'mean_dsc': float(d.mean()),
                'tertile_mean_dsc': [float(d[tert == k].mean()) for k in range(3)],
            }
    with open(os.path.splitext(out_csv)[0] + '.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\npooled (engine-style): miou={pooled["miou"]:.6f}  f1_or_dsc={pooled["f1_or_dsc"]:.6f}  '
          f'accuracy={pooled["accuracy"]:.6f}  specificity={pooled["specificity"]:.6f}  sensitivity={pooled["sensitivity"]:.6f}')
    print(f'per-image:             mean dsc={dscs.mean():.6f}  median dsc={np.median(dscs):.6f}  mean iou={ious.mean():.6f}')
    print(f'oracle (size-matched): mean={odscs.mean():.6f}  tertile means (S/M/L): '
          + '  '.join(f'{odscs[tert == t].mean():.4f}' for t in range(3)))
    if len(thresholds) > 1:
        print('sweep:  ' + '  '.join(f'thr={t:g}: miou={C.pooled_metrics(*pooled_thr[t])["miou"]:.4f}' for t in thresholds))
    print(f'wrote {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
