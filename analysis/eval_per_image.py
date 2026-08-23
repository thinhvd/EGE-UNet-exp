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

CSV_FIELDS = ['filename', 'dsc', 'iou', 'tp', 'fp', 'fn', 'tn', 'area_frac', 'pred_area_frac', 'cy', 'cx',
              'centroid_offset', 'touches_border', 'bbox_y0', 'bbox_x0', 'bbox_y1', 'bbox_x1']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='.pth file or experiment work_dir')
    p.add_argument('--data-path', required=True, help='dataset root containing val/images and val/masks')
    p.add_argument('--dataset', default='isic17', choices=['isic17', 'isic18'])
    p.add_argument('--input-size', type=int, default=256)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--threshold', type=float, default=0.5)
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

    rows = []
    TP = FP = FN = TN = 0
    dx, dy = (args.shift if args.shift is not None else (0, 0))
    with torch.no_grad():
        for i in range(n):
            img, msk = ds[i]
            img, msk = img.float(), msk.float()
            if dx or dy:
                img, msk = C.shift_tensor(img, dx, dy), C.shift_tensor(msk, dx, dy)
            gt_pre, out = model(img[None].to(device))
            pred = out[0, 0].cpu().numpy() >= args.threshold
            gt = msk[0].numpy() >= 0.5
            tp, fp, fn, tn = C.binary_counts(pred, gt)
            TP += tp; FP += fp; FN += fn; TN += tn
            row = {'filename': os.path.basename(ds.data[i][0]),
                   'dsc': C.dsc_from_counts(tp, fp, fn), 'iou': C.iou_from_counts(tp, fp, fn),
                   'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                   'pred_area_frac': float(pred.mean())}
            row.update(C.mask_descriptors(gt))
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{n}')

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval='')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in CSV_FIELDS})

    pooled = C.pooled_metrics(TP, FP, FN, TN)
    dscs = np.array([r['dsc'] for r in rows]); ious = np.array([r['iou'] for r in rows])
    summary = {
        'checkpoint': ckpt, 'checkpoint_meta': meta, 'model_config': cfg, 'dataset': args.dataset,
        'data_path': args.data_path, 'input_size': args.input_size, 'threshold': args.threshold,
        'shift': [dx, dy], 'gate_override': args.gate_override, 'n_images': n,
        'pooled': pooled,
        'per_image': {'mean_dsc': float(dscs.mean()), 'median_dsc': float(np.median(dscs)),
                      'mean_iou': float(ious.mean()), 'median_iou': float(np.median(ious))},
    }
    with open(os.path.splitext(out_csv)[0] + '.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\npooled (engine-style): miou={pooled["miou"]:.6f}  f1_or_dsc={pooled["f1_or_dsc"]:.6f}  '
          f'accuracy={pooled["accuracy"]:.6f}  specificity={pooled["specificity"]:.6f}  sensitivity={pooled["sensitivity"]:.6f}')
    print(f'per-image:             mean dsc={dscs.mean():.6f}  median dsc={np.median(dscs):.6f}  mean iou={ious.mean():.6f}')
    print(f'wrote {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
