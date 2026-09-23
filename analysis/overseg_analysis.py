'''
Where does the apparent anti-over-segmentation gain of the attention fusion variants come from?

Four confounds are separated:
  calibration     - operating-point shifts a threshold change could replicate (probe sweeps a fine
                    threshold grid on all val images; report compares models at each one's optimal
                    and at an area-calibrated threshold, not only at 0.5)
  coverage        - per-image spill (FP / GT area) measured at a FIXED per-image recall, so models
                    are compared at equal lesion coverage
  boundary        - Boundary-IoU (2 px band), HD95 and ASSD on the small-lesion group
  causality       - --knockout zeroes the attention delta of a trained checkpoint at inference;
                    if the anti-spill behaviour survives, the attention block is not what causes it

Usage:
  python analysis/overseg_analysis.py probe --checkpoint <run_dir> --ref-csv <base per-image csv> \
      --data-path data/data_isic1718/isic2017 --out <file.npz> [--knockout]
  python analysis/overseg_analysis.py report --dir <folder with npz> --official <small_lesion per_image.csv>
'''
import os
import sys
import csv
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLDS = np.round(np.arange(0.30, 0.8001, 0.025), 3)          # includes 0.5
BOUNDARY_THRESHOLDS = [0.4, 0.5, 0.6, 0.65, 0.7]
RECALL_TARGETS = [0.80, 0.90, 0.95]
BAND = 2   # px, Boundary-IoU band


def small_indices(ref_csv):
    rows = list(csv.DictReader(open(ref_csv)))
    area = np.array([float(r['area_frac']) for r in rows])
    edges = np.quantile(area, np.linspace(0, 1, 4)[1:-1])
    return np.nonzero(np.digitize(area, edges) == 0)[0], [r['filename'] for r in rows]


# ----------------------------------------------------------------------------- boundary metrics

def boundary_stats(pred, gt):
    '''Boundary-IoU (band 2 px), HD95 and ASSD between binary masks. GT is never empty here;
    an empty prediction gives biou 0 and NaN distances (counted separately).'''
    from scipy.ndimage import binary_erosion, distance_transform_edt
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    if not pred.any():
        return 0.0, np.nan, np.nan, True
    gb = gt & ~binary_erosion(gt, cross, border_value=0)
    pb = pred & ~binary_erosion(pred, cross, border_value=0)
    d_to_g = distance_transform_edt(~gb)
    d_to_p = distance_transform_edt(~pb)
    d = np.concatenate([d_to_g[pb], d_to_p[gb]])
    hd95 = float(np.percentile(d, 95))
    assd = float(d.mean())
    g_band = gt & (distance_transform_edt(gt) <= BAND)
    p_band = pred & (distance_transform_edt(pred) <= BAND)
    inter = np.logical_and(g_band, p_band).sum()
    union = np.logical_or(g_band, p_band).sum()
    biou = float(inter / union) if union else 1.0
    return biou, hd95, assd, False


# ----------------------------------------------------------------------------- probe

def probe(a):
    import torch
    from analysis import common as C
    from models.fusion import CrossStageAxialAttention
    from datasets.dataset import NPY_datasets

    small, ref_names = small_indices(a.ref_csv)
    ckpt = C.resolve_checkpoint(a.checkpoint)
    sd, _ = C.load_state_dict(ckpt)
    model, cfg = C.build_model(sd, 'cpu', None, strict=True)
    knocked = 0
    if a.knockout:
        for m in model.modules():
            if isinstance(m, CrossStageAxialAttention):
                m.forward = lambda x: torch.zeros_like(x)      # delta = 0: attention removed
                knocked += 1
        if knocked == 0:
            raise SystemExit(f'{a.checkpoint}: --knockout requested but the model has no attention block')

    config = C.make_config('isic17', 256)
    ds = NPY_datasets(a.data_path, config, train=False)
    if [os.path.basename(p[0]) for p in ds.data] != ref_names:
        raise SystemExit('dataset order does not match the reference csv')

    n, nt = len(ds), len(THRESHOLDS)
    tp = np.zeros((n, nt), np.int32); fp = np.zeros((n, nt), np.int32); fn = np.zeros((n, nt), np.int32)
    gt_area = np.zeros(n, np.int32)
    in_small = np.zeros(n, bool); in_small[small] = True
    pos = {int(i): k for k, i in enumerate(small)}
    fr_fp = np.zeros((len(small), len(RECALL_TARGETS)), np.int32)
    nb = len(BOUNDARY_THRESHOLDS)
    biou = np.zeros((len(small), nb)); hd95 = np.zeros((len(small), nb))
    assd = np.zeros((len(small), nb)); empty = np.zeros((len(small), nb), bool)

    with torch.no_grad():
        for i in range(n):
            img, msk = ds[i]
            _, out = model(img.float()[None])
            prob = out[0, 0].numpy()
            gt = msk.float()[0].numpy() >= 0.5
            gt_area[i] = int(gt.sum())
            gtf, pf = gt.ravel(), prob.ravel()
            for j, t in enumerate(THRESHOLDS):
                pred = pf >= t
                tp[i, j] = int((pred & gtf).sum()); fp[i, j] = int((pred & ~gtf).sum())
                fn[i, j] = int((~pred & gtf).sum())
            if in_small[i]:
                k = pos[i]
                gp = np.sort(pf[gtf])[::-1]                     # GT-pixel probs, descending
                for j, r in enumerate(RECALL_TARGETS):
                    tau = gp[min(int(np.ceil(r * gt_area[i])), gt_area[i]) - 1]
                    fr_fp[k, j] = int((pf[~gtf] >= tau).sum())
                for j, t in enumerate(BOUNDARY_THRESHOLDS):
                    biou[k, j], hd95[k, j], assd[k, j], empty[k, j] = boundary_stats(prob >= t, gt)
            if (i + 1) % 130 == 0:
                print(f'  {a.label}: {i + 1}/{n}', flush=True)

    np.savez_compressed(a.out, thresholds=THRESHOLDS, boundary_thresholds=np.array(BOUNDARY_THRESHOLDS),
                        recall_targets=np.array(RECALL_TARGETS), tp=tp, fp=fp, fn=fn, gt_area=gt_area,
                        small=small, fr_fp=fr_fp, biou=biou, hd95=hd95, assd=assd, pred_empty=empty,
                        label=np.array(a.label), knockout=np.array(bool(a.knockout)),
                        checkpoint=np.array(ckpt))
    print(f'{a.label}: wrote {a.out}')


# ----------------------------------------------------------------------------- report helpers

def t2_count(tpv, fpv, fnv):
    rec = tpv / (tpv + fnv)
    prec = np.where(tpv + fpv > 0, tpv / np.maximum(tpv + fpv, 1), 0.0)
    dsc = 2 * tpv / (2 * tpv + fpv + fnv)
    return int(((dsc < 0.85) & (rec >= 0.8) & (prec < 0.8)).sum())


def at_threshold(d, j):
    s = d['small']
    tpv, fpv, fnv = d['tp'][s, j].astype(float), d['fp'][s, j].astype(float), d['fn'][s, j].astype(float)
    rec, dsc = tpv / (tpv + fnv), 2 * tpv / (2 * tpv + fpv + fnv)
    prec = np.where(tpv + fpv > 0, tpv / np.maximum(tpv + fpv, 1), 0.0)
    spill = fpv / (tpv + fnv)
    return dict(t=float(d['thresholds'][j]), dsc=dsc.mean(), rec=rec.mean(), prec=prec.mean(),
                t2=t2_count(tpv, fpv, fnv), p90=float(np.percentile(spill, 90)),
                fpgt=int((spill > 1).sum()), spill=spill, dsc_vec=dsc)


def best_small_j(d):
    s = d['small']
    dscs = [(2 * d['tp'][s, j] / (2 * d['tp'][s, j] + d['fp'][s, j] + d['fn'][s, j])).mean()
            for j in range(len(d['thresholds']))]
    return int(np.argmax(dscs))


def area_cal_j(d):
    tot_gt = int(d['gt_area'].sum())
    diffs = [abs(int(d['tp'][:, j].sum() + d['fp'][:, j].sum()) - tot_gt) for j in range(len(d['thresholds']))]
    return int(np.argmin(diffs))


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, (n, diff.size))
    m = diff[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(a):
    files = sorted(f for f in os.listdir(a.dir) if f.endswith('.npz'))
    D = {}
    for f in files:
        z = np.load(os.path.join(a.dir, f), allow_pickle=False)
        D[str(z['label'])] = {k: z[k] for k in z.files}
    official = {}
    if a.official:
        for r in csv.DictReader(open(a.official, encoding='utf-8')):
            official.setdefault(r['model'], []).append(r)

    j05 = int(np.nonzero(np.isclose(next(iter(D.values()))['thresholds'], 0.5))[0][0])

    print('=== khớp với đánh giá chính thức tại threshold 0,5 (nhóm nhỏ, 217 ảnh) ===')
    for lab, d in D.items():
        if lab in official:
            off = official[lab]
            s = d['small']
            same = sum((int(d['tp'][i, j05]), int(d['fp'][i, j05]), int(d['fn'][i, j05]))
                       == (int(r['tp']), int(r['fp']), int(r['fn'])) for i, r in zip(s, off))
            print(f'  {lab:22s} {same}/217 ảnh trùng tuyệt đối (lệch còn lại là vài pixel sát ngưỡng CPU/GPU)')

    print('\n=== (A) So sánh khử calibration — nhóm nhỏ ===')
    print(f'{"model":26s} {"@0.5":>34s} | {"@threshold tốt nhất riêng":>40s} | {"@threshold cân diện tích":>40s}')
    print(f'{"":26s} {"DSC":>6s} {"R":>6s} {"P":>6s} {"T2":>3s} {"p90":>5s} {"FP>GT":>5s} | '
          f'{"t":>5s} {"DSC":>6s} {"R":>6s} {"P":>6s} {"T2":>3s} {"p90":>5s} {"FP>GT":>5s} | '
          f'{"t":>5s} {"DSC":>6s} {"R":>6s} {"P":>6s} {"T2":>3s} {"p90":>5s} {"FP>GT":>5s}')
    for lab, d in D.items():
        r0 = at_threshold(d, j05); rb = at_threshold(d, best_small_j(d)); ra = at_threshold(d, area_cal_j(d))
        def fmt(r):
            return f'{r["dsc"]*100:6.2f} {r["rec"]:6.3f} {r["prec"]:6.3f} {r["t2"]:3d} {r["p90"]:5.2f} {r["fpgt"]:5d}'
        print(f'{lab:26s} {fmt(r0)} | {rb["t"]:5.3f} {fmt(rb)} | {ra["t"]:5.3f} {fmt(ra)}')

    print('\n=== (B) Spill tại recall cố định từng ảnh (FP / diện tích thật, nhóm nhỏ) ===')
    rt = D[a.base]['recall_targets']
    print(f'{"model":26s} ' + ' | '.join(f'{"R="+format(r,".2f"):>24s}' for r in rt))
    print(f'{"":26s} ' + ' | '.join(f'{"trung vị":>8s} {"p90":>7s} {"FP>GT":>7s}' for _ in rt))
    base_sp = {j: D[a.base]['fr_fp'][:, j] / D[a.base]['gt_area'][D[a.base]['small']] for j in range(len(rt))}
    for lab, d in D.items():
        sp = {j: d['fr_fp'][:, j] / d['gt_area'][d['small']] for j in range(len(rt))}
        cells = [f'{np.median(sp[j]):8.3f} {np.percentile(sp[j], 90):7.2f} {int((sp[j] > 1).sum()):7d}'
                 for j in range(len(rt))]
        print(f'{lab:26s} ' + ' | '.join(cells))
    print('\n  chênh lệch log-spill trung bình vs base tại R=0.90 (âm = ít tô tràn hơn; CI bootstrap 95% theo ảnh):')
    for lab, d in D.items():
        if lab == a.base:
            continue
        diff = np.log1p(d['fr_fp'][:, 1] / d['gt_area'][d['small']]) - np.log1p(base_sp[1])
        lo, hi = boot_ci(diff)
        mark = ' *' if hi < 0 or lo > 0 else ''
        print(f'    {lab:24s} {diff.mean():+.4f}  [{lo:+.4f}; {hi:+.4f}]{mark}')

    print('\n=== (C) Chỉ số biên — nhóm nhỏ, tại 0,5 và tại threshold tốt nhất riêng ===')
    jb05 = int(np.nonzero(np.isclose(D[a.base]['boundary_thresholds'], 0.5))[0][0])
    print(f'{"model":26s} {"BIoU@2px":>9s} {"HD95":>7s} {"ASSD":>6s} {"rỗng":>4s} | {"t*":>5s} {"BIoU":>6s} {"HD95":>7s} {"ASSD":>6s}')
    for lab, d in D.items():
        bt = d['boundary_thresholds']
        tstar = float(d['thresholds'][best_small_j(d)])
        jbs = int(np.argmin(np.abs(bt - tstar)))
        def m(x, j):
            v = x[:, j]
            return np.nanmean(v)
        print(f'{lab:26s} {m(d["biou"], jb05):9.3f} {np.nanmedian(d["hd95"][:, jb05]):7.2f} '
              f'{np.nanmedian(d["assd"][:, jb05]):6.2f} {int(d["pred_empty"][:, jb05].sum()):4d} | '
              f'{bt[jbs]:5.2f} {m(d["biou"], jbs):6.3f} {np.nanmedian(d["hd95"][:, jbs]):7.2f} {np.nanmedian(d["assd"][:, jbs]):6.2f}')
    print('\n  chênh lệch BIoU@0,5 trung bình vs base (CI bootstrap 95% theo ảnh; dương = biên tốt hơn):')
    b0 = D[a.base]['biou'][:, jb05]
    for lab, d in D.items():
        if lab == a.base:
            continue
        diff = d['biou'][:, jb05] - b0
        lo, hi = boot_ci(diff)
        mark = ' *' if hi < 0 or lo > 0 else ''
        print(f'    {lab:24s} {diff.mean():+.4f}  [{lo:+.4f}; {hi:+.4f}]{mark}')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('probe')
    pp.add_argument('--checkpoint', required=True)
    pp.add_argument('--ref-csv', required=True)
    pp.add_argument('--data-path', required=True)
    pp.add_argument('--out', required=True)
    pp.add_argument('--label', required=True)
    pp.add_argument('--knockout', action='store_true')
    pr = sub.add_parser('report')
    pr.add_argument('--dir', required=True)
    pr.add_argument('--base', default='base')
    pr.add_argument('--official', default=None, help='small_lesion per_image.csv for the 0.5 sanity check')
    a = p.parse_args()
    return probe(a) if a.cmd == 'probe' else report(a)


if __name__ == '__main__':
    sys.exit(main())
