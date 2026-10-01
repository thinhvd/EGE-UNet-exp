'''
EXP-13 mechanism readouts (core-to-region propagation, models/propagation.py). CPU or GPU, one pass over the val
split per model plus one per knockout. Nothing here decides win / lose (pooled DSC, exp06_summary.py).

  E  edge statistics at the 128 grid (test images): mean S and cut-AUC (score 1 - S) on
       LL_inner  both cells lesion, crossing the base prediction's contour, both > 3 cells inside the GT (the dark-core
                 edge that EXP-11's barrier locked onto)
       X_weak    edges crossing the GT contour whose lesion cell has p_base < 0.5 (where the propagation must stop
                 after recovering the pale rim)
       X_all     every edge crossing the GT contour;  BB_near  background edges within 15 cells outside the GT
     gap share = share of X_all edges with S > 0.5; ranking accuracy = share of images whose mean S on LL_inner
     exceeds the mean S on X_weak.
  G  growth (pixels at 256 that the propagation turns positive, p >= 0.5 with p_base < 0.5): count, precision
     (share inside the GT), coverage of the interior miss (GT, > 10 px inside, p_base < 0.5), share of large-tertile
     images whose growth reaches > 10 px outside the GT, leak area (> 10 px outside).
  K  same-checkpoint knockouts: propagation off (G = 0), S = 1 (only the relative cap), cap off (U = 1), T = 16, T = 48.
     Delta pooled DSC, delta FP, interior miss of the large tertile and of the --focus set (relative, 95 % bootstrap CI).
  P  polygon split of the large tertile against --ref (as analysis/exp11_mechanism.py).

  python analysis/exp13_mechanism.py --dataset isic18 --data-path data/data_isic1718/isic2018 \
      --run D1=... --run E1=... --run E2=... --run E3=... --ref D1 \
      --focus heavy=analysis/lists/exp11_heavy_isic18.csv --drop 390 --out-dir <dir>
'''
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                                    # noqa: E402
from analysis.exp09_mechanism import GroundTruth, pooled, write_csv, vn, px        # noqa: E402
from analysis.exp11_errors import load_focus                                        # noqa: E402
from analysis.exp11_mechanism import auc, boot_ratio, polygon_score, solidity, plain_rows   # noqa: E402
from datasets.dataset import NPY_datasets                                           # noqa: E402
from models.propagation import shift, OFFS_HALF                                     # noqa: E402

TERTILES = ('small', 'mid', 'large')
GROUPS = ('LL_inner', 'X_weak', 'X_all', 'BB_near')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--ref', default=None, metavar='LABEL')
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--focus', action='append', default=[], metavar='NAME=CSV')
    p.add_argument('--drop', action='append', default=[], metavar='IMAGE')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--near', type=float, default=3.0)
    p.add_argument('--mid', type=float, default=10.0)
    p.add_argument('--boot', type=int, default=2000)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def edge_groups(gt256, pb128):
    '''Bool masks (4,h,w) per group at the 128 grid for one image.'''
    lab = F.adaptive_avg_pool2d(torch.from_numpy(gt256.astype(np.float32))[None, None], pb128.shape[-2:])[0, 0] >= 0.5
    d_in = torch.from_numpy(distance_transform_edt(lab.numpy())).float()
    d_out = torch.from_numpy(distance_transform_edt(~lab.numpy())).float() if lab.any() else torch.full(lab.shape, 1e9)
    out = {g: [] for g in GROUPS}
    for dy, dx in OFFS_HALF:
        valid = shift(torch.ones(lab.shape), dy, dx) > 0.5
        ly = shift(lab.float(), dy, dx) > 0.5
        py = shift(pb128.float(), dy, dx) > 0.5
        cross = valid & (lab != ly)
        lesion_low = torch.where(lab, ~pb128, ~py)
        out['X_all'].append(cross)
        out['X_weak'].append(cross & lesion_low)
        out['LL_inner'].append(valid & lab & ly & (pb128 != py) & (d_in > 3) & (shift(d_in, dy, dx) > 3))
        out['BB_near'].append(valid & ~lab & ~ly & (d_out <= 15) & (shift(d_out, dy, dx, fill=1e9) <= 15))
    return {g: torch.stack(v) for g, v in out.items()}


def intact_pass(model, ds, gt, device, thr):
    rec = {}
    h1 = model.final.register_forward_hook(lambda m, i, o: rec.__setitem__('z', o.detach()))
    h2 = model.propagate.register_forward_hook(lambda m, i, o: rec.__setitem__('res', {k: v.detach() for k, v in o.items()}))
    rows = []
    vals = {g: [] for g in GROUPS}
    rank = []
    gr = dict(flip=0, flip_in=0, miss=0, miss_cov=0, leak_px=0, large_imgs=0, large_leak_imgs=0)
    with torch.no_grad():
        for i in range(gt.n):
            img, _ = ds[i]
            _, out = model(img.float()[None].to(device))
            p = out[0, 0].cpu().numpy()
            rows.append(gt.counts(i, p >= thr))
            z = rec['z']
            pb = torch.sigmoid(F.interpolate(z, scale_factor=(2, 2), mode='bilinear', align_corners=True))[0, 0].cpu().numpy()
            s = torch.sigmoid(rec['res']['s_logit'])[0].cpu()
            g = gt.gt[i]
            groups = edge_groups(g, (torch.sigmoid(z) >= 0.5)[0, 0].cpu())
            for name, m in groups.items():
                vals[name].append(s[m].numpy())
            if groups['LL_inner'].any() and groups['X_weak'].any():
                rank.append(float(s[groups['LL_inner']].mean() > s[groups['X_weak']].mean()))
            d_in = distance_transform_edt(g) if g.any() else np.zeros(g.shape)
            d_out = distance_transform_edt(~g) if g.any() else np.full(g.shape, np.inf)
            flip = (p >= thr) & (pb < thr)
            miss = g & (d_in > gt_mid) & (pb < thr)
            gr['flip'] += int(flip.sum())
            gr['flip_in'] += int((flip & g).sum())
            gr['miss'] += int(miss.sum())
            gr['miss_cov'] += int((flip & miss).sum())
            leak = flip & ~g & (d_out > gt_mid)
            gr['leak_px'] += int(leak.sum())
            if gt.tertile[i] == 2:
                gr['large_imgs'] += 1
                gr['large_leak_imgs'] += int(leak.any())
    h1.remove()
    h2.remove()
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in vals.items()}
    edges = {f'mean_S_{k}': float(v.mean()) if len(v) else float('nan') for k, v in cat.items()}
    edges.update({f'n_{k}': int(len(v)) for k, v in cat.items()})
    edges['cut_auc_Xweak_vs_LLinner'] = auc(1 - cat['X_weak'], 1 - cat['LL_inner'])
    edges['cut_auc_Xall_vs_LLinner'] = auc(1 - cat['X_all'], 1 - cat['LL_inner'])
    edges['gap_share_X_all_S_gt_half'] = float((cat['X_all'] > 0.5).mean()) if len(cat['X_all']) else float('nan')
    edges['rank_acc'] = float(np.mean(rank)) if rank else float('nan')
    edges['rank_images'] = len(rank)
    growth = {'flipped_px': gr['flip'], 'growth_precision': gr['flip_in'] / gr['flip'] if gr['flip'] else float('nan'),
              'interior_miss_px': gr['miss'], 'interior_miss_covered': gr['miss_cov'] / max(gr['miss'], 1),
              'leak_px_gt10': gr['leak_px'], 'large_imgs_leaking': gr['large_leak_imgs'], 'large_imgs': gr['large_imgs']}
    return rows, edges, growth


def knockout_rows(model, ds, gt, device, thr, **switch):
    c = model.propagate
    saved = {k: getattr(c, k) for k in switch}
    for k, v in switch.items():
        setattr(c, k, v)
    try:
        return plain_rows(model, ds, gt, device, thr)
    finally:
        for k, v in saved.items():
            setattr(c, k, v)


gt_mid = 10.0


def main():
    global gt_mid
    a = parse_args()
    gt_mid = a.mid
    os.makedirs(a.out_dir, exist_ok=True)
    device = C.pick_device(a.device)
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=False)
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    print(f'{a.dataset}: {n} val images')
    gt = GroundTruth(ds, n, a.near, a.mid)
    focus = load_focus(a.focus, gt.names) if a.focus else {}
    large = [i for i in range(n) if gt.tertile[i] == 2]
    drop = {i for i, nm in enumerate(gt.names) if nm in set(a.drop)}
    runs, rows, edges, growth, ko = [], {}, [], [], []
    for spec in a.run:
        label, path = spec.split('=', 1)
        sd, _ = C.load_state_dict(C.resolve_checkpoint(path))
        model, cfg = C.build_model(sd, device, None, strict=True)
        mode = cfg.get('propagate_mode', 'none')
        runs.append(label)
        print(f'  {label}: propagate={mode}')
        if mode == 'none':
            rows[label] = plain_rows(model, ds, gt, device, a.threshold)
            continue
        rows[label], e, g = intact_pass(model, ds, gt, device, a.threshold)
        edges.append({'run': label, 'mode': mode, **e})
        growth.append({'run': label, 'mode': mode, **g})
        variants = [('propagation off (G = 0)', dict(enabled=False)), ('S = 1 (relative cap only)', dict(s_one=True))]
        if mode != 'pair_nocap':
            variants.append(('cap off (U = 1)', dict(cap_off=True)))
        variants += [('T = 16', dict(steps_override=16)), ('T = 48', dict(steps_override=48))]
        base_all = pooled(rows[label], range(n))
        for name, sw in variants:
            kr = knockout_rows(model, ds, gt, device, a.threshold, **sw)
            k_all = pooled(kr, range(n))
            row = {'run': label, 'variant': name, 'dsc_pooled': k_all['dsc'], 'delta_dsc_pooled': k_all['dsc'] - base_all['dsc'],
                   'delta_fp': k_all['fp'] - base_all['fp'], 'delta_fn': k_all['fn'] - base_all['fn']}
            for sname, idx in {'large': large, **focus}.items():
                r_rel, lo, hi = boot_ratio([kr[i]['fn_far'] for i in idx], [rows[label][i]['fn_far'] for i in idx], a.boot)
                row.update({f'miss_{sname}_intact': sum(rows[label][i]['fn_far'] for i in idx),
                            f'miss_{sname}_knockout': sum(kr[i]['fn_far'] for i in idx),
                            f'miss_{sname}_rel_change_pct': 100 * r_rel, f'miss_{sname}_ci_lo_pct': 100 * lo,
                            f'miss_{sname}_ci_hi_pct': 100 * hi})
            ko.append(row)
            print(f'    {name}: pooled DSC {row["delta_dsc_pooled"]:+.2f}, interior miss (large) {row["miss_large_rel_change_pct"]:+.1f} %')
    poly_img = [{'image': gt.names[i], 'tertile': TERTILES[gt.tertile[i]], 'polygon_score': polygon_score(gt.gt[i]),
                 'solidity': solidity(gt.gt[i])} for i in range(n)]
    poly = []
    if a.ref and a.ref in rows:
        sc = np.array([poly_img[i]['polygon_score'] for i in large], dtype=np.float64)
        if np.isfinite(sc).any():
            med = float(np.nanmedian(sc))
            halves = {'polygon-like': [i for i in large if poly_img[i]['polygon_score'] >= med],
                      'smooth': [i for i in large if poly_img[i]['polygon_score'] < med]}
            if drop:
                halves.update({f'{k}, no {"/".join(sorted(a.drop))}': [i for i in v if i not in drop] for k, v in list(halves.items())})
            for r in runs:
                if r == a.ref:
                    continue
                for hname, idx in halves.items():
                    pr, pv = pooled(rows[a.ref], idx), pooled(rows[r], idx)
                    poly.append({'run': r, 'reference': a.ref, 'half': hname, 'n_images': len(idx), 'dsc_ref': pr['dsc'],
                                 'dsc_run': pv['dsc'], 'miss_ref': pr['fn_far'], 'miss_run': pv['fn_far'],
                                 'miss_rel_change_pct': 100 * (pv['fn_far'] / pr['fn_far'] - 1) if pr['fn_far'] else float('nan'),
                                 'fp_ref': pr['fp'], 'fp_run': pv['fp']})
    for name, data in (('edges', edges), ('growth', growth), ('knockout', ko), ('polygon', poly), ('polygon_per_image', poly_img)):
        write_csv(os.path.join(a.out_dir, f'{name}.csv'), data)

    md = [f'# EXP-13 — đọc cơ chế của core-to-region propagation ({a.dataset.upper()}, {n} ảnh val)', '',
          'S = affinity trên cạnh nối hai ô lân cận ở lưới 128 (1 = cùng nhãn, 0 = cắt). LL_inner = cạnh ở mép lõi sẫm, nằm trong GT '
          '(không được cắt). X_weak = cạnh cắt GT boundary ở chỗ phía lesion có p_base < 0,5 (phải cắt sau khi lấy lại vành nhạt). '
          'cut-AUC = khả năng 1 − S xếp X_weak cao hơn LL_inner (1 = tách hoàn toàn, 0,5 = không tách).', '']
    if edges:
        md += ['## E. Cạnh', '',
               '| Run | S TB: LL_inner · X_weak · X_all · BB_near | cut-AUC X_weak vs LL_inner | cut-AUC X_all vs LL_inner | % cạnh X_all có S > 0,5 (lỗ hổng) | độ chính xác ranking theo ảnh |',
               '|---|---|---|---|---|---|']
        md += [f'| {e["run"]} | {vn(e["mean_S_LL_inner"], 3)} · {vn(e["mean_S_X_weak"], 3)} · {vn(e["mean_S_X_all"], 3)} · '
               f'{vn(e["mean_S_BB_near"], 3)} | {vn(e["cut_auc_Xweak_vs_LLinner"], 3)} | {vn(e["cut_auc_Xall_vs_LLinner"], 3)} '
               f'| {vn(100 * e["gap_share_X_all_S_gt_half"], 1)} | {vn(e["rank_acc"], 2)} ({e["rank_images"]} ảnh) |' for e in edges]
    if growth:
        md += ['', '## G. Vùng lan (pixel 256 được propagation lật thành dương)', '',
               '| Run | pixel lật | độ đúng (trong GT) | % vùng bỏ sót bên trong được phủ | pixel lan > 10 px ngoài GT | lesion to có tràn > 10 px |',
               '|---|---|---|---|---|---|']
        md += [f'| {g["run"]} | {px(g["flipped_px"])} | {vn(g["growth_precision"], 3)} | {vn(100 * g["interior_miss_covered"], 1)} '
               f'| {px(g["leak_px_gt10"])} | {g["large_imgs_leaking"]} / {g["large_imgs"]} |' for g in growth]
    if ko:
        sets = ['large'] + list(focus)
        md += ['', '## K. Knockout (so với chính model đó còn nguyên)', '',
               '| Run | Knockout | DSC gộp | Δ DSC gộp | Δ FP | ' + ' | '.join(f'Bỏ sót bên trong ({s}): Δ % [CI 95 %]' for s in sets) + ' |',
               '|---|---|---|---|---|' + '---|' * len(sets)]
        for r in ko:
            cells = [f'{vn(r[f"miss_{s}_rel_change_pct"], 1, True)} % [{vn(r[f"miss_{s}_ci_lo_pct"], 1, True)}; '
                     f'{vn(r[f"miss_{s}_ci_hi_pct"], 1, True)}]' for s in sets]
            md.append(f'| {r["run"]} | {r["variant"]} | {vn(r["dsc_pooled"])} | {vn(r["delta_dsc_pooled"], 2, True)} '
                      f'| {px(r["delta_fp"], True)} | ' + ' | '.join(cells) + ' |')
    if poly:
        md += ['', f'## P. Nhóm to theo độ "đa giác" của GT (so với {a.ref})', '',
               '| Run | Nửa | Số ảnh | DSC gộp mốc → run | Bỏ sót bên trong mốc → run (Δ %) | FP mốc → run |', '|---|---|---|---|---|---|']
        md += [f'| {p["run"]} | {p["half"]} | {p["n_images"]} | {vn(p["dsc_ref"])} → {vn(p["dsc_run"])} | {px(p["miss_ref"])} → '
               f'{px(p["miss_run"])} ({vn(p["miss_rel_change_pct"], 1, True)} %) | {px(p["fp_ref"])} → {px(p["fp_run"])} |' for p in poly]
    with open(os.path.join(a.out_dir, 'mechanism.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f'wrote {a.out_dir}/mechanism.md and csv files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
