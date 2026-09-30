'''
EXP-10 mechanism readouts (dec5 boundary residual refinement, models/refine.py). CPU or GPU, one pass over
the val split per model plus one per knockout. Nothing here decides win / lose (that is pooled DSC,
exp06_summary.py); these numbers say whether the residual engages at the final mask, where, and how.

  R4 errors by distance to the true contour for every run (≤ 3 / 3-10 / > 10 px, FP outside, FN inside),
     per size tertile, pooled / per-image precision and recall, paired per-image tests of the error counts
     (every run against the first, plus --pair VAR=REF). Reuses analysis/exp09_mechanism.py.
  R1a same-checkpoint knockout of the residual (refine.enabled = False, i.e. out = upsample(z_base)):
     pooled DSC and errors per bin against the intact model, and every pixel whose decision flips, by
     distance bin and size tertile, split into flips towards the truth and away from it. The final conv
     and the residual were trained together, so this measures whether and where the residual acts, NOT
     whether it helps; benefit is read across arms.
  R1b gate knockouts on the same checkpoints: B2 with gate = 1 everywhere and with gate = the true
     contour zone at 128; B3 with its residual restricted to the true zone.
  R2 boundary-map quality (gated runs): soft Dice of B5 with the true zone at 128, mean B5 inside / outside
     the zone, share of cells with B5 > 0.5 against the true zone coverage, precision / recall of B5 > 0.5.
  R3 leverage: RMS(applied residual) / RMS(z_base) on zone cells and overall, share of cells where the
     applied residual exceeds 0.5 logit, where its |mass| lies by distance to the contour and inside the
     true zone, delta.bias, and the mean signed residual on zone cells inside / outside the lesion.
  Threshold control: best pooled DSC over the threshold sweep of every run (per_image_metrics_full.json),
     to tell a real refinement from a zone-restricted threshold shift.
  R5 zone loss during training (gated runs): train_extra_bnd_dec5 over epochs 1-10 vs 290-300 and its
     weighted share of train_loss at the end.

Outputs in --out-dir: mechanism.md plus one csv per readout.
'''
import os
import sys
import csv
import json
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                                    # noqa: E402
from analysis.exp09_mechanism import (GroundTruth, pooled, stratum_index, error_tests,  # noqa: E402
                                      write_csv, vn, px, BINS, STRATA)
from datasets.dataset import NPY_datasets                                           # noqa: E402
import contour_losses as CL                                                         # noqa: E402

TERTILES = ('small', 'mid', 'large')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR',
                   help='repeat; the first run is the reference of the default paired tests')
    p.add_argument('--pair', action='append', default=[], metavar='VAR=REF')
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--near', type=float, default=3.0)
    p.add_argument('--mid', type=float, default=10.0)
    p.add_argument('--radius', type=float, default=10.0, help='contour-zone radius used in training (px at 256)')
    p.add_argument('--zone-weight', type=float, default=0.3, help='weight of the zone loss in training (for R5)')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


class RefineProbe:
    '''Hooks on model.final (z_base) and model.refine (applied residual, raw residual, boundary logit).'''

    def __init__(self, model):
        self.rec = {}
        self.handles = [model.final.register_forward_hook(self._hook('z_base')),
                        model.refine.register_forward_hook(self._hook_refine())]

    def _hook(self, key):
        def h(mod, inp, out):
            self.rec[key] = out.detach()
        return h

    def _hook_refine(self):
        def h(mod, inp, out):
            self.rec['applied'], self.rec['raw'] = out[0].detach(), out[1].detach()
            self.rec['logit'] = None if out[2] is None else out[2].detach()
        return h

    def remove(self):
        for h in self.handles:
            h.remove()


def zone_maps(msk, radius, grid):
    '''True contour zone at full resolution (bool, H x W) and at the refine grid (float, 1 x 1 x g x g).'''
    t = msk.float()[None]
    z256 = CL.boundary_zone(t, radius)
    return z256[0, 0].numpy() > 0.5, CL.band_at_grid(z256, grid)


def refine_pass(model, ds, gt, device, threshold, radius, collect):
    '''One pass over the val images of a refine model. Returns (rows_intact, rows_off) and fills
    `collect` with flip counts, B5 quality and leverage accumulators.'''
    probe = RefineProbe(model)
    rows_i, rows_o = [], []
    fl = {(st, b, d): 0 for st in TERTILES for b in BINS for d in ('to_truth', 'away')}
    bq = dict(inter=0.0, sum_b=0.0, sum_z=0.0, b_in=0.0, n_in=0, b_out=0.0, n_out=0, pos=0, tp=0, cells=0)
    lv = dict(sq_app_zone=0.0, sq_base_zone=0.0, sq_app=0.0, sq_base=0.0, big=0, big_zone=0, cells=0, cells_zone=0,
              mass=np.zeros(3), mass_zone=0.0, mass_tot=0.0, sgn_in=0.0, n_in=0, sgn_out=0.0, n_out=0)
    with torch.no_grad():
        for i in range(gt.n):
            img, msk = ds[i]
            _, out = model(img.float()[None].to(device))
            prob_i = out[0, 0].cpu().numpy()
            z_base = probe.rec['z_base']
            prob_o = torch.sigmoid(F.interpolate(z_base, scale_factor=(2, 2), mode='bilinear', align_corners=True))[0, 0].cpu().numpy()
            pi, po = prob_i >= threshold, prob_o >= threshold
            rows_i.append(gt.counts(i, pi))
            rows_o.append(gt.counts(i, po))
            g = gt.gt[i]
            code = np.where(g, gt.fn_code[i], gt.fp_code[i])
            flip = pi != po
            toward = flip & (pi == g)
            tert = TERTILES[gt.tertile[i]]
            for k, b in enumerate(BINS):
                fl[(tert, b, 'to_truth')] += int((toward & (code == k)).sum())
                fl[(tert, b, 'away')] += int((flip & ~toward & (code == k)).sum())
            grid = z_base.shape[2:4]
            z256, zg = zone_maps(msk, radius, grid)
            zg = zg.to(device)
            app = probe.rec['applied']
            zmask = zg > 0.5
            lv['sq_app_zone'] += float((app[zmask] ** 2).sum())
            lv['sq_base_zone'] += float((z_base[zmask] ** 2).sum())
            lv['sq_app'] += float((app ** 2).sum())
            lv['sq_base'] += float((z_base ** 2).sum())
            lv['big'] += int((app.abs() > 0.5).sum())
            lv['big_zone'] += int((app.abs()[zmask] > 0.5).sum())
            lv['cells'] += app.numel()
            lv['cells_zone'] += int(zmask.sum())
            w = F.interpolate(app, scale_factor=(2, 2), mode='bilinear', align_corners=True)[0, 0].abs().cpu().numpy()
            for k in range(3):
                lv['mass'][k] += float(w[code == k].sum())
            lv['mass_zone'] += float(w[z256].sum())
            lv['mass_tot'] += float(w.sum())
            lesion_g = F.adaptive_avg_pool2d(msk.float()[None].to(device), grid) > 0.5
            lv['sgn_in'] += float(app[zmask & lesion_g].sum())
            lv['n_in'] += int((zmask & lesion_g).sum())
            lv['sgn_out'] += float(app[zmask & ~lesion_g].sum())
            lv['n_out'] += int((zmask & ~lesion_g).sum())
            if probe.rec['logit'] is not None:
                B = torch.sigmoid(probe.rec['logit'])
                bq['inter'] += float((B * zg).sum())
                bq['sum_b'] += float(B.sum())
                bq['sum_z'] += float(zg.sum())
                bq['b_in'] += float(B[zmask].sum())
                bq['n_in'] += int(zmask.sum())
                bq['b_out'] += float(B[~zmask].sum())
                bq['n_out'] += int((~zmask).sum())
                pos = B > 0.5
                bq['pos'] += int(pos.sum())
                bq['tp'] += int((pos & zmask).sum())
                bq['cells'] += B.numel()
    probe.remove()
    collect.update(flips=fl, bq=bq, lv=lv)
    return rows_i, rows_o


def override_pass(model, ds, gt, device, threshold, radius, how):
    '''Pass with an inference-time gate override: 'ones' or 'zone' (the true zone of each image).'''
    rows = []
    with torch.no_grad():
        for i in range(gt.n):
            img, msk = ds[i]
            if how == 'ones':
                model.refine.gate_override = 'ones'
            else:
                model.refine.gate_override = zone_maps(msk, radius, (msk.shape[-2] // 2, msk.shape[-1] // 2))[1].to(device)
            _, out = model(img.float()[None].to(device))
            rows.append(gt.counts(i, out[0, 0].cpu().numpy() >= threshold))
    model.refine.gate_override = None
    return rows


def plain_pass(model, ds, gt, device, threshold):
    rows = []
    with torch.no_grad():
        for i in range(gt.n):
            img, _ = ds[i]
            _, out = model(img.float()[None].to(device))
            rows.append(gt.counts(i, out[0, 0].cpu().numpy() >= threshold))
    return rows


def delta_row(label, variant, rows, ref_rows, gt):
    kp, ip = pooled(rows, range(gt.n)), pooled(ref_rows, range(gt.n))
    ks, isml = pooled(rows, stratum_index(gt, 'small')), pooled(ref_rows, stratum_index(gt, 'small'))
    kl, il = pooled(rows, stratum_index(gt, 'large')), pooled(ref_rows, stratum_index(gt, 'large'))
    return {'run': label, 'variant': variant, 'dsc_pooled': kp['dsc'], 'delta_dsc_pooled': kp['dsc'] - ip['dsc'],
            'delta_dsc_small': ks['dsc'] - isml['dsc'], 'delta_dsc_large': kl['dsc'] - il['dsc'],
            **{f'delta_err_{b}': (kp[f'fp_{b}'] + kp[f'fn_{b}']) - (ip[f'fp_{b}'] + ip[f'fn_{b}']) for b in BINS}}


def best_threshold(run_dir):
    path = os.path.join(run_dir, 'analysis', 'per_image_metrics_full.json')
    if not os.path.isfile(path):
        return None
    sweep = json.load(open(path)).get('sweep') or {}
    if not sweep:
        return None
    t, v = max(((float(k), 100 * s['pooled']['f1_or_dsc']) for k, s in sweep.items()), key=lambda kv: kv[1])
    at05 = 100 * sweep.get('0.5', {}).get('pooled', {}).get('f1_or_dsc', float('nan'))
    return {'best_threshold': t, 'best_dsc_pooled': v, 'dsc_pooled_at_0.5': at05}


def zone_loss_row(label, run_dir, weight):
    path = os.path.join(run_dir, 'metrics.csv')
    if not os.path.isfile(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows or 'train_extra_bnd_dec5' not in rows[0]:
        return None
    first = [float(r['train_extra_bnd_dec5']) for r in rows if int(r['epoch']) <= 10]
    last = [r for r in rows if int(r['epoch']) >= len(rows) - 10]
    z_last = float(np.mean([float(r['train_extra_bnd_dec5']) for r in last]))
    t_last = float(np.mean([float(r['train_loss']) for r in last]))
    z_first = float(np.mean(first))
    return {'run': label, 'zone_first10': z_first, 'zone_last11': z_last, 'change_pct': 100 * (z_last / z_first - 1),
            'weighted_share_of_train_loss_pct': 100 * weight * z_last / t_last, 'epochs': len(rows)}


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = C.pick_device(a.device)
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=False)
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    print(f'{a.dataset}: {n} val images; bins <= {a.near:g} / <= {a.mid:g} / > {a.mid:g} px; zone r = {a.radius:g} px')
    gt = GroundTruth(ds, n, a.near, a.mid)

    runs, rows_by_run, dirs = [], {}, {}
    ko, flips, bq, lev, thr, zl = [], [], [], [], [], []
    for spec in a.run:
        label, path = spec.split('=', 1)
        runs.append(label)
        ckpt = C.resolve_checkpoint(path)
        dirs[label] = C.infer_work_dir(ckpt)
        sd, _ = C.load_state_dict(ckpt)
        model, cfg = C.build_model(sd, device, None, strict=True)
        mode = cfg.get('refine_mode', 'none')
        print(f'  {label}: {os.path.basename(ckpt)}  fusion={cfg["fusion_mode"]}/{cfg["fusion_stages"]}/d{cfg["fusion_dim"]}  refine={mode}')
        t = best_threshold(dirs[label])
        if t:
            thr.append({'run': label, **t})
        if mode == 'none':
            rows_by_run[label] = plain_pass(model, ds, gt, device, a.threshold)
            continue
        col = {}
        rows_i, rows_o = refine_pass(model, ds, gt, device, a.threshold, a.radius, col)
        rows_by_run[label] = rows_i
        ko.append(delta_row(label, 'residual off (out = z_base)', rows_o, rows_i, gt))
        overrides = (('gate = 1', 'ones'), ('gate = true zone', 'zone')) if mode == 'gate' else (('residual only in true zone', 'zone'),)
        for name, how in overrides:
            ko.append(delta_row(label, name, override_pass(model, ds, gt, device, a.threshold, a.radius, how), rows_i, gt))
            print(f'    {name}: pooled DSC delta {ko[-1]["delta_dsc_pooled"]:+.2f}')
        print(f'    residual off: pooled DSC delta {ko[-len(overrides) - 1]["delta_dsc_pooled"]:+.2f}')
        for (st, b, d), v in col['flips'].items():
            flips.append({'run': label, 'tertile': st, 'bin': b, 'direction': d, 'pixels': v})
        q = col['bq']
        if q['cells']:
            bq.append({'run': label, 'soft_dice_zone': 2 * q['inter'] / (q['sum_b'] + q['sum_z']),
                       'mean_B_in_zone': q['b_in'] / max(q['n_in'], 1), 'mean_B_out_zone': q['b_out'] / max(q['n_out'], 1),
                       'cells_B_gt_half_pct': 100 * q['pos'] / q['cells'], 'true_zone_cover_pct': 100 * q['n_in'] / q['cells'],
                       'precision_B_half_pct': 100 * q['tp'] / max(q['pos'], 1), 'recall_B_half_pct': 100 * q['tp'] / max(q['n_in'], 1)})
        v = col['lv']
        lev.append({'run': label, 'rms_ratio_zone_pct': 100 * math.sqrt(v['sq_app_zone'] / max(v['sq_base_zone'], 1e-12)),
                    'rms_ratio_all_pct': 100 * math.sqrt(v['sq_app'] / max(v['sq_base'], 1e-12)),
                    'cells_abs_gt_half_pct': 100 * v['big'] / v['cells'],
                    'zone_cells_abs_gt_half_pct': 100 * v['big_zone'] / max(v['cells_zone'], 1),
                    **{f'mass_{b}_pct': 100 * v['mass'][k] / max(v['mass_tot'], 1e-12) for k, b in enumerate(BINS)},
                    'mass_in_true_zone_pct': 100 * v['mass_zone'] / max(v['mass_tot'], 1e-12),
                    'mean_signed_zone_inside_lesion': v['sgn_in'] / max(v['n_in'], 1),
                    'mean_signed_zone_outside_lesion': v['sgn_out'] / max(v['n_out'], 1),
                    'delta_bias': float(model.refine.delta.bias.detach().cpu()[0])})
        z = zone_loss_row(label, dirs[label], a.zone_weight) if mode == 'gate' else None
        if z:
            zl.append(z)

    summ = []
    for label in runs:
        for st in STRATA:
            p = pooled(rows_by_run[label], stratum_index(gt, st))
            err = p['fp'] + p['fn']
            summ.append({'run': label, 'stratum': st, **{k: p[k] for k in ('tp', 'fp', 'fn', 'dsc', 'precision', 'recall',
                                                                        'precision_img', 'recall_img')},
                         **{f'{e}_{b}': p[f'{e}_{b}'] for e in ('fp', 'fn') for b in BINS},
                         **{f'err_{b}_share_pct': 100 * (p[f'fp_{b}'] + p[f'fn_{b}']) / err if err else float('nan') for b in BINS}})
    pairs = [(r, runs[0]) for r in runs[1:]]
    for spec in a.pair:
        v, r = spec.split('=', 1)
        if v not in rows_by_run or r not in rows_by_run:
            sys.exit(f'--pair {spec}: unknown run label')
        if (v, r) not in pairs:
            pairs.append((v, r))
    tests = []
    for v, r in pairs:
        tests += error_tests(v, r, rows_by_run[v], rows_by_run[r], gt)
    bins_img = [{'run': lab, 'image': gt.names[i], 'tertile': TERTILES[gt.tertile[i]], **rows_by_run[lab][i]}
                for lab in runs for i in range(gt.n)]
    for name, rows in (('bins_per_image', bins_img), ('bins_summary', summ), ('bins_tests', tests), ('knockout', ko),
                       ('flips', flips), ('bquality', bq), ('leverage', lev), ('threshold', thr), ('zone_loss', zl)):
        write_csv(os.path.join(a.out_dir, f'{name}.csv'), rows)

    # ------------------------------------------------------------------ markdown
    md = [f'# EXP-10 — đọc cơ chế ({a.dataset.upper()}, {n} ảnh val)', '',
          f'Khoảng cách tới viền thật: gần ≤ {a.near:g} px, giữa {a.near:g}–{a.mid:g} px, xa > {a.mid:g} px (FP đo ngoài tổn thương, '
          f'FN đo trong). Vùng gần viền dùng khi train: ≤ {a.radius:g} px. Không số nào ở đây quyết thắng thua; knockout trên cùng '
          'checkpoint chỉ cho biết phần sửa có tác động và tác động ở đâu, không cho biết nó có lợi.', '',
          '## R4. Pixel sai theo khoảng cách tới viền', '',
          '| Run | Nhóm | DSC gộp | Precision / recall gộp | Precision / recall theo ảnh | FP gần · giữa · xa | FN gần · giữa · xa | Tỉ lệ lỗi gần · giữa · xa |',
          '|---|---|---|---|---|---|---|---|']
    for r in summ:
        md.append(f'| {r["run"]} | {r["stratum"]} | {vn(r["dsc"])} | {vn(r["precision"])} / {vn(r["recall"])} '
                  f'| {vn(r["precision_img"])} / {vn(r["recall_img"])} | {px(r["fp_near"])} · {px(r["fp_mid"])} · {px(r["fp_far"])} '
                  f'| {px(r["fn_near"])} · {px(r["fn_mid"])} · {px(r["fn_far"])} '
                  f'| {vn(r["err_near_share_pct"], 1)} % · {vn(r["err_mid_share_pct"], 1)} % · {vn(r["err_far_share_pct"], 1)} % |')
    if tests:
        md += ['', '### Kiểm định cặp theo ảnh trên số pixel sai (dương = run nhiều lỗi hơn mốc)', '',
               '| Run | Mốc | Nhóm | Dải | Δ px/ảnh | t | p | tỉ lệ ảnh ít lỗi hơn |', '|---|---|---|---|---|---|---|---|']
        for t in tests:
            md.append(f'| {t["variant"]} | {t["reference"]} | {t["stratum"]} | {t["errors"]} | {vn(t["diff"], 1, True)} '
                      f'| {vn(t["t_stat"], 2, True)} | {t["p_ttest"]:.4f} | {vn(100 * t["frac_fewer_errors"], 0)} % |')
    if thr:
        md += ['', '## Đối chứng dịch ngưỡng (quét ngưỡng 0,3–0,7 của từng run)', '',
               '| Run | DSC gộp ở 0,5 | DSC gộp tốt nhất | ở ngưỡng |', '|---|---|---|---|']
        md += [f'| {r["run"]} | {vn(r["dsc_pooled_at_0.5"])} | {vn(r["best_dsc_pooled"])} | {vn(r["best_threshold"], 2)} |' for r in thr]
    if ko:
        md += ['', '## R1a / R1b. Knockout trên cùng checkpoint (so với chính model đó còn nguyên)', '',
               '| Run | Knockout | DSC gộp | Δ gộp | Δ nhóm nhỏ | Δ nhóm to | Δ lỗi gần | Δ lỗi giữa | Δ lỗi xa |',
               '|---|---|---|---|---|---|---|---|---|']
        for r in ko:
            md.append(f'| {r["run"]} | {r["variant"]} | {vn(r["dsc_pooled"])} | {vn(r["delta_dsc_pooled"], 2, True)} '
                      f'| {vn(r["delta_dsc_small"], 2, True)} | {vn(r["delta_dsc_large"], 2, True)} '
                      f'| {px(r["delta_err_near"], True)} | {px(r["delta_err_mid"], True)} | {px(r["delta_err_far"], True)} |')
    if flips:
        md += ['', '### R1a. Pixel đổi quyết định khi bỏ phần sửa, theo dải và nhóm kích thước', '',
               '"Về đúng" = với phần sửa thì pixel đó đúng, bỏ phần sửa thì sai.', '',
               '| Run | Nhóm | Dải | Về đúng | Về sai | Ròng |', '|---|---|---|---|---|---|']
        key = {}
        for r in flips:
            key.setdefault((r['run'], r['tertile'], r['bin']), {})[r['direction']] = r['pixels']
        for lab in dict.fromkeys(r['run'] for r in flips):
            for b in BINS:
                tot = {d: sum(key[(lab, st, b)][d] for st in TERTILES) for d in ('to_truth', 'away')}
                md.append(f'| {lab} | tất cả | {b} | {px(tot["to_truth"])} | {px(tot["away"])} | {px(tot["to_truth"] - tot["away"], True)} |')
            for st in TERTILES:
                for b in BINS:
                    d = key[(lab, st, b)]
                    md.append(f'| {lab} | {st} | {b} | {px(d["to_truth"])} | {px(d["away"])} | {px(d["to_truth"] - d["away"], True)} |')
    if bq:
        md += ['', '## R2. Chất lượng bản đồ B5 (lưới 128, so với vùng thật)', '',
               '| Run | soft Dice | B5 TB trong vùng | ngoài vùng | % ô B5 > 0,5 | % ô thuộc vùng thật | precision / recall (B5 > 0,5) |',
               '|---|---|---|---|---|---|---|']
        md += [f'| {r["run"]} | {vn(r["soft_dice_zone"], 3)} | {vn(r["mean_B_in_zone"], 3)} | {vn(r["mean_B_out_zone"], 3)} '
               f'| {vn(r["cells_B_gt_half_pct"], 1)} | {vn(r["true_zone_cover_pct"], 1)} '
               f'| {vn(r["precision_B_half_pct"], 1)} / {vn(r["recall_B_half_pct"], 1)} |' for r in bq]
    if lev:
        md += ['', '## R3. Đòn bẩy và vị trí của phần sửa', '',
               '| Run | RMS sửa / RMS z_base: trong vùng · toàn ảnh | % ô \\|sửa\\| > 0,5 logit: toàn ảnh · trong vùng '
               '| Khối lượng \\|sửa\\| theo dải gần · giữa · xa | % khối lượng trong vùng thật | sửa TB trong vùng: trong lesion · ngoài lesion | bias |',
               '|---|---|---|---|---|---|---|']
        md += [f'| {r["run"]} | {vn(r["rms_ratio_zone_pct"], 1)} % · {vn(r["rms_ratio_all_pct"], 1)} % '
               f'| {vn(r["cells_abs_gt_half_pct"], 1)} · {vn(r["zone_cells_abs_gt_half_pct"], 1)} '
               f'| {vn(r["mass_near_pct"], 1)} · {vn(r["mass_mid_pct"], 1)} · {vn(r["mass_far_pct"], 1)} % '
               f'| {vn(r["mass_in_true_zone_pct"], 1)} | {vn(r["mean_signed_zone_inside_lesion"], 3, True)} · '
               f'{vn(r["mean_signed_zone_outside_lesion"], 3, True)} | {vn(r["delta_bias"], 3, True)} |' for r in lev]
    if zl:
        md += ['', '## R5. Loss vùng gần viền trên train', '',
               '| Run | TB epoch 1–10 | TB 11 epoch cuối | thay đổi | phần của train_loss (đã nhân trọng số) |', '|---|---|---|---|---|']
        md += [f'| {r["run"]} | {vn(r["zone_first10"], 3)} | {vn(r["zone_last11"], 3)} | {vn(r["change_pct"], 0, True)} % '
               f'| {vn(r["weighted_share_of_train_loss_pct"], 1)} % |' for r in zl]
    with open(os.path.join(a.out_dir, 'mechanism.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f'wrote {a.out_dir}/mechanism.md and csv files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
