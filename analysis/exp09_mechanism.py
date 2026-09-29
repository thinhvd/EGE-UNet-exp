'''
EXP-9 mechanism readouts (boundary-guided cross-stage fusion). CPU or GPU, one pass over the val split per
model plus one per knockout. Nothing here decides win / lose (that is pooled DSC, exp06_summary.py); these
numbers say WHY a run moved or did not.

  (b) errors by distance to the true contour, for every run: FP and FN pixels within 3 px, 3-10 px and
      beyond 10 px of the ground-truth contour (FP measured outside the lesion, FN inside, as in
      small_lesion_errors.boundary_shares), per lesion-size tertile; pooled / per-image precision and
      recall; the pooled-DSC ceiling if all errors of one distance bin were fixed; paired per-image
      tests of the error counts (every run against the first one, plus --pair VAR=REF).
  (a) source weights read from the checkpoint (sum and bg_stage runs): w_j inside, w_j * (1 + a_j) on
      the contour, and the gain (1 + a_j) relative to the other sources of the same stage (the only
      scale-free reading: the projection's GroupNorm, the head and weight decay rescale all sources of
      a stage alike). A decay-only reference says how much weight decay alone shrank a parameter.
  (a') data-weighted contributions (bg_stage): RMS of w_j (1 + a_j B) P_j over contour-band, lesion
      interior and background cells of each stage's grid, and each source's share of the total.
  (a'') inference-time knockouts (bg_stage): a = 0 everywhere; only the shallow sources' a (t1, t2);
      only the deep sources' a (t4, t5); each source removed (w_j = 0). Delta pooled DSC and delta
      errors per distance bin against the intact model.
  (c) boundary-map quality (bg_stage): BceDice(0.5, 1) and soft Dice of sigmoid(logit) against the
      contour band at each head's grid (the same functions as the training loss), mean B on / off the band.
  (d) leverage (bg_stage): RMS(fused output added to the decoder) / RMS(decoder feature it is added to).

EXP-9 step 0 runs (b) alone on the EXP-8 baselines:
  python analysis/exp09_mechanism.py --dataset isic17 --data-path data/data_isic1718/isic2017 \
      --run A0=results/EGE-UNet-results-exp8/egeunet_isic17_learnable_s42 --out-dir <dir>

Outputs in --out-dir: mechanism.md plus one csv per readout (no spaces in names).
'''
import os
import sys
import csv
import math
import copy
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C                                   # noqa: E402
from analysis.significance_tests import paired_stats, holm         # noqa: E402
from datasets.dataset import NPY_datasets                          # noqa: E402
import contour_losses as CL                                        # noqa: E402
from utils import BceDiceLoss                                      # noqa: E402

BINS = ('near', 'mid', 'far')          # <= NEAR px, (NEAR, MID] px, > MID px from the true contour
STRATA = ('all', 'small', 'mid', 'large')
SOURCES = ('t1', 't2', 't3', 't4', 't5')
SHALLOW, DEEP = (0, 1), (3, 4)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, metavar='LABEL=DIR',
                   help='repeat; the first run is the reference of the default paired tests')
    p.add_argument('--pair', action='append', default=[], metavar='VAR=REF',
                   help='extra paired tests on the error counts, e.g. A3e=A1e')
    p.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    p.add_argument('--data-path', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--near', type=float, default=3.0)
    p.add_argument('--mid', type=float, default=10.0)
    p.add_argument('--batch-size', type=int, default=8, help='training batch size, for the decay-only reference')
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--no-knockout', action='store_true')
    p.add_argument('--max-images', type=int, default=None, help='debug: first N val images only')
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def vn(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}').replace('.', ',')


def px(n, sign=False):
    '''Pixel count with a dot as thousands separator (Vietnamese style), optional sign.'''
    return (f'{n:+,}' if sign else f'{n:,}').replace(',', '.')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- ground truth, once

class GroundTruth:
    '''Binary masks, distance-bin codes, size tertiles and contour bands of the val images.'''

    def __init__(self, ds, n, near, mid):
        self.names, self.gt, self.fp_code, self.fn_code, self.area = [], [], [], [], []
        for i in range(n):
            _, msk = ds[i]
            g = msk.float()[0].numpy() >= 0.5
            self.names.append(os.path.splitext(os.path.basename(ds.data[i][0]))[0])
            self.gt.append(g)
            self.area.append(float(g.mean()))
            if g.any():
                d_out = distance_transform_edt(~g)   # outside: distance to the nearest lesion pixel
                d_in = distance_transform_edt(g)     # inside: distance to the nearest background pixel
            else:
                d_out = np.full(g.shape, np.inf)
                d_in = np.zeros(g.shape)
            self.fp_code.append(self._code(d_out, near, mid))
            self.fn_code.append(self._code(d_in, near, mid))
        a = np.array(self.area)
        self.tertile = np.digitize(a, np.quantile(a, [1 / 3, 2 / 3]))   # 0 small, 1 mid, 2 large
        self.n = n

    @staticmethod
    def _code(d, near, mid):
        return np.where(d <= near, 0, np.where(d <= mid, 1, 2)).astype(np.uint8)

    def counts(self, i, pred):
        g = self.gt[i]
        fp, fn = pred & ~g, ~pred & g
        row = {'tp': int((pred & g).sum()), 'fp': int(fp.sum()), 'fn': int(fn.sum())}
        for k, b in enumerate(BINS):
            row[f'fp_{b}'] = int((fp & (self.fp_code[i] == k)).sum())
            row[f'fn_{b}'] = int((fn & (self.fn_code[i] == k)).sum())
        return row


# ----------------------------------------------------------------------------- one pass over the val split

class FusionProbe:
    '''Hooks on a bg_stage model: records per stage the projections, the decoder feature, the boundary
    logit, the fused feature and the head output; accumulates (a'), (c) and (d).'''

    def __init__(self, model):
        self.f = model.fusion
        self.stages = list(model.fusion_stages)
        self.rec, self.handles = {}, []
        orig_project = self.f.project

        def project(feats):
            p = orig_project(feats)
            self.rec['p'] = p
            return p
        self.f.project = project                              # instance attribute shadows the method
        for s in self.stages:
            self.handles.append(self.f.bnd_heads[s].register_forward_hook(self._hook_bnd(s)))
            self.handles.append(self.f.heads[s].register_forward_hook(self._hook_head(s)))
        self.bd = BceDiceLoss(wb=0.5, wd=1)
        z = lambda: {s: np.zeros(len(SOURCES)) for s in self.stages}
        self.sq = {r: z() for r in ('band', 'interior', 'background')}
        self.cnt = {r: {s: 0 for s in self.stages} for r in ('band', 'interior', 'background')}
        self.lev = {s: [0.0, 0.0] for s in self.stages}
        self.bq = {s: {'bd': 0.0, 'n': 0, 'inter': 0.0, 'sum_b': 0.0, 'sum_t': 0.0,
                       'b_on': 0.0, 'n_on': 0, 'b_off': 0.0, 'n_off': 0} for s in self.stages}
        self.max_gap = 0.0

    def _hook_bnd(self, s):
        def h(mod, inp, out):
            self.rec[('feat', s)] = inp[0].detach()
            self.rec[('logit', s)] = out.detach()
        return h

    def _hook_head(self, s):
        def h(mod, inp, out):
            self.rec[('fused', s)] = inp[0].detach()
            self.rec[('fuse', s)] = out.detach()
        return h

    def update(self, gt_t):
        '''gt_t: (1,1,H,W) float target of the image just forwarded.'''
        band256 = CL.boundary_band(gt_t)
        for s in self.stages:
            feat, logit = self.rec[('feat', s)], self.rec[('logit', s)]
            grid = feat.shape[2:4]
            B = torch.sigmoid(logit)
            w, a = self.f.sum_w[s].detach(), self.f.bg_alpha[s].detach()
            resized = [q if q.shape[2:4] == grid else
                       F.interpolate(q, size=grid, mode='bilinear', align_corners=True) for q in self.rec['p']]
            contrib = [w[j] * (1 + a[j] * B) * resized[j] for j in range(len(SOURCES))]
            self.max_gap = max(self.max_gap, float((sum(contrib) - self.rec[('fused', s)]).abs().max()))
            band = CL.band_at_grid(band256, grid)[0, 0] > 0
            lesion = F.adaptive_avg_pool2d((gt_t > 0.5).float(), grid)[0, 0] > 0.5
            regions = {'band': band, 'interior': lesion & ~band, 'background': ~lesion & ~band}
            for r, m in regions.items():
                nm = int(m.sum())
                if nm == 0:
                    continue
                self.cnt[r][s] += nm
                for j, c in enumerate(contrib):
                    self.sq[r][s][j] += float((c[0][:, m] ** 2).mean(0).sum())   # mean over channels, sum over cells
            fuse = self.rec[('fuse', s)]
            self.lev[s][0] += float((fuse ** 2).sum())
            self.lev[s][1] += float((feat ** 2).sum())
            bandf = band.float()[None, None]
            q = self.bq[s]
            q['bd'] += float(self.bd(B, bandf))
            q['n'] += 1
            q['inter'] += float((B * bandf).sum())
            q['sum_b'] += float(B.sum())
            q['sum_t'] += float(bandf.sum())
            q['b_on'] += float(B[0, 0][band].sum())
            q['n_on'] += int(band.sum())
            q['b_off'] += float(B[0, 0][~band].sum())
            q['n_off'] += int((~band).sum())


def forward_pass(model, ds, gt, device, threshold, probe=None):
    rows = []
    with torch.no_grad():
        for i in range(gt.n):
            img, msk = ds[i]
            _, out = model(img.float()[None].to(device))
            pred = out[0, 0].cpu().numpy() >= threshold
            rows.append(gt.counts(i, pred))
            if probe is not None:
                probe.update(msk.float()[None].to(device))
    return rows


# ----------------------------------------------------------------------------- summaries

def pooled(rows, idx):
    tot = {k: sum(rows[i][k] for i in idx) for k in rows[0]}
    tp, fp, fn = tot['tp'], tot['fp'], tot['fn']
    tot['dsc'] = 100 * 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float('nan')
    tot['precision'] = 100 * tp / (tp + fp) if tp + fp else float('nan')
    tot['recall'] = 100 * tp / (tp + fn) if tp + fn else float('nan')
    prec = [rows[i]['tp'] / (rows[i]['tp'] + rows[i]['fp']) for i in idx if rows[i]['tp'] + rows[i]['fp'] > 0]
    rec = [rows[i]['tp'] / (rows[i]['tp'] + rows[i]['fn']) for i in idx if rows[i]['tp'] + rows[i]['fn'] > 0]
    tot['precision_img'] = 100 * float(np.mean(prec)) if prec else float('nan')
    tot['recall_img'] = 100 * float(np.mean(rec)) if rec else float('nan')
    return tot


def stratum_index(gt, name):
    if name == 'all':
        return list(range(gt.n))
    k = {'small': 0, 'mid': 1, 'large': 2}[name]
    return [i for i in range(gt.n) if gt.tertile[i] == k]


def ceiling_rows(label, rows, gt):
    '''Pooled DSC over ALL images if the errors of one (stratum, bin) were fixed.'''
    allp = pooled(rows, range(gt.n))
    TP, FP, FN = allp['tp'], allp['fp'], allp['fn']
    base = 2 * TP / (2 * TP + FP + FN)
    err_total = FP + FN
    out = []
    for st in STRATA:
        part = pooled(rows, stratum_index(gt, st))
        for b in BINS:
            fpb, fnb = part[f'fp_{b}'], part[f'fn_{b}']
            dsc = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
            out.append({'run': label, 'stratum': st, 'bin': b,
                        'fp_px': fpb, 'fn_px': fnb, 'share_of_all_errors_pct': 100 * (fpb + fnb) / err_total,
                        'gain_fix_fp': 100 * (dsc(TP, FP - fpb, FN) - base),
                        'gain_fix_fn': 100 * (dsc(TP + fnb, FP, FN - fnb) - base),
                        'gain_fix_both': 100 * (dsc(TP + fnb, FP - fpb, FN - fnb) - base)})
    return out


def error_tests(var, ref, rows_v, rows_r, gt):
    out = []
    for st in STRATA:
        idx = stratum_index(gt, st)
        block = []
        for b in BINS + ('total',):
            key = (lambda r: r['fp'] + r['fn']) if b == 'total' else (lambda r, b=b: r[f'fp_{b}'] + r[f'fn_{b}'])
            a = np.array([key(rows_v[i]) for i in idx], dtype=np.float64)
            r = np.array([key(rows_r[i]) for i in idx], dtype=np.float64)
            s = paired_stats(a, r)
            s['frac_fewer_errors'] = float((a < r).mean())   # lower is better for error counts
            s.pop('frac_better', None)
            block.append({'variant': var, 'reference': ref, 'stratum': st, 'errors': b, **s})
        adj = holm([x['p_ttest'] for x in block])
        for x, p in zip(block, adj):
            x['p_ttest_holm_within_stratum'] = float(p)
        out += block
    return out


def decay_reference(run_dir, n_train, batch, wd):
    '''Factor weight decay alone applies to a parameter up to the chosen epoch (AdamW: p *= 1 - lr*wd per step).'''
    import json
    mpath, tpath = os.path.join(run_dir, 'metrics.csv'), os.path.join(run_dir, 'test_results.json')
    if not (os.path.isfile(mpath) and os.path.isfile(tpath)):
        return float('nan')
    min_epoch = json.load(open(tpath)).get('min_epoch')
    steps = math.ceil(n_train / batch)
    logf = 0.0
    for r in csv.DictReader(open(mpath)):
        if int(r['epoch']) <= min_epoch:
            logf += steps * math.log(1 - float(r['lr']) * wd)
    return math.exp(logf)


def alpha_rows(label, sd, decay):
    out = []
    stages = sorted({k.split('.')[2] for k in sd if k.startswith('fusion.sum_w.')})
    for s in stages:
        w = sd[f'fusion.sum_w.{s}'].float().numpy()
        a_key = f'fusion.bg_alpha.{s}'
        a = sd[a_key].float().numpy() if a_key in sd else np.zeros_like(w)
        gain = 1 + a
        for j in range(len(w)):
            out.append({'run': label, 'stage': s, 'source': SOURCES[j], 'w': float(w[j]), 'alpha': float(a[j]),
                        'weight_interior': float(w[j]), 'weight_contour': float(w[j] * gain[j]),
                        'gain': float(gain[j]), 'gain_rel_stage_mean': float(gain[j] / gain.mean()),
                        'decay_only_factor': decay})
    return out


def model_kind(sd):
    cfg = C.infer_model_config(sd)
    return cfg['fusion_mode'], cfg


def knockout_variants(model):
    '''(name, modified deep copy) for the bg_stage knockouts.'''
    out = []
    f = model.fusion

    def with_alpha(fn):
        m = copy.deepcopy(model)
        with torch.no_grad():
            for s in m.fusion_stages:
                fn(m.fusion.bg_alpha[s])
        return m
    out.append(('alpha=0', with_alpha(lambda a: a.zero_())))
    out.append(('alpha_shallow_only', with_alpha(lambda a: a[2:].zero_())))
    out.append(('alpha_deep_only', with_alpha(lambda a: a[:3].zero_())))
    for j, src in enumerate(SOURCES):
        m = copy.deepcopy(model)
        with torch.no_grad():
            for s in m.fusion_stages:
                m.fusion.sum_w[s][j] = 0.0
        out.append((f'drop_{src}', m))
    del f
    return out


# ----------------------------------------------------------------------------- main

def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = C.pick_device(a.device)
    config = C.make_config(a.dataset, 256)
    ds = NPY_datasets(a.data_path, config, train=False)
    n = len(ds) if a.max_images is None else min(a.max_images, len(ds))
    n_train = len(os.listdir(os.path.join(a.data_path, 'train', 'images')))
    print(f'{a.dataset}: {n} val images, {n_train} train images; bins <= {a.near:g} / <= {a.mid:g} / > {a.mid:g} px')
    gt = GroundTruth(ds, n, a.near, a.mid)

    runs, per_run_rows = [], {}
    bins_img, ceil, alpha, contrib, bq, lev, ko, recon = [], [], [], [], [], [], [], []
    max_gap = 0.0
    for spec in a.run:
        label, path = spec.split('=', 1)
        runs.append(label)
        ckpt = C.resolve_checkpoint(path)
        sd, _ = C.load_state_dict(ckpt)
        mode, cfg = model_kind(sd)
        model, _ = C.build_model(sd, device, None, strict=True)
        print(f'  {label}: {os.path.basename(ckpt)}  fusion={mode}/{cfg["fusion_stages"]}/d{cfg["fusion_dim"]}')
        probe = FusionProbe(model) if mode == 'bg_stage' else None
        rows = forward_pass(model, ds, gt, device, a.threshold, probe)
        per_run_rows[label] = rows
        for i, r in enumerate(rows):
            bins_img.append({'run': label, 'image': gt.names[i], 'area_frac': gt.area[i],
                             'tertile': ['small', 'mid', 'large'][gt.tertile[i]], **r})
        ceil += ceiling_rows(label, rows, gt)

        pi = os.path.join(C.infer_work_dir(ckpt), 'analysis', 'per_image_metrics_full.csv')
        if os.path.isfile(pi):
            off = {rr['filename'].rsplit('.', 1)[0]: rr for rr in csv.DictReader(open(pi))}
            gaps = [max(abs(int(off[gt.names[i]][k]) - rows[i][k]) for k in ('tp', 'fp', 'fn'))
                    for i in range(n) if gt.names[i] in off]
            recon.append((label, max(gaps) if gaps else float('nan'), float(np.mean(gaps)) if gaps else float('nan')))

        if mode in ('sum', 'bg_stage'):
            alpha += alpha_rows(label, sd, decay_reference(C.infer_work_dir(ckpt), n_train, a.batch_size,
                                                           a.weight_decay))
        if probe is not None:
            max_gap = max(max_gap, probe.max_gap)
            for s in probe.stages:
                for r in ('band', 'interior', 'background'):
                    cnt = probe.cnt[r][s]
                    rms = np.sqrt(probe.sq[r][s] / cnt) if cnt else np.full(len(SOURCES), np.nan)
                    tot = float(np.nansum(rms))
                    for j, src in enumerate(SOURCES):
                        contrib.append({'run': label, 'stage': s, 'region': r, 'source': src, 'cells': cnt,
                                        'rms': float(rms[j]), 'share_pct': 100 * float(rms[j]) / tot if tot else float('nan')})
                q = probe.bq[s]
                bq.append({'run': label, 'stage': s, 'bd_mean': q['bd'] / q['n'],
                           'soft_dice_pooled': 2 * q['inter'] / (q['sum_b'] + q['sum_t']),
                           'mean_B_on_band': q['b_on'] / max(q['n_on'], 1),
                           'mean_B_off_band': q['b_off'] / max(q['n_off'], 1),
                           'band_cell_frac_pct': 100 * q['n_on'] / max(q['n_on'] + q['n_off'], 1)})
                lev.append({'run': label, 'stage': s,
                            'rms_ratio_fuse_over_feature': math.sqrt(probe.lev[s][0] / probe.lev[s][1])})
            if not a.no_knockout:
                intact = pooled(rows, range(n))
                for name, km in knockout_variants(model):
                    krows = forward_pass(km, ds, gt, device, a.threshold)
                    kp = pooled(krows, range(n))
                    ks, kl = pooled(krows, stratum_index(gt, 'small')), pooled(krows, stratum_index(gt, 'large'))
                    ins, inl = pooled(rows, stratum_index(gt, 'small')), pooled(rows, stratum_index(gt, 'large'))
                    ko.append({'run': label, 'knockout': name, 'dsc_pooled': kp['dsc'],
                               'delta_dsc_pooled': kp['dsc'] - intact['dsc'],
                               'delta_dsc_pooled_small': ks['dsc'] - ins['dsc'],
                               'delta_dsc_pooled_large': kl['dsc'] - inl['dsc'],
                               **{f'delta_err_{b}': (kp[f'fp_{b}'] + kp[f'fn_{b}']) - (intact[f'fp_{b}'] + intact[f'fn_{b}'])
                                  for b in BINS}})
                    print(f'    knockout {name}: pooled DSC {kp["dsc"]:.2f} ({kp["dsc"] - intact["dsc"]:+.2f})')

    summ = []
    for label in runs:
        for st in STRATA:
            p = pooled(per_run_rows[label], stratum_index(gt, st))
            err = p['fp'] + p['fn']
            summ.append({'run': label, 'stratum': st, 'images': len(stratum_index(gt, st)),
                         **{k: p[k] for k in ('tp', 'fp', 'fn', 'dsc', 'precision', 'recall',
                                              'precision_img', 'recall_img')},
                         **{f'{e}_{b}': p[f'{e}_{b}'] for e in ('fp', 'fn') for b in BINS},
                         **{f'err_{b}_share_pct': 100 * (p[f'fp_{b}'] + p[f'fn_{b}']) / err if err else float('nan')
                            for b in BINS}})
    pairs = [(r, runs[0]) for r in runs[1:]]
    for spec in a.pair:
        v, r = spec.split('=', 1)
        if v not in per_run_rows or r not in per_run_rows:
            sys.exit(f'--pair {spec}: unknown run label')
        pairs.append((v, r))
    tests = []
    for v, r in pairs:
        tests += error_tests(v, r, per_run_rows[v], per_run_rows[r], gt)

    for name, rows in (('bins_per_image', bins_img), ('bins_summary', summ), ('ceiling', ceil),
                       ('bins_tests', tests), ('alpha', alpha), ('contrib', contrib),
                       ('bquality', bq), ('leverage', lev), ('knockout', ko)):
        write_csv(os.path.join(a.out_dir, f'{name}.csv'), rows)

    # ------------------------------------------------------------------ markdown
    md = [f'# EXP-9 — đọc cơ chế ({a.dataset.upper()}, {n} ảnh val)', '',
          f'Khoảng cách tới viền thật: gần ≤ {a.near:g} px, giữa {a.near:g}–{a.mid:g} px, xa > {a.mid:g} px. '
          'FP đo ở phía ngoài tổn thương, FN ở phía trong. Nhóm nhỏ/vừa/to = ba phần bằng nhau theo diện tích '
          'tổn thương của dataset này. Không số nào ở đây quyết thắng thua.', '']
    if recon:
        md += ['Đối chiếu với `per_image_metrics_full.csv` (TP/FP/FN theo ảnh, lệch lớn nhất / trung bình): ' +
               '; '.join(f'{lab} {vn(g, 0)} / {vn(m)} px' for lab, g, m in recon), '']
    md += ['## (b) Pixel sai theo khoảng cách tới viền', '',
           '| Run | Nhóm | DSC gộp | Precision gộp | Recall gộp | Precision theo ảnh | Recall theo ảnh '
           '| FP gần · giữa · xa | FN gần · giữa · xa | Tỉ lệ lỗi gần · giữa · xa |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in summ:
        md.append(f'| {r["run"]} | {r["stratum"]} | {vn(r["dsc"])} | {vn(r["precision"])} | {vn(r["recall"])} '
                  f'| {vn(r["precision_img"])} | {vn(r["recall_img"])} '
                  f'| {px(r["fp_near"])} · {px(r["fp_mid"])} · {px(r["fp_far"])} '
                  f'| {px(r["fn_near"])} · {px(r["fn_mid"])} · {px(r["fn_far"])} '
                  f'| {vn(r["err_near_share_pct"], 1)} % · {vn(r["err_mid_share_pct"], 1)} % · {vn(r["err_far_share_pct"], 1)} % |')
    md += ['', f'## Trần DSC gộp nếu sửa hết lỗi của một dải (run {runs[0]})', '',
           '| Nhóm | Dải | FP px | FN px | % tổng lỗi | + DSC nếu sửa FP | + DSC nếu sửa FN | + DSC nếu sửa cả hai |',
           '|---|---|---|---|---|---|---|---|']
    for r in [c for c in ceil if c['run'] == runs[0]]:
        md.append(f'| {r["stratum"]} | {r["bin"]} | {px(r["fp_px"])} | {px(r["fn_px"])} | {vn(r["share_of_all_errors_pct"], 1)} '
                  f'| {vn(r["gain_fix_fp"], 2, True)} | {vn(r["gain_fix_fn"], 2, True)} | {vn(r["gain_fix_both"], 2, True)} |')
    if tests:
        md += ['', '## Kiểm định cặp theo ảnh trên số pixel sai (dương = run nhiều lỗi hơn mốc)', '',
               '| Run | Mốc | Nhóm | Dải | Δ px/ảnh | t | p | tỉ lệ ảnh ít lỗi hơn |', '|---|---|---|---|---|---|---|---|']
        for t in tests:
            md.append(f'| {t["variant"]} | {t["reference"]} | {t["stratum"]} | {t["errors"]} | {vn(t["diff"], 1, True)} '
                      f'| {vn(t["t_stat"], 2, True)} | {t["p_ttest"]:.4f} | {vn(100 * t["frac_fewer_errors"], 0)} % |')
    if alpha:
        md += ['', '## (a) Trọng số nguồn đọc từ checkpoint', '',
               'Chỉ tỉ số giữa các nguồn trong cùng tầng có nghĩa. "Hệ số (1+α) so với TB tầng" > 1 nghĩa là nguồn đó '
               'được ưu tiên ở viền hơn các nguồn khác của tầng. Mốc "chỉ do weight decay" là hệ số mà riêng weight '
               'decay đã nhân vào mọi tham số tới epoch được chọn.', '',
               '| Run | Tầng | Nguồn | w (vùng trong) | w·(1+α) (viền) | 1+α | (1+α)/TB tầng | chỉ do weight decay |',
               '|---|---|---|---|---|---|---|---|']
        for r in alpha:
            md.append(f'| {r["run"]} | {r["stage"]} | {r["source"]} | {vn(r["weight_interior"], 3)} '
                      f'| {vn(r["weight_contour"], 3)} | {vn(r["gain"], 3)} | {vn(r["gain_rel_stage_mean"], 3)} '
                      f'| {vn(r["decay_only_factor"], 3)} |')
    if contrib:
        md += ['', "## (a') Đóng góp theo dữ liệu (RMS của w·(1+αB)·P, tỉ trọng trong vùng)", '',
               '| Run | Tầng | Vùng | ' + ' | '.join(SOURCES) + ' | nông t1+t2 | sâu t4+t5 |',
               '|---|---|---|' + '---|' * (len(SOURCES) + 2)]
        keyed = {}
        for r in contrib:
            keyed.setdefault((r['run'], r['stage'], r['region']), {})[r['source']] = r['share_pct']
        for (run, s, reg), sh in keyed.items():
            md.append(f'| {run} | {s} | {reg} | ' + ' | '.join(vn(sh[x], 1) for x in SOURCES) +
                      f' | {vn(sh["t1"] + sh["t2"], 1)} | {vn(sh["t4"] + sh["t5"], 1)} |')
        md += ['', f'Kiểm tra công thức: |Σ đóng góp − feature fused thật| lớn nhất = {max_gap:.2e}.']
    if bq:
        md += ['', '## (c) Chất lượng bản đồ viền B (cùng hàm với loss, tại lưới từng head)', '',
               '| Run | Tầng | BceDice(0,5; 1) | soft Dice | B trung bình trên dải | B ngoài dải | % ô thuộc dải |',
               '|---|---|---|---|---|---|---|']
        for r in bq:
            md.append(f'| {r["run"]} | {r["stage"]} | {vn(r["bd_mean"], 3)} | {vn(r["soft_dice_pooled"], 3)} '
                      f'| {vn(r["mean_B_on_band"], 3)} | {vn(r["mean_B_off_band"], 3)} | {vn(r["band_cell_frac_pct"], 1)} |')
    if lev:
        md += ['', '## (d) Đòn bẩy của nhánh: RMS(phần cộng vào) / RMS(feature decoder)', '',
               '| Run | Tầng | Tỉ số |', '|---|---|---|']
        md += [f'| {r["run"]} | {r["stage"]} | {vn(100 * r["rms_ratio_fuse_over_feature"], 1)} % |' for r in lev]
    if ko:
        md += ['', "## (a'') Knockout lúc suy luận (so với chính model đó còn nguyên)", '',
               '| Run | Knockout | DSC gộp | Δ DSC gộp | Δ nhóm nhỏ | Δ nhóm to | Δ lỗi gần | Δ lỗi giữa | Δ lỗi xa |',
               '|---|---|---|---|---|---|---|---|---|']
        for r in ko:
            md.append(f'| {r["run"]} | {r["knockout"]} | {vn(r["dsc_pooled"])} | {vn(r["delta_dsc_pooled"], 2, True)} '
                      f'| {vn(r["delta_dsc_pooled_small"], 2, True)} | {vn(r["delta_dsc_pooled_large"], 2, True)} '
                      f'| {px(r["delta_err_near"], True)} | {px(r["delta_err_mid"], True)} | {px(r["delta_err_far"], True)} |')
    with open(os.path.join(a.out_dir, 'mechanism.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f'wrote {a.out_dir}/mechanism.md and csv files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
