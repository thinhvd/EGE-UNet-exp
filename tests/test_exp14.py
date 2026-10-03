'''
Checks for the EXP-14 protocol and loss terms. CPU, about a minute:

    python tests/test_exp14.py [MINI_DATA_ROOT]     # MINI_DATA_ROOT with train/val/test enables the end-to-end checks

1. normalized_depth: 1 at the deepest lesion pixel, small at the contour, 0 outside; scipy == torch; empty / full masks;
   size-invariance (a disc and a disc twice as large give the same depth profile in relative radius);
2. interior hard-positive: hand formula, mask = interior & lesion & u < p_low, the gate carries no gradient,
   zero gradient outside the mask;
3. focal and Tversky: hand formulas, alpha weighting, Tversky asymmetry (a miss costs more than a spill);
4. criterion: names, weighted sum, describe() strings (ASCII, older ones unchanged), unknown options refused;
5. train.py / config: older flag sets produce no new keys; EXP-14 flags produce exactly theirs; --test-split;
6. (with MINI_DATA_ROOT) split_dataset pairing; one epoch of train.py with --test-split writes test_results.json on
   the test folder; eval_per_image --split test --interior-tau writes the interior columns.
'''
import os
import sys
import csv
import json
import shutil
import tempfile
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import contour_losses as CL                                           # noqa: E402
from utils import GT_BceDiceLoss                                      # noqa: E402

FAILED = []
H = W = 256


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def disc(r, cy=H / 2, cx=W / 2):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float().view(1, 1, H, W)


def main():
    mini = sys.argv[1] if len(sys.argv) > 1 else None
    torch.manual_seed(0)

    print('1. normalized depth')
    d = CL.normalized_depth(disc(60))[0, 0]
    check('max 1 at the centre, 0 outside, contour <= 1/60 + eps', abs(float(d.max()) - 1) < 1e-6 and float(d[0, 0]) == 0
          and float(d[128, 128 - 60]) <= 1.5 / 60 and float(d[128, 128]) > 0.98)
    t = torch.cat([disc(40), disc(15, 60, 190)])
    dt = CL.normalized_depth(t, backend='torch')
    check('torch path == scipy path', float((dt - CL.normalized_depth(t)).abs().max()) < 1e-5)
    check('empty mask -> 0, full mask -> 1', float(CL.normalized_depth(torch.zeros(1, 1, H, W)).sum()) == 0
          and float(CL.normalized_depth(torch.ones(1, 1, H, W)).min()) == 1)
    small, big = CL.normalized_depth(disc(30))[0, 0], CL.normalized_depth(disc(60))[0, 0]
    fs = float((small >= 0.5).sum()) / float((disc(30) > 0).sum())
    fb = float((big >= 0.5).sum()) / float((disc(60) > 0).sum())
    check('size-invariant: share of a disc with depth >= 0.5 is ~0.25 for radius 30 and 60', abs(fs - 0.25) < 0.03 and abs(fb - 0.25) < 0.02,
          f'{fs:.3f} / {fb:.3f}')

    print('2. interior hard-positive')
    target = disc(60)
    u = torch.full((1, 1, H, W), 0.3)
    u[..., 120:136, 120:136] = 0.8                                   # confident core: not penalized
    ihp = CL.InteriorHardPositive(tau=0.2, gamma=2.0, p_low=0.5)
    m = ((CL.normalized_depth(target) >= 0.2) & (target >= 0.5) & (u < 0.5)).float()
    hand = float((m * 0.7 ** 2 * -np.log(0.3)).sum() / u.numel())
    check('value = sum_mask (1-u)^2 (-log u) / N', abs(float(ihp(u, target)) - hand) < 1e-6, f'{float(ihp(u, target)):.6f} vs {hand:.6f}')
    check('mask excludes the confident core, the rim (depth < 0.2) and the background',
          float(ihp.mask(u, target)[..., 120:136, 120:136].sum()) == 0 and float(ihp.mask(u, target)[..., 128, 128 - 59]) == 0
          and float(ihp.mask(u, target)[..., 0, 0]) == 0)
    uu = u.clone().requires_grad_(True)
    ihp(uu, target).backward()
    check('gradient negative on masked pixels, zero elsewhere', float(uu.grad[m > 0].max()) < 0 and float(uu.grad[m == 0].abs().max()) == 0)

    print('3. focal and Tversky')
    g = target
    u = torch.rand(1, 1, H, W).clamp(0.01, 0.99)
    fl = CL.FocalLoss(alpha=0.75, gamma=2.0)
    hand = float((-0.75 * g * (1 - u) ** 2 * torch.log(u) - 0.25 * (1 - g) * u ** 2 * torch.log(1 - u)).mean())
    check('focal = hand formula', abs(float(fl(u, g)) - hand) < 1e-6)
    check('focal alpha 0.5, gamma 0 = 0.5 * BCE', abs(float(CL.FocalLoss(0.5, 0.0)(u, g)) - 0.5 * float(F.binary_cross_entropy(u, g))) < 1e-6)
    ue = torch.cat([torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)], 2).requires_grad_(True)
    ge = torch.cat([torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)], 2)
    fl(ue, ge).backward()
    check('focal: finite value and gradient at u = 0 / 1 (fully wrong pixels)', bool(torch.isfinite(ue.grad).all()))
    tv = CL.TverskyLoss(alpha=0.3, beta=0.7)
    tp, fp, fn = float((u * g).sum()), float((u * (1 - g)).sum()), float(((1 - u) * g).sum())
    check('tversky = hand formula', abs(float(tv(u, g)) - (1 - (tp + 1) / (tp + 0.3 * fp + 0.7 * fn + 1))) < 1e-6)
    miss = g.clone()
    miss[..., :, :128] = 0                                            # half the lesion missed
    spill = torch.clamp(g + disc(85) * (1 - g), 0, 1)                 # a ring of spill of similar area
    n_miss, n_spill = float((g - miss).sum()), float((spill - g).sum())
    check('tversky: a miss costs more than a spill of similar size', float(tv(miss, g)) > float(tv(spill, g)),
          f'miss {n_miss:.0f} px -> {float(tv(miss, g)):.3f}, spill {n_spill:.0f} px -> {float(tv(spill, g)):.3f}')

    print('4. criterion and describe()')
    crit = CL.build_criterion(extra_term='ihp', extra_weight='2.5', ihp_tau=0.25, ihp_gamma=2.0, ihp_plow=0.5)
    check('ihp criterion: one term named ihp with the given tau', crit.names == ['ihp'] and crit.terms[0].tau == 0.25)
    gp = tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5))
    out = torch.sigmoid(torch.randn(2, 1, H, W))
    tt = torch.cat([disc(60), disc(20, 70, 180)])
    expect = float(GT_BceDiceLoss(1, 1)(gp, out, tt)) + 2.5 * float(CL.InteriorHardPositive(0.25, 2.0, 0.5)(out, tt))
    check('loss = original + 2.5 * ihp', abs(float(crit(gp, out, tt)) - expect) < 1e-5)
    s_ihp = CL.describe('bcedice', 'ihp', '2.5', ihp_tau=0.25, ihp_gamma=2.0, ihp_plow=0.5)
    s_fl = CL.describe('bcedice', 'focal', '1.3', focal_alpha=0.75, focal_gamma=2.0)
    s_tv = CL.describe('bcedice', 'tversky', '0.4', tversky_alpha=0.3, tversky_beta=0.7)
    check('describe() strings', s_ihp == 'bcedice + 2.5 * ihp (ihp depth>=0.25, p<0.5, gamma 2)'
          and s_fl == 'bcedice + 1.3 * focal (focal alpha 0.75, gamma 2)'
          and s_tv == 'bcedice + 0.4 * tversky (tversky alpha 0.3, beta 0.7)' and (s_ihp + s_fl + s_tv).isascii(), f'{s_ihp} | {s_fl} | {s_tv}')
    check('older describe() unchanged', CL.describe('bcedice', 'bl', '0.095') == 'bcedice + 0.095 * bl'
          and CL.describe(extra_term='hp', extra_weight='1.2', hp_radius=10.0, hp_gamma=2.0) == 'bcedice + 1.2 * hp (hp r=10px, gamma 2)')
    check('defaults still build the bare GT_BceDiceLoss', type(CL.build_criterion()) is GT_BceDiceLoss)
    try:
        CL.build_criterion(extra_term='ihp', extra_weight='1', ihp_taux=0.2)
        check('unknown loss option refused', False)
    except ValueError:
        check('unknown loss option refused', True)

    print('5. train.py / config keys')
    py = sys.executable
    probe = ('import sys, json; sys.argv = ["train.py"] + json.loads(sys.argv[1]); import train; '
             'from configs.config_setting import get_config; c = get_config(train.parse_args()); '
             'print("CFG" + json.dumps({"loss": sorted(getattr(c, "loss_config", {}).keys()), "test_split": c.test_split}))')

    def cfg(flags):
        p = subprocess.run([py, '-c', probe, json.dumps(flags)], cwd=REPO, capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('CFG')]
        return p.returncode, (json.loads(line[0][3:]) if line else None)
    base = ['--work-dir', '/tmp/x', '--device', 'cpu']
    csf = ['--fusion', 'sum', '--fusion-stages', 'shallow3', '--fusion-dim', '8']
    rc, c0 = cfg(base + csf)
    check('CSF flags: no loss_config, test_split None', rc == 0 and c0 == {'loss': [], 'test_split': None}, str(c0))
    rc, c1 = cfg(base + csf + ['--extra-term', 'hp', '--extra-weight', '1.2'])
    check('EXP-11 hp flags: no ihp/focal/tversky keys', rc == 0 and not any(k.startswith(('ihp', 'focal', 'tversky')) for k in c1['loss']), str(c1))
    for term, keys in (('ihp', {'ihp_tau', 'ihp_gamma', 'ihp_plow'}), ('focal', {'focal_alpha', 'focal_gamma'}),
                       ('tversky', {'tversky_alpha', 'tversky_beta'})):
        rc, cc = cfg(base + csf + ['--extra-term', term, '--extra-weight', '1', '--test-split', 'test'])
        extra = {k for k in cc['loss'] if k.startswith(('ihp', 'focal', 'tversky'))} if cc else None
        check(f'{term} flags: exactly its keys + test_split', rc == 0 and extra == keys and cc['test_split'] == 'test', str(cc))

    if mini and os.path.isdir(os.path.join(mini, 'test', 'images')):
        print('6. end-to-end on the mini split')
        from configs.config_setting import setting_config, build_transforms
        from datasets.dataset import split_dataset, NPY_datasets
        setting_config.datasets = 'isic17'
        build_transforms(setting_config)
        ds_t = split_dataset(mini, setting_config, 'test')
        ds_v = split_dataset(mini, setting_config, 'val')
        pairs_ok = all(os.path.basename(m).startswith(os.path.splitext(os.path.basename(i))[0]) for i, m in ds_t.data)
        check('split_dataset: test folder, sorted pairs, val unchanged', len(ds_t) == len(os.listdir(os.path.join(mini, 'test', 'images')))
              and pairs_ok and ds_v.data == NPY_datasets(mini, setting_config, train=False).data)
        wd = tempfile.mkdtemp()
        p = subprocess.run([py, '-W', 'ignore', 'train.py', '--work-dir', wd, '--data-path', mini, '--dataset', 'isic17', '--epochs', '1',
                            '--batch-size', '8', '--device', 'cpu', '--seed', '42', '--no-resume', *csf, '--extra-term', 'ihp',
                            '--extra-weight', '1', '--test-split', 'test'], cwd=REPO, capture_output=True, text=True)
        tr = json.load(open(os.path.join(wd, 'test_results.json'))) if os.path.isfile(os.path.join(wd, 'test_results.json')) else {}
        log = open(os.path.join(wd, 'log', 'train.info.log')).read() if os.path.isdir(os.path.join(wd, 'log')) else ''
        check('train.py --test-split test: test_results.json on the test folder, ihp logged', p.returncode == 0
              and tr.get('test_split') == 'test' and tr.get('n_test_images') == len(ds_t) and 'test split: test' in log
              and 'train_extra' in open(os.path.join(wd, 'metrics.csv')).readline(), p.stderr[-300:])
        out_csv = os.path.join(wd, 'per_test.csv')
        p = subprocess.run([py, '-W', 'ignore', 'analysis/eval_per_image.py', '--checkpoint', wd, '--data-path', mini, '--dataset', 'isic17',
                            '--split', 'test', '--interior-tau', '0.2', '--out-csv', out_csv], cwd=REPO, capture_output=True, text=True)
        rows = list(csv.DictReader(open(out_csv))) if os.path.isfile(out_csv) else []
        js = json.load(open(os.path.splitext(out_csv)[0] + '.json')) if rows else {}
        pooled_eval = js.get('pooled', {}).get('f1_or_dsc', -1)
        check('eval_per_image --split test: interior columns, pooled DSC == test_results.json',
              len(rows) == len(ds_t) and 'fn_interior' in rows[0] and all(int(r['fn_interior']) <= int(r['gt_interior']) for r in rows)
              and abs(pooled_eval - tr.get('f1_or_dsc', -2)) < 1e-9, f'{pooled_eval} vs {tr.get("f1_or_dsc")}')
        shutil.rmtree(wd, ignore_errors=True)
    else:
        print('6. [skip] no MINI_DATA_ROOT with a test folder given')

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
