'''
EXP-17 checks (CPU): split_dataset without a val folder, the --no-val training path, the LSA + E3a criterion,
and the 80/20 split lists.

    python tests/test_exp17.py [MINI_DATA_ROOT]     # MINI_DATA_ROOT with train/val/test enables the training checks
'''
import os
import sys
import csv
import json
import glob
import shutil
import tempfile
import subprocess

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis import common as C                          # noqa: E402
from datasets.dataset import NPY_datasets, split_dataset  # noqa: E402
import contour_losses as CL                               # noqa: E402
from models.fusion import DeepSupervisionOutputs          # noqa: E402
from utils import GT_BceDiceLoss                          # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def old_split_dataset(path, config, split):
    '''The EXP-14 implementation, for comparison.'''
    ds = NPY_datasets(path, config, train=False)
    if split != 'val':
        idir, mdir = os.path.join(path, split, 'images'), os.path.join(path, split, 'masks')
        ds.data = [[os.path.join(idir, i), os.path.join(mdir, m)] for i, m in zip(sorted(os.listdir(idir)), sorted(os.listdir(mdir)))]
    return ds


def main():
    mini = sys.argv[1] if len(sys.argv) > 1 else None
    py = sys.executable
    print('1. split_dataset')
    config = C.make_config('isic17', 256)
    if mini and os.path.isdir(os.path.join(mini, 'test', 'images')):
        for split in ('val', 'test', 'train'):
            a, b = split_dataset(mini, config, split), old_split_dataset(mini, config, split)
            same = a.data == b.data and a.transformer is b.transformer and torch.equal(a[0][0], b[0][0]) and torch.equal(a[0][1], b[0][1])
            check(f'split {split}: same files, transform and first sample as the EXP-14 version', same)
        mini80 = tempfile.mkdtemp()
        for split in ('train', 'test'):
            shutil.copytree(os.path.join(mini, split), os.path.join(mini80, split))
        try:
            ds = split_dataset(mini80, config, 'test')
            ok = len(ds) == len(os.listdir(os.path.join(mini80, 'test', 'images')))
        except FileNotFoundError as e:
            ok, ds = False, str(e)
        check('split_dataset works without a val folder', ok)
    else:
        mini80 = None
        print('  [skip] no MINI_DATA_ROOT')

    print('2. LSA + E3a criterion')
    kw = dict(loss='bcedice', extra_term='bl', extra_weight='0.095', lsa_aux='probe', lsa_tau=0.15)
    crit = CL.build_criterion(**kw)
    check('build_criterion -> WithExpertAux(WithExtraTerm)', type(crit).__name__ == 'WithExpertAux' and type(crit.inner).__name__ == 'WithExtraTerm')
    want = ('bcedice + 0.095 * bl + expert aux probe (0.5*[BD(D seg)+BD(.5,1)(D ring +-3px)] + '
            '0.3*[BD(C seg)+focal interior depth>=0.15 /|H|])')
    check('describe() of the E3a arm', CL.describe(**kw) == want, CL.describe(**kw))
    torch.manual_seed(0)
    target = torch.zeros(2, 1, 64, 64)
    target[0, :, 10:40, 12:44] = 1
    target[1, :, 30:60, 5:30] = 1
    gts = tuple(torch.rand(2, 1, 64, 64) * 0.98 + 0.01 for _ in range(5))
    out = torch.rand(2, 1, 64, 64) * 0.98 + 0.01
    bnd = {'lsa_d': torch.randn(2, 2, 32, 32), 'lsa_c': torch.randn(2, 1, 16, 16)}
    gt_pre = DeepSupervisionOutputs(gts, boundary=bnd)
    total = float(crit(gt_pre, out, target))
    t = crit.aux_terms(bnd, target)
    hand = (float(GT_BceDiceLoss(1, 1)(gts, out, target)) + 0.095 * float(CL.BoundaryLoss()(out, target))
            + 0.5 * float(t['aux_d_seg'] + t['aux_d_ring']) + 0.3 * float(t['aux_c_seg'] + t['aux_c_ihr']))
    check('total = BceDice (deep supervision) + 0.095 * BL(final) + expert aux', abs(total - hand) < 1e-5, f'{total:.6f} vs {hand:.6f}')
    check('logged terms: bl first, then the aux terms and gates', crit.names[0] == 'bl' and 'aux_c_ihr' in crit.names)

    print('3. config / flags')
    probe = ('import sys, json; sys.argv = ["train.py"] + json.loads(sys.argv[1]); import train; '
             'from configs.config_setting import get_config; c = get_config(train.parse_args()); '
             'print("CFG" + json.dumps({"no_val": getattr(c, "no_val", None), "test_split": c.test_split}))')

    def cfg(flags):
        p = subprocess.run([py, '-c', probe, json.dumps(flags)], cwd=REPO, capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('CFG')]
        return p.returncode, (json.loads(line[0][3:]) if line else None)
    base = ['--work-dir', '/tmp/x', '--device', 'cpu']
    rc, c0 = cfg(base + ['--test-split', 'test'])
    check('without --no-val: no no_val key', rc == 0 and c0['no_val'] is None, str(c0))
    rc, c1 = cfg(base + ['--test-split', 'test', '--no-val'])
    check('--no-val sets config.no_val', rc == 0 and c1['no_val'] is True, str(c1))
    rc, _ = cfg(base + ['--no-val'])
    check('--no-val without --test-split refused', rc == 2)

    print('4. --no-val training (LSA + E3a, 2 epochs, mini train/test)')
    if mini80:
        wd = tempfile.mkdtemp()
        args = [py, '-W', 'ignore', 'train.py', '--work-dir', wd, '--data-path', mini80, '--dataset', 'isic17', '--epochs', '2',
                '--batch-size', '8', '--device', 'cpu', '--seed', '123', '--test-split', 'test', '--no-val', '--save-every', '1',
                '--save-from', '1', '--fusion', 'sum', '--fusion-stages', 'shallow3', '--fusion-dim', '8', '--lsa', 'route',
                '--lsa-aux', 'probe', '--extra-term', 'bl', '--extra-weight', '0.095']
        p = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
        res = json.load(open(os.path.join(wd, 'test_results.json'))) if os.path.isfile(os.path.join(wd, 'test_results.json')) else {}
        check('train.py --no-val finished', p.returncode == 0, p.stderr[-400:])
        check('test_results.json: final-epoch2.pth, no_val, test split, metrics', res.get('checkpoint') == 'final-epoch2.pth'
              and res.get('no_val') is True and res.get('test_split') == 'test' and 'f1_or_dsc' in res and res.get('min_epoch') == 2, str(res)[:200])
        ck = os.listdir(os.path.join(wd, 'checkpoints'))
        check('checkpoints: final, latest, epoch001/002, no best', {'final-epoch2.pth', 'latest.pth', 'epoch001.pth', 'epoch002.pth'} <= set(ck)
              and not any(c.startswith('best') for c in ck), str(sorted(ck)))
        rows = list(csv.DictReader(open(os.path.join(wd, 'metrics.csv'))))
        check('metrics.csv: val_loss nan, bl and gate columns', len(rows) == 2 and rows[0]['val_loss'] == 'nan' and 'train_extra_bl' in rows[0]
              and 'train_extra_gnull_les_dec5' in rows[0], str(list(rows[0])[:6]))
        log = open(os.path.join(wd, 'log', 'train.info.log')).read()
        check('log: rotation angle of seed 123, loss line, no-validation line', 'rotation angle: 18.85089558633996 (seed 123)' in log
              and 'loss: bcedice + 0.095 * bl + expert aux probe' in log and 'no validation split: testing the last-epoch weights' in log)
        final = torch.load(os.path.join(wd, 'checkpoints', 'final-epoch2.pth'), map_location='cpu')
        e2 = torch.load(os.path.join(wd, 'checkpoints', 'epoch002.pth'), map_location='cpu')
        check('final-epoch2.pth = the epoch-2 weights', all(torch.equal(final[k], e2[k]) for k in final))
        m, _ = C.build_model(final, 'cpu', None, strict=True)
        check('analysis.common rebuilds the LSA model from the final checkpoint', getattr(m, 'lsa_mode', 'none') == 'route')
        # resuming a finished run tests the same weights again
        p2 = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
        res2 = json.load(open(os.path.join(wd, 'test_results.json')))
        check('resume of a finished --no-val run: same test result', p2.returncode == 0 and res2['f1_or_dsc'] == res['f1_or_dsc'])
    else:
        print('  [skip] no MINI_DATA_ROOT')

    print('5. 80/20 split lists')
    totals = {'isic17': 2150, 'isic18': 2694}
    for ds, n in totals.items():
        spec = json.load(open(os.path.join(REPO, 'analysis', 'lists', f'split_8020_{ds}.json')))
        tr, te = spec['train'], spec['test']
        dn = [r[3] for r in tr + te]
        src = {(r[0], r[1]) for r in tr + te}
        check(f'{ds}: {len(tr)} train + {len(te)} test = {n}, disjoint, unique destination names', len(tr) + len(te) == n
              and len(set(dn)) == n and len(src) == n and spec['seed'] == 123)
        check(f'{ds}: test = round(n/5) per tertile', all(t['test'] == int(round(t['n'] / 5)) for t in spec['per_tertile']), str(spec['per_tertile']))
        trn = {f'{r[0]}/{r[1]}' for r in tr}
        check(f'{ds}: every duplicate group in train', all(all(x in trn for x in g) for g in spec['duplicate_groups']), str(spec['duplicate_groups']))
        if ds == 'isic17':
            ok = all(r[4] == r[3][:-4] + '_segmentation.png' for r in tr + te)
        else:
            ok = all(r[3] == ('tr_' if r[0] == 'train' else 'va_') + r[1] and r[4] == ('tr_' if r[0] == 'train' else 'va_') + r[2] and r[1] == r[2]
                     for r in tr + te)
        check(f'{ds}: image / mask names pair up', ok)
        for split in ('train', 'test'):
            d = os.path.join(REPO, 'data', 'data_isic1718_8020', {'isic17': 'isic2017', 'isic18': 'isic2018'}[ds], split)
            if os.path.isdir(d):
                check(f'{ds} {split}: built folder matches the list', sorted(os.listdir(os.path.join(d, 'images'))) == sorted(r[3] for r in spec[split]))
    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
