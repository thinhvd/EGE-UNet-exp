'''
EXP-16 checks (CPU, synthetic data): region definitions, prototypes and relative evidence, AUC helpers, oracle
selection counts.

    python tests/test_exp16.py
'''
import os
import sys
import inspect

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis import exp16_relative_evidence as X        # noqa: E402
from analysis.info_probe import pixel_sets, auc as mw_auc  # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def main():
    rng = np.random.default_rng(0)
    print('1. regions')
    gt = disk(96, 96, 48, 48, 30)
    prob = np.where(disk(96, 96, 48, 48, 18), 0.95, 0.05) + 0.01 * rng.random((96, 96))
    ev = X.eval_regions(gt, prob)
    ref = pixel_sets(gt, prob)
    check('H equals info_probe.pixel_sets H', np.array_equal(ev['H'], ref['H']), f'|H| {ev["H"].sum()}')
    check('B12 equals the EXP-12 band', np.array_equal(ev['B12'], ref['band']))
    d_out = ev['d_out']
    check('B = skin 5-20 px outside the lesion', ev['B'].any() and not (ev['B'] & gt).any()
          and d_out[ev['B']].min() > 5 and d_out[ev['B']].max() <= 20)
    core, bhat, d_pred = X.prediction_regions(prob, 0.8)
    pred = prob >= 0.5
    check('core = P > t_c inside the prediction; B^ = P < 0.1, 10-30 px from the predicted mask',
          np.array_equal(core, prob > 0.8) and not (core & ~pred).any() and bhat.any() and (prob[bhat] < 0.1).all()
          and d_pred[bhat].min() > 10 and d_pred[bhat].max() <= 30)
    sig = inspect.signature(X.prediction_regions).parameters
    sig2 = inspect.signature(X.relative_evidence).parameters
    check('prototype functions take no ground truth', list(sig) == ['prob', 'core_t'] and list(sig2) == ['feat', 'core', 'bg', 'proto'])

    print('2. prototypes and relative evidence')
    feat = torch.randn(6, 40, 40)
    m = np.zeros((40, 40), bool)
    m[5:15, 8:20] = True
    mu = X.global_proto(feat, m)
    check('global prototype = mean over the mask', torch.allclose(mu, feat[:, torch.from_numpy(m)].mean(1)))
    a = X.cos_to(feat, mu)
    hand = torch.nn.functional.cosine_similarity(feat, mu[:, None, None].expand_as(feat), dim=0).numpy()
    check('cosine map', np.allclose(a, hand, atol=1e-5))
    ws = X.window_sum(feat, 3)
    y, x = 2, 37
    brute = feat[:, max(0, y - 3):y + 4, max(0, x - 3):x + 4].sum((1, 2))
    check('integral-image window sum = brute force (border pixel)', torch.allclose(ws[:, y, x], brute, atol=1e-4))
    lc = X.local_cos(feat, m, 4)
    yy, xx = 20, 14                     # window rows 16-24 meets the mask rows 5-14? no -> must be NaN
    check('local cosine undefined where the window holds no core pixel', np.isnan(lc[yy, xx]))
    yy, xx = 16, 14                     # window rows 12-20 holds mask rows 12-14
    sub = m[12:21, 10:19]
    mu_loc = feat[:, 12:21, 10:19][:, torch.from_numpy(sub)].mean(1)
    check('local cosine = cosine with the windowed core mean', abs(lc[yy, xx] - float(torch.nn.functional.cosine_similarity(feat[:, yy, xx], mu_loc, dim=0))) < 1e-4)
    a_l, a_b, R = X.relative_evidence(feat, m, ~m, 'global')
    check('R = A_L - A_B', np.allclose(R, a_l - a_b))

    print('3. statistics')
    pos, neg = rng.normal(1, 1, 500), rng.normal(0, 1, 700)
    from sklearn.metrics import roc_auc_score
    ref_auc = roc_auc_score(np.r_[np.ones(500), np.zeros(700)], np.r_[pos, neg])
    check('Mann-Whitney AUC = sklearn', abs(mw_auc(pos, neg) - ref_auc) < 1e-9)
    hp, hn = np.zeros(X.NB, np.int64), np.zeros(X.NB, np.int64)
    X.hist_add(hp, np.clip(pos / 4, -1.99, 1.99), 'R')
    X.hist_add(hn, np.clip(neg / 4, -1.99, 1.99), 'R')
    check('histogram AUC within 0.002 of the exact AUC', abs(X.auc_hist(hp, hn) - ref_auc) < 2e-3, f'{X.auc_hist(hp, hn):.4f} vs {ref_auc:.4f}')
    check('fraction above 0 from the histogram', abs(X.frac_above(hp, 'R', 0.0) - (np.clip(pos / 4, -1.99, 1.99) >= 0).mean()) < 2e-3)
    pr = X.pct_ranks(np.array([3.0, 1.0, 2.0]))
    check('percentile ranks', np.allclose(pr, [1.0, 0.0, 0.5]))

    print('4. oracle selection counts')
    score = np.array([0.9, np.nan, 0.1, 0.5, -0.2, 0.7, 0.3, 0.0, 0.8, 0.6])
    attrs = np.array([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                      [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
    out = X.selection_counts(score, attrs, (10, 30, 100), (0.55, 0.0))
    # top 10 % = 1 pixel (0.9); top 30 % = 3 pixels (0.9, 0.8, 0.7); top 100 % = 9 defined pixels (NaN never selected)
    check('top-k: count and attribute sums', out[0].tolist() == [1, 1, 1, 0, 0] and out[1].tolist() == [3, 2, 2, 1, 1]
          and out[2, 0] == 9 and out[2, 2] == 2, str(out[:3].tolist()))
    # tau 0.55: 0.9, 0.8, 0.7, 0.6 -> 4 pixels; tau 0.0: strictly greater -> 7 pixels (0.0 and -0.2 excluded)
    check('threshold rule: score > tau', out[3, 0] == 4 and out[3, 1] == 3 and out[4, 0] == 7, str(out[3:].tolist()))
    base = np.array([[100, 10, 40, 30, 20, 50, 2, 300000]])
    oracle = np.zeros((1, 1, 1, 2, 6), np.int32)
    oracle[0, 0, 0, 0] = [20, 15, 12, 3, 8, 1]
    t = X.oracle_table(oracle, base, 0, 0, 0)
    check('completion metrics: DSC after adding 15 TP and 5 FP', abs(t['dsc'][0] - 2 * 115 / (2 * 115 + 15 + 25)) < 1e-12)
    o = X.oracle_summary(oracle, base, 0, 0, 0)
    check('summary: H fixed, new FP, expansion precision and recall', o['H_fixed'] == 12 and o['new_fp'] == 5 and o['new_fp_far3'] == 3
          and abs(o['expansion_precision'] - 75.0) < 1e-9 and abs(o['expansion_recall_H'] - 40.0) < 1e-9)
    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
