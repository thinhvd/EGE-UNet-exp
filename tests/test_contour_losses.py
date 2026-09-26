'''
Checks for the EXP-6 loss terms (contour_losses.py). CPU, a few seconds, no dataset needed:

    python tests/test_contour_losses.py

1. the defaults build the original GT_BceDiceLoss object;
2. every term is 0 (or at its minimum) when the prediction equals the ground truth;
3. gradients are finite, also for empty and full masks;
4. scale behaviour on discs of radius 10 / 20 / 40 / 80 px with the same RELATIVE dilation: snbl, area and
   tv_match stay (nearly) constant, the pixel boundary loss grows like r^3;
5. the region term (E2) pushes up on lesion pixels and down on background pixels with equal strength
   (decision threshold 0.5), while Dice pushes lesion pixels harder;
6. WithExtraTerm adds exactly weight * term and records the term's running mean.
'''
import os
import sys
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import GT_BceDiceLoss, DiceLoss                      # noqa: E402
import contour_losses as C                                       # noqa: E402

H = W = 256
FAILED = []


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def disc(r, cy=H / 2, cx=W / 2):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                            indexing='ij')
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float().view(1, 1, H, W)


def terms():
    return {'tv': C.TVLength(), 'tv_match': C.TVMatch(), 'area': C.LogArea(0.05), 'bl': C.BoundaryLoss(),
            'snbl': C.ScaleNormBoundaryLoss(3.0), 'region': C.RegionLoss()}


def main():
    torch.manual_seed(0)

    print('1. defaults')
    crit = C.build_criterion()
    check('build_criterion() is the original GT_BceDiceLoss', type(crit) is GT_BceDiceLoss)
    check('bce_region builds GT_BceRegionLoss', type(C.build_criterion('bce_region')) is C.GT_BceRegionLoss)
    try:
        C.build_criterion(extra_term='area')
        check('extra term without weight is refused', False)
    except ValueError:
        check('extra term without weight is refused', True)

    print('2. value at a perfect prediction')
    g = disc(30)
    for name, t in terms().items():
        v = float(t(g.clone(), g))
        if name in ('tv_match', 'area', 'snbl', 'region'):
            check(f'{name} = 0 at u = g', abs(v) < 1e-9, f'{v:.3g}')
        elif name == 'bl':
            u = torch.rand_like(g)
            check('bl is lower at u = g than at a random map', v < float(t(u, g)), f'{v:.4f}')
        else:  # tv: the length of the contour, not 0
            per = float(C.tv_map(g).sum()) - (H - 1) * (W - 1) * math.sqrt(C.TV_EPS)
            # forward differences on a pixel staircase overestimate a circle's length (~18 % here)
            check('tv at u = g ~ disc perimeter', abs(per / (2 * math.pi * 30) - 1) < 0.25,
                  f'TV sum {per:.1f} vs 2*pi*r {2 * math.pi * 30:.1f}')

    print('3. finite gradients (normal, empty and full masks)')
    for label, gt in (('disc', disc(20)), ('empty', torch.zeros(1, 1, H, W)), ('full', torch.ones(1, 1, H, W))):
        for name, t in terms().items():
            u = torch.sigmoid(torch.randn(1, 1, H, W)).requires_grad_(True)
            t(u, gt).backward()
            check(f'{name} on {label} mask', torch.isfinite(u.grad).all().item())

    print('4. scale: same relative dilation (20 % of the radius) at radius 10 / 20 / 40 / 80')
    vals = {k: [] for k in ('snbl', 'area', 'tv_match', 'bl')}
    for r in (10, 20, 40, 80):
        g, u = disc(r), disc(r * 1.2)
        for k in vals:
            vals[k].append(float(terms()[k](u, g)) - (float(terms()[k](g, g)) if k == 'bl' else 0.0))
    for k in ('snbl', 'area', 'tv_match'):
        v = vals[k]
        spread = (max(v) - min(v)) / max(v)
        check(f'{k} nearly scale-free', spread < 0.2, ' / '.join(f'{x:.4f}' for x in v) + f', spread {spread:.0%}')
    v = vals['bl']
    check('pixel boundary loss grows ~ r^3', 250 < v[-1] / v[0] < 800,
          ' / '.join(f'{x:.5f}' for x in v) + f', ratio 80 vs 10 = {v[-1] / v[0]:.0f} (r^3 ratio 512)')

    print('5. decision threshold: region term vs Dice')
    g = disc(12)
    u = torch.full_like(g, 0.5).requires_grad_(True)
    C.RegionLoss()(u, g).backward()
    up, down = -float(u.grad[g > 0].mean()), float(u.grad[g == 0].mean())
    check('region: push on lesion = push on background', abs(up / down - 1) < 1e-6, f'{up:.3g} vs {down:.3g}')
    u = torch.full_like(g, 0.5).requires_grad_(True)
    DiceLoss()(u, g).backward()
    up, down = -float(u.grad[g > 0].mean()), float(u.grad[g == 0].mean())
    check('dice: pushes lesion pixels harder (reference)', up > down, f'ratio {up / down:.2f}')

    print('6. WithExtraTerm')
    base = GT_BceDiceLoss(1, 1)
    crit = C.build_criterion(extra_term='area', extra_weight=0.3)
    g = disc(25).repeat(2, 1, 1, 1)
    out = torch.sigmoid(torch.randn(2, 1, H, W))
    gt_pre = tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5))
    total = float(crit(gt_pre, out, g))
    expect = float(base(gt_pre, out, g)) + 0.3 * float(C.LogArea(0.05)(out, g))
    check('loss = base + weight * term', abs(total - expect) < 1e-6, f'{total:.6f} vs {expect:.6f}')
    check('running mean recorded', crit._n == 1 and abs(crit.epoch_mean() - float(C.LogArea(0.05)(out, g))) < 1e-6)

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
