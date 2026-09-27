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

    print('7. edt_torch (GPU path of the distance map) equals scipy')
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    masks = [disc(1, 5, 250).view(H, W) > 0, disc(40).view(H, W) > 0, torch.rand(H, W) > 0.97, torch.rand(H, W) > 0.03]
    mask_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'data_isic1718', 'isic2017', 'train', 'masks')
    if os.path.isdir(mask_dir):       # real lesion masks, also rotated, when the dataset is present
        from PIL import Image
        for name in sorted(os.listdir(mask_dir))[:12]:
            m = np.array(Image.open(os.path.join(mask_dir, name)).convert('L').resize((W, H))) >= 128
            masks += [torch.from_numpy(m), torch.from_numpy(np.ascontiguousarray(np.rot90(m)))]
    worst = 0.0
    for m in masks:
        for feat in (m, ~m):
            if feat.any():
                ref = distance_transform_edt(~feat.numpy())
                worst = max(worst, float(np.abs(C.edt_torch(feat).double().numpy() - ref).max()))
    check(f'max |edt_torch - scipy| over {len(masks)} masks x 2 sides < 1e-4', worst < 1e-4, f'{worst:.2e}')
    g = torch.stack([m.float() for m in masks[:6]]).unsqueeze(1)
    cpu_path = C.signed_distance(g)                                    # scipy
    same = [C.edt_torch(m) - (C.edt_torch(~m) - 1) * m if (m.any() and not m.all()) else torch.zeros(H, W)
            for m in (g[:, 0] >= 0.5)]
    diff = float((cpu_path[:, 0] - torch.stack(same)).abs().max())
    check('signed distance: torch formula = scipy formula', diff < 1e-4, f'{diff:.2e}')

    print('8. EXP-7: fn_dp (misses weighted by distance to the prediction) and several terms at once')
    g = disc(40)
    fn = C.FNDistance()
    check('fn_dp = 0 when the prediction covers the lesion', float(fn(g.clone(), g)) == 0.0)
    check('fn_dp = 0 for a larger prediction (spill is not its business)', float(fn(disc(55), g)) == 0.0)
    u = disc(30).requires_grad_(True)            # a shrunk prediction: a 10 px missed ring
    fn(u, g).backward()
    missed = (g > 0) & (u.detach() < 0.5)
    check('fn_dp gradient only on missed lesion pixels', bool((u.grad[~missed] == 0).all()) and bool((u.grad[missed] < 0).all()))
    empty = torch.zeros(1, 1, H, W).requires_grad_(True)
    v = fn(empty, g)
    v.backward()
    inradius = float(distance_transform_edt(g[0, 0].numpy() > 0).max())
    check('empty prediction: finite, each lesion pixel weighted by the inradius',
          bool(torch.isfinite(v)) and abs(float(v) - inradius * float(g.mean())) < 1e-4, f'{float(v):.4f}')
    # distance map vs scipy, with the inradius cap
    p = disc(12, 100, 100) + disc(10, 150, 170)
    d = C.distance_to_prediction(p, g)[0, 0].numpy()
    ref = np.minimum(distance_transform_edt(~(p[0, 0].numpy() >= 0.5)), inradius)
    check('distance_to_prediction = min(scipy edt, inradius)', float(np.abs(d - ref).max()) < 1e-4)
    both = [(p, g), (torch.zeros_like(g), g), (g.clone(), g), (disc(30), torch.zeros_like(g)), (disc(5), torch.ones_like(g))]
    worst = max(float((C.distance_to_prediction(a_, b_, backend='torch') - C.distance_to_prediction(a_, b_)).abs().max())
                for a_, b_ in both)
    check('GPU (edt_torch) path = scipy path, incl. empty / full cases', worst < 1e-4, f'{worst:.1e}')
    # a dropped chunk costs more per pixel than a thin missed rim of the same lesion
    chunk = g.clone(); chunk[..., 128:, :] = 0                 # lower half missed
    rim = disc(38)                                              # 2 px rim missed
    per_px = lambda pr: float(fn(pr, g)) / max(float(((g > 0) & (pr < 0.5)).float().mean()), 1e-9)
    check('missed chunk costs more per pixel than a missed rim', per_px(chunk) > 3 * per_px(rim),
          f'chunk {per_px(chunk):.1f} vs rim {per_px(rim):.1f} per missed px')
    # multi-term wrapper
    crit = C.build_criterion(extra_term='bl,fn_dp', extra_weight='0.095,0.35')
    out = torch.sigmoid(torch.randn(2, 1, H, W)); gg = disc(25).repeat(2, 1, 1, 1)
    gt_pre = tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5))
    total = float(crit(gt_pre, out, gg))
    expect = float(GT_BceDiceLoss(1, 1)(gt_pre, out, gg)) + 0.095 * float(C.BoundaryLoss()(out, gg)) \
        + 0.35 * float(C.FNDistance()(out, gg))
    check('bl,fn_dp: loss = base + 0.095 * bl + 0.35 * fn_dp', abs(total - expect) < 1e-5, f'{total:.6f} vs {expect:.6f}')
    check('per-term running means recorded', set(crit.epoch_means()) == {'bl', 'fn_dp'})
    single = C.build_criterion(extra_term='bl', extra_weight='0.095')
    check('single term keeps the EXP-6 arithmetic', abs(float(single(gt_pre, out, gg)) -
          (float(GT_BceDiceLoss(1, 1)(gt_pre, out, gg)) + 0.095 * float(C.BoundaryLoss()(out, gg)))) < 1e-6)
    for bad in (('bl,fn_dp', '0.1'), ('bl,bl', '0.1,0.1'), ('foo', '0.1')):
        try:
            C.parse_terms(*bad)
            check(f'malformed {bad} refused', False)
        except ValueError:
            check(f'malformed {bad} refused', True)

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
