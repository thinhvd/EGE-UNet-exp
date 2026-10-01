'''
Checks for the EXP-11 boundary-constrained region completion (models/completion.py), its contour-ring loss
and the hard-positive term (contour_losses.boundary_ring / HardPositive). CPU, about a minute, no dataset
needed (section 9 also loads the EXP-10 checkpoints when results/EGE-UNet-results-exp10 is present):

    python tests/test_completion.py

1. parameter counts (D1 54,965 / D2 = D3 55,656), no completion keys in older configurations, refusals;
2. D2 / D3 / D4 leave the global RNG where D1 leaves it, share D1's trunk tensors bit for bit and start with
   the same completion tensors; at init the output is D1's z_base plus an added logit in [0, alpha_0];
   enabled = False gives D1 exactly;
3. the growth: a single seed with step 1 fills a (2T+1)^2 square, a uniform step s gives s^k at Chebyshev
   distance k, a 4-connected wall stops it, a one-cell diagonal wall does not; the added logit is >= 0 and
   0 on the seed; the knockout switches do what they say;
4. gradient flow from the real init: rho, the last evidence conv and the barrier get gradient at step 1 and
   move after one AdamW step, every completion tensor after a few; the ring loss does not reach the trunk
   (barrier input detached); in 'nobarrier' the barrier learns from the ring loss only;
5. provenance buffers: a checkpoint loaded into a model built with another mode / step count / seed
   threshold is refused; analysis.common recovers the configuration and loads strictly;
6. the contour ring: r = 1 equals the EXP-9 band, the half plane gives 3 + 3 rows, nothing along the frame;
7. the hard-positive term: its mask equals the far FN band of the readouts (fn_code == 2), torch == scipy,
   the value matches a hand computation, the gradient is finite at u = 1e-7 and zero outside the deep set;
8. criteria and describe() strings (ASCII, older ones unchanged), log columns;
9. train.py / config: older flag sets produce no new config keys, invalid combinations are refused, and
   the EXP-10 checkpoints still load strictly.
'''
import os
import sys
import json
import subprocess

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import contour_losses as CL                                           # noqa: E402
from models.egeunet import EGEUNet                                    # noqa: E402
from models.completion import RegionCompletion, grow                  # noqa: E402
from models.fusion import FUSION_STAGE_SETS, DeepSupervisionOutputs   # noqa: E402
from utils import GT_BceDiceLoss, BceDiceLoss, set_seed               # noqa: E402

FAILED = []
H = W = 256
CSF = dict(fusion_mode='sum', fusion_stages=FUSION_STAGE_SETS['shallow3'], fusion_dim=8)
ALPHA0 = float(torch.nn.functional.softplus(torch.tensor(-4.0)))


def check(name, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        FAILED.append(name)


def build(**kw):
    set_seed(42)
    return EGEUNet(**kw)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def disc(r, cy=H / 2, cx=W / 2):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                            indexing='ij')
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float().view(1, 1, H, W)


class Tap:
    '''Records z_base (model.final) and the completion outputs of the last forward.'''
    def __init__(self, model):
        self.rec = {}
        model.final.register_forward_hook(lambda m, i, o: self.rec.__setitem__('z_base', o.detach()))
        if hasattr(model, 'complete'):
            model.complete.register_forward_hook(lambda m, i, o: self.rec.__setitem__('out', [t.detach() for t in o]))


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, H, W)
    target = torch.cat([disc(70), disc(15, 60, 190)])

    print('1. parameter counts and refusals')
    d0, d1 = build(), build(**CSF)
    d2, d3 = build(complete_mode='full', **CSF), build(complete_mode='nobarrier', **CSF)
    check('D0 53374 / D1 54965 / D2 55656 / D3 55656', (nparams(d0), nparams(d1), nparams(d2), nparams(d3)) ==
          (53374, 54965, 55656, 55656), f'{nparams(d0)} {nparams(d1)} {nparams(d2)} {nparams(d3)}')
    parts = {n: p.numel() for n, p in d2.complete.named_parameters()}
    check('completion = evidence 1x1 448+8, GN 16, 3x3 72+1; barrier 144+1; rho 1 (691)',
          sum(parts.values()) == 691 and parts['evidence.0.weight'] == 448 and parts['barrier.weight'] == 144)
    for label, m in (('baseline', d0), ('sum-shallow3-d8', d1), ('refine gate', build(refine_mode='gate', **CSF)),
                     ('bg_stage', build(fusion_mode='bg_stage', fusion_stages=FUSION_STAGE_SETS['shallow3'], fusion_dim=8))):
        check(f'{label}: no completion keys', not any(k.startswith('complete.') for k in m.state_dict()))
    for bad, kw in (('complete without fusion', dict(complete_mode='full')),
                    ('complete with deep3', dict(complete_mode='full', fusion_mode='sum', fusion_stages=FUSION_STAGE_SETS['deep3'])),
                    ('complete with bg_stage', dict(complete_mode='full', fusion_mode='bg_stage', fusion_stages=FUSION_STAGE_SETS['shallow3'])),
                    ('complete with refine', dict(complete_mode='full', refine_mode='plain', **CSF)),
                    ('complete without gt_ds', dict(complete_mode='full', gt_ds=False, **CSF)),
                    ('unknown mode', dict(complete_mode='grow', **CSF)),
                    ('steps 0', dict(complete_mode='full', complete_steps=0, **CSF)),
                    ('seed 0.3', dict(complete_mode='full', complete_seed=0.3, **CSF))):
        try:
            build(**kw)
            check(f'{bad} is refused', False)
        except ValueError:
            check(f'{bad} is refused', True)

    print('2. same trunk, same RNG, output within alpha_0 of D1 at init')
    draws = {}
    for label, kw in (('D1', CSF), ('D2', dict(complete_mode='full', **CSF)), ('D3', dict(complete_mode='nobarrier', **CSF)),
                      ('D2 T=8', dict(complete_mode='full', complete_steps=8, **CSF))):
        build(**kw)
        draws[label] = torch.rand(4)
    check('D2, D3 (any T) leave the global RNG where D1 leaves it (fork_rng)',
          all(torch.equal(draws['D1'], v) for v in draws.values()))
    s1, s2, s3 = d1.state_dict(), d2.state_dict(), d3.state_dict()
    check('trunk tensors of D2 and D3 identical to D1', all(torch.equal(s1[k], s2[k]) and torch.equal(s1[k], s3[k]) for k in s1))
    cp = [k for k in s2 if k.startswith('complete.') and k not in ('complete.mode_id',)]
    check('D2 and D3 completion tensors identical at init (buffers aside from the mode)', all(torch.equal(s2[k], s3[k]) for k in cp))
    c = d2.complete
    check('init: A = sigmoid(3) and B = sigmoid(-3) everywhere, alpha = softplus(-4)',
          float(c.evidence[-1].weight.abs().sum()) == 0 and float(c.evidence[-1].bias) == 3.0
          and float(c.barrier.weight.abs().sum()) == 0 and float(c.barrier.bias) == -3.0 and abs(float(c.alpha) - ALPHA0) < 1e-9)
    t1, t2 = Tap(d1), Tap(d2)
    for train_mode in (True, False):
        d1.train(train_mode)
        d2.train(train_mode)
        with torch.no_grad():
            g1, o1 = d1(x)
            g2, o2 = d2(x)
        add = t2.rec['out'][0]
        same_base = torch.equal(t1.rec['z_base'], t2.rec['z_base'])
        same_ds = all(torch.equal(a, b) for a, b in zip(g1, g2))
        check(f'D2 z_base and deep supervision == D1; added logit in [0, alpha_0] ({"train" if train_mode else "eval"})',
              same_base and same_ds and float(add.min()) >= 0 and float(add.max()) <= ALPHA0 + 1e-7,
              f'add in [{float(add.min()):.4f}, {float(add.max()):.4f}], {100 * float((add > 0).float().mean()):.1f} % cells > 0')
    for m in (d1, d2, d3):
        m.eval()
    with torch.no_grad():
        _, o1 = d1(x)
        d2.complete.enabled = False
        _, o2off = d2(x)
        d2.complete.enabled = True
        g2, o2 = d2(x)
        g1, _ = d1(x)
    check('enabled = False: D2 == D1 (torch.equal)', torch.equal(o2off, o1))
    check('D1 returns a plain tuple, D2 a DeepSupervisionOutputs with the ring logit at 128',
          type(g1) is tuple and isinstance(g2, DeepSupervisionOutputs) and list(g2.boundary) == ['ring']
          and tuple(g2.boundary['ring'].shape) == (2, 1, 128, 128))

    print('3. growth')
    seed = torch.zeros(1, 1, 64, 64)
    seed[..., 32, 32] = 1
    r = grow(seed, torch.ones_like(seed), 5)
    check('one seed, step 1, T = 5 -> an 11 x 11 square of ones', int(r.sum()) == 121 and float(r[..., 27:38, 27:38].min()) == 1.0)
    r = grow(seed, torch.full_like(seed, 0.5), 6)
    k = torch.tensor([float(r[0, 0, 32, 32 + d]) for d in range(8)])
    check('uniform step 0.5 -> 0.5^k at Chebyshev distance k (k <= T), 0 beyond',
          torch.allclose(k[:7], 0.5 ** torch.arange(7.0)) and float(k[7]) == 0.0, f'{k.tolist()}')
    wall = torch.ones_like(seed)
    wall[..., :, 40] = 0
    r = grow(seed, wall, 40)
    check('a 4-connected wall (full column of step 0) stops the growth', float(r[..., :, 41:].max()) == 0.0 and float(r[..., :, 39].min()) == 1.0)
    diag = torch.ones_like(seed)
    idx = torch.arange(64)
    diag[0, 0, idx, (idx + 8).clamp(max=63)] = 0          # one-cell diagonal line: 8-connected only
    r = grow(seed, diag, 64)
    check('a one-cell diagonal wall lets the growth through (8-connected moves)', float(r[0, 0, 0, 63]) == 1.0)
    stair = diag.clone()
    stair[0, 0, idx[:-1], (idx[:-1] + 9).clamp(max=63)] = 0   # thicken to a 4-connected staircase
    r = grow(seed, stair, 64)
    check('the same line thickened to a 4-connected staircase stops it', float(r[0, 0, 0, 63]) == 0.0)
    m = RegionCompletion(8, 8, 'full', steps=4)
    m.set_init()
    zb = torch.full((1, 1, 32, 32), -5.0)
    zb[..., 16, 16] = 5.0
    d5 = torch.randn(1, 8, 32, 32)
    projs = [torch.randn(1, 8, 32 // 2 ** i, 32 // 2 ** i) for i in range(5)]
    with torch.no_grad():
        add, rl, rt, a, b = m(zb, d5, projs)
        s = float(torch.sigmoid(torch.tensor(3.0)) * torch.sigmoid(torch.tensor(3.0)))
        ok_vals = abs(float(rt[0, 0, 16, 20]) - s ** 4) < 1e-6 and float(rt[0, 0, 16, 21]) == 0.0
        check('module: seed = p >= 0.9, R_T = (A (1 - B))^k at init, add = alpha (R_T - seed) >= 0 and 0 on the seed',
              ok_vals and float(add.min()) >= 0 and float(add[0, 0, 16, 16]) == 0.0
              and torch.allclose(add, m.alpha * (rt - (torch.sigmoid(zb) >= 0.9).float())), f'R at 4 cells {float(rt[0, 0, 16, 20]):.4f}')
        m.barrier_off = True
        _, _, rt_nb, _, _ = m(zb, d5, projs)
        m.barrier_off = False
        m.evidence_one = True
        _, _, rt_a1, _, _ = m(zb, d5, projs)
        m.evidence_one = False
        m.steps_override = 0
        add0, _, _, _, _ = m(zb, d5, projs)
        m.steps_override = None
        mn = RegionCompletion(8, 8, 'nobarrier', steps=4)
        mn.load_state_dict({k: v for k, v in m.state_dict().items() if k != 'mode_id'} | {'mode_id': mn.mode_id}, strict=True)
        _, _, rt_n, _, _ = mn(zb, d5, projs)
    s_a = float(torch.sigmoid(torch.tensor(3.0)))
    check('barrier_off: R_T = A^k; nobarrier mode equals full with barrier_off',
          abs(float(rt_nb[0, 0, 16, 20]) - s_a ** 4) < 1e-6 and torch.equal(rt_nb, rt_n))
    check('evidence_one: R_T = (1 - B)^k', abs(float(rt_a1[0, 0, 16, 20]) - s_a ** 4) < 1e-6)
    check('steps_override = 0: nothing added', float(add0.abs().max()) == 0.0)

    print('4. gradient flow')
    ring_crit = lambda w: CL.build_criterion(boundary_weight=w, boundary_stages={'ring': 1.0}, ring_radius=3)
    grads = {}
    for label, mode, w in (('D2', 'full', 0.3), ('D2 w0', 'full', 0.0), ('D3', 'nobarrier', 0.3), ('D3 w0', 'nobarrier', 0.0)):
        mdl = build(complete_mode=mode, **CSF)
        mdl.train()
        gp, o = mdl(x)
        ring_crit(w)(gp, o, target).backward()
        grads[label] = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in mdl.named_parameters()}
    nz = lambda g, n: float(g[n].abs().sum()) > 0
    g2 = grads['D2']
    check('D2 step 1: rho, evidence.3 (weight, bias), barrier (weight, bias) get gradient',
          all(nz(g2, n) for n in ('complete.rho', 'complete.evidence.3.weight', 'complete.evidence.3.bias',
                                  'complete.barrier.weight', 'complete.barrier.bias')))
    check('D2 step 1: evidence.0 gets none yet (the last evidence conv starts at 0)',
          not any(nz(g2, n) for n in g2 if n.startswith('complete.evidence.0.')))
    check('D2 with ring weight 0: the barrier still learns through the growth', nz(grads['D2 w0'], 'complete.barrier.weight'))
    check('D3 with ring weight 0: the barrier gets no gradient (not used in the growth)',
          not nz(grads['D3 w0'], 'complete.barrier.weight') and not nz(grads['D3 w0'], 'complete.barrier.bias'))
    check('D3 with ring weight 0.3: the barrier learns from the ring loss', nz(grads['D3'], 'complete.barrier.weight'))
    trunk = [n for n in g2 if not n.startswith('complete.')]
    check('the ring loss does not reach the trunk: D3 trunk gradients with weight 0.3 == weight 0 (exact)',
          all(torch.equal(grads['D3'][n], grads['D3 w0'][n]) for n in trunk))
    mdl = build(complete_mode='full', **CSF)
    mdl.train()
    crit = ring_crit(0.3)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-2)
    before = {n: p.detach().clone() for n, p in mdl.complete.named_parameters()}
    opt.zero_grad()
    gp, o = mdl(x)
    crit(gp, o, target).backward()
    opt.step()
    moved = {n: not torch.equal(before[n], p.detach()) for n, p in mdl.complete.named_parameters()}
    check('one AdamW step moves rho, evidence.3 and the barrier',
          all(moved[n] for n in ('rho', 'evidence.3.weight', 'evidence.3.bias', 'barrier.weight', 'barrier.bias')))
    for _ in range(3):
        opt.zero_grad()
        gp, o = mdl(x)
        crit(gp, o, target).backward()
        opt.step()
    check('every completion tensor moved after 3 more steps',
          all(not torch.equal(before[n], p.detach()) for n, p in mdl.complete.named_parameters()))

    print('5. provenance buffers and analysis.common')
    sd2 = d2.state_dict()
    check('buffers in the state_dict: mode_id 0 / n_steps 32 / seed 0.9',
          int(sd2['complete.mode_id']) == 0 and int(sd2['complete.n_steps']) == 32 and abs(float(sd2['complete.seed_threshold']) - 0.9) < 1e-6)
    for label, kw in (('D2 checkpoint into a nobarrier model', dict(complete_mode='nobarrier')),
                      ('T=32 checkpoint into a T=24 model', dict(complete_mode='full', complete_steps=24)),
                      ('seed 0.9 checkpoint into a seed 0.8 model', dict(complete_mode='full', complete_seed=0.8))):
        try:
            build(**kw, **CSF).load_state_dict(sd2, strict=True)
            check(f'{label} is refused', False)
        except RuntimeError:
            check(f'{label} is refused', True)
    try:
        from analysis import common as AC
        for label, mode, steps in (('D2', 'full', 32), ('D3', 'nobarrier', 32), ('D2 T=24', 'full', 24)):
            model = build(complete_mode=mode, complete_steps=steps, **CSF)
            g = torch.Generator().manual_seed(3)
            with torch.no_grad():
                for p in model.complete.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.2)
            sd = model.state_dict()
            cfg = AC.infer_model_config(sd)
            rebuilt, _ = AC.build_model(sd, strict=True)
            with torch.no_grad():
                same = torch.equal(rebuilt(x)[1], model.eval()(x)[1])
            check(f'{label}: complete {mode!r} T={steps} seed 0.9 recovered, strict load, same output',
                  cfg['complete_mode'] == mode and cfg['complete_steps'] == steps and cfg['complete_seed'] == 0.9 and same)
        check('older configurations infer no complete_* keys', 'complete_mode' not in AC.infer_model_config(d1.state_dict()))
    except ImportError as e:
        print(f'  [skip] analysis.common needs {e.name}')

    print('6. contour ring')
    worst = max(int((CL.boundary_ring(g, 1) != CL.boundary_band(g)).sum()) for g in (disc(40), disc(3, 20, 200), target))
    check('ring r = 1 == the EXP-9 band', worst == 0, f'{worst} px differ')
    half = torch.zeros(1, 1, H, W)
    half[..., :128, :] = 1
    rows = torch.nonzero(CL.boundary_ring(half, 3)[0, 0].sum(1)).flatten()
    check('half plane: ring = rows 125..130 (3 inside + 3 outside), nothing along the frame',
          int(rows.min()) == 125 and int(rows.max()) == 130 and float(CL.boundary_ring(half, 3)[0, 0, :, 0].sum()) == 6,
          f'rows {int(rows.min())}..{int(rows.max())}')
    rows128 = torch.nonzero(CL.band_at_grid(CL.boundary_ring(half, 3), (128, 128))[0, 0].sum(1)).flatten()
    check('at the 128 grid: rows 62..65', int(rows128.min()) == 62 and int(rows128.max()) == 65)
    z, o = torch.zeros(1, 1, H, W), torch.ones(1, 1, H, W)
    check('empty / full mask -> empty ring', float(CL.boundary_ring(z, 3).sum()) == 0 and float(CL.boundary_ring(o, 3).sum()) == 0)

    print('7. hard-positive term')
    from scipy.ndimage import distance_transform_edt
    worst = 0
    for g in (disc(40), disc(8, 30, 30), disc(90, 128, 250), target[:1], target[1:]):
        m_ = g[0, 0].numpy() >= 0.5
        d_in = distance_transform_edt(m_)
        code = np.where(d_in <= 3, 0, np.where(d_in <= 10, 1, 2))
        ref = (m_ & (code == 2)).astype(np.float32)
        worst = max(worst, int((CL.deep_positive_mask(g, 10)[0, 0].numpy() != ref).sum()),
                    int((CL.deep_positive_mask(g, 10, backend='torch')[0, 0].numpy() != ref).sum()))
    check('deep mask == lesion & fn_code 2 of the readouts (scipy and torch paths)', worst == 0, f'{worst} px differ')
    check('small lesion (r = 8) has no deep pixels; empty mask -> zeros',
          float(CL.deep_positive_mask(disc(8), 10).sum()) == 0 and float(CL.deep_positive_mask(z, 10).sum()) == 0)
    hp = CL.HardPositive(10, 2)
    u = torch.full((2, 1, H, W), 0.2)
    n_deep = float(CL.deep_positive_mask(target, 10).sum())
    hand = n_deep * 0.8 ** 2 * -np.log(0.2) / u.numel()
    check('value: sum_H (1-u)^2 (-log u) / (B*H*W)', abs(float(hp(u, target)) - hand) < 1e-6, f'{float(hp(u, target)):.6f} vs {hand:.6f}')
    check('u = 1 on the lesion -> 0', float(hp(target.clone(), target)) == 0.0)
    uu = torch.full((2, 1, H, W), 1e-7, requires_grad=True)
    hp(uu, target).backward()
    gmask = CL.deep_positive_mask(target, 10) > 0
    check('gradient finite and negative on deep pixels at u = 1e-7, zero elsewhere',
          bool(torch.isfinite(uu.grad).all()) and float(uu.grad[gmask].max()) < 0 and float(uu.grad[~gmask].abs().max()) == 0)
    try:
        CL.HardPositive(10, 0.5)
        check('gamma 0.5 refused', False)
    except ValueError:
        check('gamma 0.5 refused', True)

    print('8. criteria, describe(), log columns')
    check('defaults still build the bare GT_BceDiceLoss', type(CL.build_criterion()) is GT_BceDiceLoss)
    check('EXP-10 describe() unchanged', CL.describe(boundary_weight=0.3, boundary_stages={'dec5': 1.0}, boundary_radius=10)
          == 'bcedice + 0.3 * zone (BceDice(0.5,1) on the r=10px contour zone, 1*dec5)')
    check('EXP-9 describe() unchanged', CL.describe('bcedice', 'bl', '0.095', boundary_weight=1.0)
          == 'bcedice + 0.095 * bl + 1 * bnd (0.1*dec3 + 0.2*dec4 + 0.3*dec5, BceDice(0.5,1) on the contour band per head grid)')
    d_ring = CL.describe(boundary_weight=0.3, boundary_stages={'ring': 1.0}, ring_radius=3)
    check('D2 describe() is ASCII', d_ring == 'bcedice + 0.3 * ring (BceDice(0.5,1) on the +-3px contour ring, barrier head)'
          and d_ring.isascii(), d_ring)
    d_hp = CL.describe('bcedice', 'hp', '0.42', boundary_weight=0.3, boundary_stages={'ring': 1.0}, ring_radius=3,
                       hp_radius=10.0, hp_gamma=2.0)
    check('D4 describe() is ASCII', d_hp == 'bcedice + 0.42 * hp (hp r=10px, gamma 2) + 0.3 * ring (BceDice(0.5,1) on the '
          '+-3px contour ring, barrier head)' and d_hp.isascii(), d_hp)
    c2 = ring_crit(0.3)
    c4 = CL.build_criterion(extra_term='hp', extra_weight='0.42', boundary_weight=0.3, boundary_stages={'ring': 1.0},
                            ring_radius=3, hp_radius=10.0, hp_gamma=2.0)
    check('D2 criterion: one term bnd_ring on the r=3 ring; D4: hp + bnd_ring',
          c2.names == ['bnd_ring'] and c2.target_fn.keywords == {'radius': 3} and c4.names == ['hp', 'bnd_ring'])
    fields = lambda crit: (['train_extra'] if len(crit.names) == 1 and not crit.names[0].startswith('bnd_')
                           else [f'train_extra_{n}' for n in crit.names])   # train.py's rule
    check('log columns: train_extra_bnd_ring / train_extra_hp + train_extra_bnd_ring',
          fields(c2) == ['train_extra_bnd_ring'] and fields(c4) == ['train_extra_hp', 'train_extra_bnd_ring'])
    logit = torch.randn(2, 1, 128, 128)
    gt_pre = DeepSupervisionOutputs(tuple(torch.sigmoid(torch.randn(2, 1, H, W)) for _ in range(5)), boundary={'ring': logit})
    out = torch.sigmoid(torch.randn(2, 1, H, W))
    ring128 = CL.band_at_grid(CL.boundary_ring(target, 3), (128, 128))
    hand = BceDiceLoss(0.5, 1)(torch.sigmoid(logit), ring128)
    expect = GT_BceDiceLoss(1, 1)(gt_pre, out, target) + 0.3 * hand
    check('D2 loss = inner + 0.3 * BceDice(sigmoid(ring logit), ring@128)', abs(float(c2(gt_pre, out, target) - expect)) < 1e-6)
    expect4 = expect + 0.42 * CL.HardPositive(10, 2)(out, target)
    check('D4 loss = D2 loss + 0.42 * hp', abs(float(c4(gt_pre, out, target) - expect4)) < 1e-5)
    try:
        CL.build_criterion(boundary_weight=0.3, boundary_radius=10, ring_radius=3)
        check('zone and ring together refused', False)
    except ValueError:
        check('zone and ring together refused', True)

    print('9. train.py / config and older checkpoints')
    py = sys.executable
    probe = ('import sys, json; sys.argv = ["train.py"] + json.loads(sys.argv[1]); import train; '
             'from configs.config_setting import get_config; c = get_config(train.parse_args()); '
             'print("CFG" + json.dumps({"model": {k: v for k, v in c.model_config.items() if k.startswith(("complete", "refine"))}, '
             '"loss": sorted(getattr(c, "loss_config", {}).keys())}))')
    def cfg(flags):
        p = subprocess.run([py, '-c', probe, json.dumps(flags)], cwd=REPO, capture_output=True, text=True)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith('CFG')]
        return p.returncode, (json.loads(line[0][3:]) if line else None), p.stderr
    base = ['--work-dir', '/tmp/x', '--device', 'cpu']
    csf = ['--fusion', 'sum', '--fusion-stages', 'shallow3', '--fusion-dim', '8']
    rc, c_old, _ = cfg(base + csf)
    check('D1 flags: no complete_* / refine keys, no loss_config', rc == 0 and c_old == {'model': {}, 'loss': []}, str(c_old))
    rc, c_e10, _ = cfg(base + csf + ['--refine', 'gate', '--boundary-weight', '0.3'])
    check('EXP-10 B2 flags: loss_config keys unchanged', rc == 0 and c_e10['loss'] == sorted(
        ['loss', 'extra_term', 'extra_weight', 'snbl_tau', 'area_delta', 'boundary_weight', 'boundary_stages', 'boundary_radius'])
          and c_e10['model'] == {'refine_mode': 'gate'}, str(c_e10))
    rc, c_d4, _ = cfg(base + csf + ['--complete', 'full', '--boundary-weight', '0.3', '--extra-term', 'hp', '--extra-weight', '0.42'])
    check('D4 flags: complete_mode/steps/seed, ring_radius, hp_radius/gamma', rc == 0
          and c_d4['model'] == {'complete_mode': 'full', 'complete_steps': 32, 'complete_seed': 0.9}
          and {'ring_radius', 'hp_radius', 'hp_gamma', 'boundary_stages'} <= set(c_d4['loss']) and 'boundary_radius' not in c_d4['loss'],
          str(c_d4))
    for label, flags in (('--complete without --boundary-weight', csf + ['--complete', 'full']),
                         ('--complete with deep3', ['--fusion', 'sum', '--fusion-stages', 'deep3', '--complete', 'full', '--boundary-weight', '0.3']),
                         ('--complete with --refine', csf + ['--complete', 'full', '--refine', 'gate', '--boundary-weight', '0.3']),
                         ('--boundary-weight with plain CSF', csf + ['--boundary-weight', '0.3'])):
        rc, _, err = cfg(base + flags)
        check(f'{label} refused by train.py', rc == 2, err.strip().splitlines()[-1] if err.strip() else '')
    exp10 = os.path.join(REPO, 'results', 'EGE-UNet-results-exp10')
    if os.path.isdir(exp10):
        from analysis import common as AC
        ok = True
        for dname in sorted(os.listdir(exp10)):
            if not dname.startswith('egeunet_'):
                continue
            sd, _ = AC.load_state_dict(AC.resolve_checkpoint(os.path.join(exp10, dname)))
            try:
                AC.build_model(sd, strict=True)
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f'    {dname}: {e}')
        check('every EXP-10 checkpoint still loads strictly', ok)
    else:
        print('  [skip] results/EGE-UNet-results-exp10 not present')

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) FAILED: {FAILED}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
