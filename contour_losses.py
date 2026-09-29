'''
EXP-6 / EXP-7 loss terms (branches exp/06-contour-loss, exp/07-e3a-refine), all default-off.

With `--loss bcedice --extra-term none` (the defaults) build_criterion() returns the original
utils.GT_BceDiceLoss object, so training is the author's computation bit for bit.

Every extra term looks only at the final output and is ADDED to the unchanged GT_BceDiceLoss:

    L = GT_BceDice(gt_pre, out, target) + weight * term(out, target)

The five deep-supervision heads keep plain BceDice, so the masks GAB reads are trained as in the
baseline. The one exception is `--loss bce_region` (E2), which swaps the Dice part of all six BceDice
terms for the normalized active-contour region term and adds nothing.

Terms (u = probability map of the final output, g = target mask; computed per image, then averaged
over the batch unless stated otherwise):

  tv        E1a  ACL length term (Chen et al., CVPR 2019), mean form: mean sqrt(dx^2 + dy^2 + eps)
  tv_match  E1b  |TV(u) / TV(g) - 1|: the same length, matched to the ground truth instead of minimized
  area      E4   SmoothL1_delta(log((sum u + 1) / (sum g + 1))): scale-invariant area term, the
                 Chan-Vese balloon term that ACL drops, with its sign taken from the ground truth
  bl        E3a  boundary loss (Kervadec et al., MIDL 2019): mean over pixels of phi_G * u, with phi_G
                 the signed distance to the ground-truth contour in pixels (negative inside)
  snbl      E3b  sum min(|phi_G| / r_G, tau) * |u - g| / (sum g + 1), with r_G = sqrt(|G| / pi): the
                 same distance weighting measured in radii of the lesion itself, so a spill of a
                 given relative width costs the same whatever the lesion size

  fn_dp     EXP-7 mean over pixels of g * (1 - u) * d_P, with d_P the distance to the current
                 prediction {u >= 0.5} (no gradient through d_P), capped at the lesion's inradius:
                 the prediction-side half of Karimi & Salcudean's two-sided distance loss (IEEE TMI
                 2020). A missed pixel costs more the farther it lies from what the model draws, so
                 dropping a whole chunk of a lesion is expensive while a thin missed rim is cheap.
                 Added to bl it gives E3a a penalty for misses as well as for spill.

Several terms can be added at once, each with its own weight:
    --extra-term bl,fn_dp --extra-weight 0.095,0.35

Region term used by E2 (the ACL region term with c1 = 1, c2 = 0, normalized by the lesion area):
  region         sum [u (1 - g)^2 + (1 - u) g^2] / (sum g + 1)

Distances are computed on the CPU with scipy from the binarized target (g >= 0.5), per sample; an
image with no foreground (or no background) gets a zero distance map, as in Kervadec's code.

EXP-9 boundary-band auxiliary loss (--boundary-weight, only with the boundary-guided fusion):
  bnd  the boundary heads of the fusion (models/fusion.py, 'bg_stage') are trained to find the lesion
       contour. Target: dilate3x3(g) - erode3x3(g) of the binarized mask, a ~2 px band on both sides
       of the contour (max-pool padding is -inf, so the image frame is never marked as contour),
       max-pooled down to each head's own grid (32 / 64 / 128 for dec3 / dec4 / dec5) so the band
       is at least one cell wide there. Loss = 0.1 * BD(dec3) + 0.2 * BD(dec4) + 0.3 * BD(dec5),
       BD = BceDice(wb=0.5, wd=1) on sigmoid(logit) at the head's grid (weights from LB-UNet,
       not calibrated here). With weight 0 the terms are only logged, under no_grad.
'''
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BCELoss, BceDiceLoss, GT_BceDiceLoss

LOSS_MODES = ['bcedice', 'bce_region']
EXTRA_TERMS = ['none', 'tv', 'tv_match', 'area', 'bl', 'snbl', 'fn_dp']
TV_EPS = 1e-8
BOUNDARY_STAGE_WEIGHTS = {'dec3': 0.1, 'dec4': 0.2, 'dec5': 0.3}


# ----------------------------------------------------------------------------- helpers

def _per_image(x):
    return x.reshape(x.size(0), -1)


def tv_map(u):
    '''sqrt(dx^2 + dy^2 + eps) on the (H-1) x (W-1) grid where both differences exist.'''
    dx = u[:, :, 1:, :] - u[:, :, :-1, :]
    dy = u[:, :, :, 1:] - u[:, :, :, :-1]
    return torch.sqrt(dx[:, :, :, :-1] ** 2 + dy[:, :, :-1, :] ** 2 + TV_EPS)


def edt_torch(feature):
    '''Exact Euclidean distance of every pixel to the nearest True pixel of `feature` (H, W bool).

    Two separable passes of squared distances (columns, then rows), each a brute-force min over one
    axis. All intermediate values are exact integers in float32, so the result equals
    scipy.ndimage.distance_transform_edt(~feature) up to the last bit of the square root. Needs
    H*H*W + H*W*W floats of scratch (67 MB each at 256x256): meant for the GPU, where it takes a few
    ms per image instead of the ~20 ms of scipy on a busy CPU.'''
    h, w = feature.shape
    big = torch.tensor(1e8, dtype=torch.float32, device=feature.device)   # "no feature in this column"
    f = torch.where(feature, torch.zeros((), device=feature.device), big)
    ys = torch.arange(h, dtype=torch.float32, device=feature.device)
    xs = torch.arange(w, dtype=torch.float32, device=feature.device)
    col = (ys[:, None, None] - ys[None, :, None]) ** 2 + f[None, :, :]      # (y, y', x)
    g = col.amin(1)                                                        # (y, x): along columns
    row = g[:, None, :] + (xs[:, None] - xs[None, :])[None] ** 2            # (y, x, x')
    return row.amin(2).sqrt()


def signed_distance(target):
    '''Signed distance to the ground-truth contour, per sample, same shape as target (B,1,H,W).

    Kervadec's convention: outside the lesion, the distance to the nearest lesion pixel; inside,
    minus (distance to the nearest background pixel - 1), so the pixels on both sides of the
    contour get 1 and 0. Empty or full masks get zeros. On a CUDA tensor the transform runs on the
    GPU (edt_torch, same values); on CPU it uses scipy.'''
    g = target.detach() >= 0.5
    if g.is_cuda:
        out = torch.zeros(g.shape, dtype=torch.float32, device=g.device)
        for b in range(g.shape[0]):
            m = g[b, 0]
            if m.any() and not m.all():
                out[b, 0] = edt_torch(m) - (edt_torch(~m) - 1) * m
        return out
    from scipy.ndimage import distance_transform_edt
    g = g.cpu().numpy()
    out = np.zeros(g.shape, dtype=np.float32)
    for b in range(g.shape[0]):
        m = g[b, 0]
        if m.any() and not m.all():
            out[b, 0] = distance_transform_edt(~m) - (distance_transform_edt(m) - 1) * m
    return torch.from_numpy(out).to(target.device)


def boundary_band(target):
    '''EXP-9 contour band of the binarized target (B,1,H,W): dilate3x3 - erode3x3, 1 on the pixels on
    either side of the contour. max_pool2d pads with -inf, so the image frame counts as neither
    lesion nor background: a lesion touching the frame gets no band along it.'''
    m = (target.detach() > 0.5).float()
    return F.max_pool2d(m, 3, 1, 1) - (-F.max_pool2d(-m, 3, 1, 1))


def band_at_grid(band, size):
    '''Max-pool a full-resolution band down to a head's grid `size` (h, w): a cell is contour if any
    of its pixels is, so the band stays at least one cell wide.'''
    h, w = int(size[0]), int(size[1])
    H, W = band.shape[-2:]
    if (h, w) == (H, W):
        return band
    if H % h or W % w or H // h != W // w:
        raise ValueError(f'band {H}x{W} cannot be pooled to {h}x{w}')
    return F.max_pool2d(band, H // h, H // h)


def distance_to_prediction(u, target, backend=None):
    '''Distance of every pixel to the current prediction {u >= 0.5}, per sample, capped at the
    lesion's inradius (the largest distance from a lesion pixel to the background). The weight
    carries no gradient. An empty prediction gives the inradius everywhere; an empty lesion zeros.
    GPU tensors use edt_torch, CPU tensors scipy (same values); backend='torch' forces edt_torch.'''
    pred = u.detach() >= 0.5
    g = target.detach() >= 0.5
    h, w = g.shape[-2:]
    if backend == 'torch' or (backend is None and g.is_cuda):
        out = torch.zeros(g.shape, dtype=torch.float32, device=g.device)
        for b in range(g.shape[0]):
            m = g[b, 0]
            if not m.any():
                continue
            cap = edt_torch(~m)[m].max() if not m.all() else torch.tensor(float(max(h, w)), device=g.device)
            p = pred[b, 0]
            out[b, 0] = torch.minimum(edt_torch(p), cap) if p.any() else cap
        return out
    from scipy.ndimage import distance_transform_edt
    g, pred = g.cpu().numpy(), pred.cpu().numpy()
    out = np.zeros(g.shape, dtype=np.float32)
    for b in range(g.shape[0]):
        m = g[b, 0]
        if not m.any():
            continue
        cap = distance_transform_edt(m)[m].max() if not m.all() else float(max(h, w))
        p = pred[b, 0]
        out[b, 0] = np.minimum(distance_transform_edt(~p), cap) if p.any() else cap
    return torch.from_numpy(out).to(target.device)


# ----------------------------------------------------------------------------- terms

class TVLength(nn.Module):
    '''E1a: the ACL length term, mean over pixels (the form the official code uses since 2020).'''
    def forward(self, u, target):
        return tv_map(u).mean()


class TVMatch(nn.Module):
    '''E1b: per-image |TV(u) / TV(g) - 1|. Two-sided, so no shrinking pull; a ratio, so scale-free.'''
    def forward(self, u, target):
        tv_u = _per_image(tv_map(u)).sum(1)
        tv_g = _per_image(tv_map(target)).sum(1)
        # a lesion's contour is tens to hundreds of pixels long; the clamp only guards an empty mask
        return (tv_u / tv_g.clamp_min(1.0) - 1.0).abs().mean()


class LogArea(nn.Module):
    '''E4: per-image SmoothL1 of log((sum u + 1) / (sum g + 1)). Its gradient is the same on every
    pixel of an image, +-1 / (sum u + 1) outside the SmoothL1 knee: a balloon force whose strength is
    set by the relative, not the absolute, area error.'''
    def __init__(self, delta=0.05):
        super().__init__()
        self.delta = delta

    def forward(self, u, target):
        x = torch.log((_per_image(u).sum(1) + 1.0) / (_per_image(target).sum(1) + 1.0))
        return F.smooth_l1_loss(x, torch.zeros_like(x), beta=self.delta)


class BoundaryLoss(nn.Module):
    '''E3a: Kervadec et al. boundary loss, mean over all pixels of phi_G * u (phi in pixels).'''
    def forward(self, u, target):
        return (signed_distance(target) * u).mean()


class ScaleNormBoundaryLoss(nn.Module):
    '''E3b: distance weight measured in lesion radii, clipped at tau, normalized by the lesion area.

    Up to a constant this is the boundary loss with phi / r_G instead of phi (for binary g,
    sum |phi| |u - g| = sum phi u + sum_inside |phi|), clipped so that anything beyond tau radii
    costs the same.'''
    def __init__(self, tau=3.0):
        super().__init__()
        self.tau = tau

    def forward(self, u, target):
        phi = signed_distance(target)
        g = (target >= 0.5).float()
        area = _per_image(g).sum(1)
        radius = torch.sqrt(area / math.pi).clamp_min(1.0)
        weight = torch.clamp(phi.abs() / radius.view(-1, 1, 1, 1), max=self.tau)
        per_image = _per_image(weight * (u - g).abs()).sum(1) / (area + 1.0)
        return per_image.mean()


class RegionLoss(nn.Module):
    '''The ACL region term (c1 = 1, c2 = 0) per image, divided by the lesion area.

    For binary g this is (soft FP + soft FN) / |G|. Dividing by |G|, which does not depend on u,
    keeps the loss linear in u: its decision threshold is 0.5 at every lesion size, unlike Dice.'''
    def forward(self, u, target):
        num = _per_image(u * (1 - target) ** 2 + (1 - u) * target ** 2).sum(1)
        return (num / (_per_image(target).sum(1) + 1.0)).mean()


class FNDistance(nn.Module):
    '''EXP-7: mean over pixels of g * (1 - u) * d_P (see distance_to_prediction). Zero when the
    prediction covers the lesion; only missed lesion pixels get a gradient.'''
    def forward(self, u, target):
        g = (target >= 0.5).float()
        return (g * (1 - u) * distance_to_prediction(u, target)).mean()


TERM_CLASSES = {'fn_dp': FNDistance, 'tv': TVLength, 'tv_match': TVMatch, 'area': LogArea,
                'bl': BoundaryLoss, 'snbl': ScaleNormBoundaryLoss}


# ----------------------------------------------------------------------------- criteria

class BceRegionLoss(nn.Module):
    def __init__(self, wb=1, wr=1):
        super().__init__()
        self.bce = BCELoss()
        self.region = RegionLoss()
        self.wb, self.wr = wb, wr

    def forward(self, pred, target):
        return self.wr * self.region(pred, target) + self.wb * self.bce(pred, target)


class GT_BceRegionLoss(nn.Module):
    '''E2: GT_BceDiceLoss with every Dice replaced by RegionLoss; same deep-supervision weights.'''
    def __init__(self, wb=1, wr=1):
        super().__init__()
        self.bceregion = BceRegionLoss(wb, wr)

    def forward(self, gt_pre, out, target):
        loss = self.bceregion(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        loss_ds = self.bceregion(gt_pre5, target) * 0.1 + self.bceregion(gt_pre4, target) * 0.2 + \
            self.bceregion(gt_pre3, target) * 0.3 + self.bceregion(gt_pre2, target) * 0.4 + \
            self.bceregion(gt_pre1, target) * 0.5
        return loss + loss_ds


class WithExtraTerm(nn.Module):
    '''base(gt_pre, out, target) + sum_k weight_k * term_k(out, target).

    `terms` is a list of (name, module, weight). Keeps a running sum of every unweighted term so
    train.py can log its per-epoch mean. With one term the arithmetic is exactly EXP-6's
    base + weight * term.'''
    def __init__(self, base, terms):
        super().__init__()
        self.base = base
        self.names = [n for n, _, _ in terms]
        self.terms = nn.ModuleList([t for _, t, _ in terms])
        self.weights = [float(w) for _, _, w in terms]
        self.reset_stats()

    def reset_stats(self):
        self._sums, self._n = [0.0] * len(self.terms), 0

    def epoch_means(self):
        return {n: (s / self._n if self._n else float('nan')) for n, s in zip(self.names, self._sums)}

    def epoch_mean(self):
        return self.epoch_means()[self.names[0]]

    def forward(self, gt_pre, out, target):
        extras = [t(out, target) for t in self.terms]
        for k, e in enumerate(extras):
            self._sums[k] += float(e.detach())
        self._n += 1
        total = self.base(gt_pre, out, target)
        for w, e in zip(self.weights, extras):
            total = total + w * e
        return total


class WithBoundaryAux(nn.Module):
    '''EXP-9: inner(gt_pre, out, target) + weight * sum_s w_s * BD(sigmoid(logit_s), band_s).

    The boundary logits come from the boundary-guided fusion as gt_pre.boundary (a
    models.fusion.DeepSupervisionOutputs); `inner` is the usual criterion (GT_BceDiceLoss or a
    WithExtraTerm around it). Keeps a running mean of every unweighted band term, merged with the
    inner criterion's, so train.py logs them as train_extra_* columns. With weight 0 the band terms
    are computed under no_grad for the log only and the result is inner's value, unchanged.'''
    def __init__(self, inner, weight, stage_weights=None):
        super().__init__()
        self.inner = inner
        self.weight = float(weight)
        self.stage_weights = dict(stage_weights or BOUNDARY_STAGE_WEIGHTS)
        self.bd = BceDiceLoss(wb=0.5, wd=1)
        self.bnd_names = [f'bnd_{s}' for s in self.stage_weights]
        self.names = list(getattr(inner, 'names', [])) + self.bnd_names
        self.reset_stats()

    def reset_stats(self):
        if hasattr(self.inner, 'reset_stats'):
            self.inner.reset_stats()
        self._sums, self._n = [0.0] * len(self.bnd_names), 0

    def epoch_means(self):
        out = dict(self.inner.epoch_means()) if hasattr(self.inner, 'epoch_means') else {}
        out.update({n: (s / self._n if self._n else float('nan')) for n, s in zip(self.bnd_names, self._sums)})
        return out

    def epoch_mean(self):
        return self.epoch_means()[self.names[0]]

    def band_terms(self, boundary, target):
        '''Unweighted BD per stage, in stage_weights order (also used by analysis/exp09_mechanism.py).'''
        band = boundary_band(target)
        return [self.bd(torch.sigmoid(boundary[s]), band_at_grid(band, boundary[s].shape[-2:]))
                for s in self.stage_weights]

    def forward(self, gt_pre, out, target):
        boundary = getattr(gt_pre, 'boundary', None)
        if boundary is None or set(boundary) != set(self.stage_weights):
            raise ValueError('the boundary-band loss needs the boundary logits of the boundary-guided fusion '
                             f'for stages {sorted(self.stage_weights)}; got '
                             f'{None if boundary is None else sorted(boundary)}')
        total = self.inner(gt_pre, out, target)
        if self.weight == 0:
            with torch.no_grad():
                terms = self.band_terms(boundary, target)
        else:
            terms = self.band_terms(boundary, target)
        for k, t in enumerate(terms):
            self._sums[k] += float(t.detach())
        self._n += 1
        if self.weight != 0:
            total = total + self.weight * sum(w * t for w, t in zip(self.stage_weights.values(), terms))
        return total


def parse_terms(extra_term='none', extra_weight=None):
    '''"bl,fn_dp" and "0.095,0.35" (or a float) -> [('bl', 0.095), ('fn_dp', 0.35)]; [] for none.'''
    names = [t.strip() for t in str(extra_term).split(',') if t.strip()]
    if names in ([], ['none']):
        return []
    for n in names:
        if n not in EXTRA_TERMS or n == 'none':
            raise ValueError(f'unknown extra term {n!r}; choose from {EXTRA_TERMS[1:]}')
    if len(set(names)) != len(names):
        raise ValueError(f'extra term listed twice: {extra_term!r}')
    if extra_weight is None:
        raise ValueError(f'--extra-term {extra_term} needs --extra-weight')
    weights = [float(w) for w in str(extra_weight).split(',')] if isinstance(extra_weight, str) \
        else [float(extra_weight)]
    if len(weights) != len(names):
        raise ValueError(f'{len(names)} extra terms but {len(weights)} weights: {extra_term!r} / {extra_weight!r}')
    return list(zip(names, weights))


def describe(loss='bcedice', extra_term='none', extra_weight=None, boundary_weight=None, **kw):
    terms = parse_terms(extra_term, extra_weight)
    s = loss + ''.join(f' + {w:g} * {n}' for n, w in terms)
    names = [n for n, _ in terms]
    if 'snbl' in names:
        s += f' (tau {kw.get("snbl_tau", 3.0)})'
    if 'area' in names:
        s += f' (delta {kw.get("area_delta", 0.05)})'
    if boundary_weight is not None:
        bw = float(boundary_weight)
        s += (f' + {bw:g} * bnd (0.1*dec3 + 0.2*dec4 + 0.3*dec5, BceDice(0.5,1) on the contour band per head grid)'
              if bw != 0 else ' (+ bnd contour-band loss logged only, weight 0)')
    return s


def make_term(name, snbl_tau=3.0, area_delta=0.05):
    if name == 'snbl':
        return ScaleNormBoundaryLoss(tau=snbl_tau)
    if name == 'area':
        return LogArea(delta=area_delta)
    return TERM_CLASSES[name]()


def build_criterion(loss='bcedice', extra_term='none', extra_weight=None, snbl_tau=3.0, area_delta=0.05,
                    boundary_weight=None):
    '''The training criterion. Defaults return the original GT_BceDiceLoss(wb=1, wd=1); extra terms
    alone return EXP-6/7's WithExtraTerm; a boundary_weight (EXP-9, 0 allowed) wraps either in
    WithBoundaryAux.'''
    if loss not in LOSS_MODES:
        raise ValueError(f'unknown loss {loss!r}; choose from {LOSS_MODES}')
    terms = parse_terms(extra_term, extra_weight)
    base = GT_BceDiceLoss(wb=1, wd=1) if loss == 'bcedice' else GT_BceRegionLoss(wb=1, wr=1)
    crit = base if not terms else WithExtraTerm(base, [(n, make_term(n, snbl_tau, area_delta), w) for n, w in terms])
    if boundary_weight is None:
        return crit
    if float(boundary_weight) < 0:
        raise ValueError(f'boundary_weight must be >= 0, got {boundary_weight}')
    return WithBoundaryAux(crit, boundary_weight)
