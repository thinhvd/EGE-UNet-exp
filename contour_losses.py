'''
EXP-6 loss terms (branch exp/06-contour-loss), all default-off.

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

Region term used by E2 (the ACL region term with c1 = 1, c2 = 0, normalized by the lesion area):
  region         sum [u (1 - g)^2 + (1 - u) g^2] / (sum g + 1)

Distances are computed on the CPU with scipy from the binarized target (g >= 0.5), per sample; an
image with no foreground (or no background) gets a zero distance map, as in Kervadec's code.
'''
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BCELoss, GT_BceDiceLoss

LOSS_MODES = ['bcedice', 'bce_region']
EXTRA_TERMS = ['none', 'tv', 'tv_match', 'area', 'bl', 'snbl']
TV_EPS = 1e-8


# ----------------------------------------------------------------------------- helpers

def _per_image(x):
    return x.reshape(x.size(0), -1)


def tv_map(u):
    '''sqrt(dx^2 + dy^2 + eps) on the (H-1) x (W-1) grid where both differences exist.'''
    dx = u[:, :, 1:, :] - u[:, :, :-1, :]
    dy = u[:, :, :, 1:] - u[:, :, :, :-1]
    return torch.sqrt(dx[:, :, :, :-1] ** 2 + dy[:, :, :-1, :] ** 2 + TV_EPS)


def signed_distance(target):
    '''Signed distance to the ground-truth contour, per sample, same shape as target (B,1,H,W).

    Kervadec's convention: outside the lesion, the distance to the nearest lesion pixel; inside,
    minus (distance to the nearest background pixel - 1), so the pixels on both sides of the
    contour get 1 and 0. Empty or full masks get zeros.'''
    from scipy.ndimage import distance_transform_edt
    g = (target.detach() >= 0.5).cpu().numpy()
    out = np.zeros(g.shape, dtype=np.float32)
    for b in range(g.shape[0]):
        m = g[b, 0]
        if m.any() and not m.all():
            out[b, 0] = distance_transform_edt(~m) - (distance_transform_edt(m) - 1) * m
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


TERM_CLASSES = {'tv': TVLength, 'tv_match': TVMatch, 'area': LogArea,
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
    '''base(gt_pre, out, target) + weight * term(out, target).

    Keeps a running sum of the unweighted term so train.py can log its per-epoch mean.'''
    def __init__(self, base, term, weight):
        super().__init__()
        self.base, self.term, self.weight = base, term, float(weight)
        self.reset_stats()

    def reset_stats(self):
        self._sum, self._n = 0.0, 0

    def epoch_mean(self):
        return self._sum / self._n if self._n else float('nan')

    def forward(self, gt_pre, out, target):
        extra = self.term(out, target)
        self._sum += float(extra.detach())
        self._n += 1
        return self.base(gt_pre, out, target) + self.weight * extra


def describe(loss='bcedice', extra_term='none', extra_weight=None, **kw):
    s = loss if extra_term == 'none' else f'{loss} + {extra_weight} * {extra_term}'
    if extra_term == 'snbl':
        s += f' (tau {kw.get("snbl_tau", 3.0)})'
    if extra_term == 'area':
        s += f' (delta {kw.get("area_delta", 0.05)})'
    return s


def build_criterion(loss='bcedice', extra_term='none', extra_weight=None, snbl_tau=3.0, area_delta=0.05):
    '''The training criterion. Defaults return the original GT_BceDiceLoss(wb=1, wd=1).'''
    if loss not in LOSS_MODES:
        raise ValueError(f'unknown loss {loss!r}; choose from {LOSS_MODES}')
    if extra_term not in EXTRA_TERMS:
        raise ValueError(f'unknown extra term {extra_term!r}; choose from {EXTRA_TERMS}')
    base = GT_BceDiceLoss(wb=1, wd=1) if loss == 'bcedice' else GT_BceRegionLoss(wb=1, wr=1)
    if extra_term == 'none':
        return base
    if extra_weight is None:
        raise ValueError(f'--extra-term {extra_term} needs --extra-weight')
    if extra_term == 'snbl':
        term = ScaleNormBoundaryLoss(tau=snbl_tau)
    elif extra_term == 'area':
        term = LogArea(delta=area_delta)
    else:
        term = TERM_CLASSES[extra_term]()
    return WithExtraTerm(base, term, extra_weight)
