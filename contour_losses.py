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

EXP-11 (models/completion.py, --complete):
  ring  the barrier head of the region completion is trained on the +-r px ring of the contour
        (boundary_ring: dilate - erode with a (2r+1) square max-pool, r = 3 by default), max-pooled to
        the head's 128 grid; same BceDice(wb=0.5, wd=1), weight --boundary-weight, column bnd_ring.
  hp    hard-positive term on the final output (--extra-term hp): mean over all B*H*W pixels of
        1[d_in > r] * (1 - u)^gamma * BCE(u, 1), d_in = distance of a lesion pixel to the background
        (r = 10 px, gamma = 2). Normalized like utils.BCELoss, so its weight multiplies BCE directly on
        the deep lesion pixels; F.binary_cross_entropy clamps the log at -100 as the original loss does.

EXP-14 (70:10:20 protocol; extra terms added to the original loss with a weight fixed on train / val):
  ihp      interior hard-positive: mean over all B*H*W pixels of 1[g = 1, depth_norm >= tau, u < p_low] *
           (1 - u)^gamma * BCE(u, 1), depth_norm = distance of a lesion pixel to the background divided by the
           largest such distance in that image (0 on the contour, 1 at the deepest point), so "interior" means the
           same fraction of the lesion whatever its size; the u < p_low gate (no gradient) keeps only pixels the
           model currently misses.
  focal    binary focal loss (Lin et al. 2017) on the final output, mean over pixels:
           -alpha g (1 - u)^gamma log u - (1 - alpha)(1 - g) u^gamma log(1 - u); alpha > 0.5 weights class 1 up.
  tversky  Tversky loss (Salehi et al. 2017) per image, mean over the batch:
           1 - (TP + 1) / (TP + a FP + b FN + 1) with soft counts; b > a penalizes misses more than spill.

EXP-15 (models/lsa_fusion.py, --lsa): auxiliary losses of the two fusion experts (WithExpertAux)
  0.5 * [BceDice(detail seg) + BceDice(0.5,1)(detail ring, +-3 px contour ring)]
  + 0.3 * [BceDice(completion seg) + ihr(completion seg)]
  ihr = focal BCE (gamma 2) over the interior region H = {g = 1, normalized depth >= tau}, normalized by the number of
  H pixels in the batch (not by all pixels, and not per image), so it neither vanishes in the total like EXP-11's hp
  nor lets a small lesion's few interior pixels weigh as much as a large lesion's. Weights 0.5 / 0.3 = the
  deep-supervision weights of the experts' grids (128 / 32), fixed before training. Aux logits are upsampled to the
  input size as the deep-supervision maps are. In 'probe' mode the aux heads read detached experts (readout only).
  Mean router gates g_C and g_null inside / outside the lesion are logged per stage (route mode).

EXP-13 (models/propagation.py, --propagate):
  aff   the edge head of the core-to-region propagation, S = P(same label) on edges between neighbouring cells of
        the 128 grid (GT average-pooled 2x2, >= 0.5). Edge groups: X = edges crossing the GT contour, LL = both
        cells lesion, BB = both background, LL_hard = the hard_frac (20 %) LL edges of each image with the largest
        colour + P1 contrast (the dark-core edge must not become a cut). Loss = 0.5 * mean_X[-log(1 - S)] +
        0.25 * (mean_LL + mean_BB) / 2 * [-log S] + 0.25 * mean_LLhard[-log S], each mean over the batch's edges
        of that group. Logged per term (aff_X, aff_LL, aff_BB, aff_hard) plus rank_acc, the share of images whose
        mean S on LL_hard exceeds the mean S on X (not part of the loss).
  comp  (pair_coupled only) BCE of the propagated map R against the GT cells, over the GT dilated by 10 cells,
        seed cells excluded.
'''
import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BCELoss, BceDiceLoss, GT_BceDiceLoss

LOSS_MODES = ['bcedice', 'bce_region']
EXTRA_TERMS = ['none', 'tv', 'tv_match', 'area', 'bl', 'snbl', 'fn_dp', 'hp', 'ihp', 'focal', 'tversky']
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


def boundary_zone(target, radius):
    '''EXP-10 contour zone of the binarized target (B,1,H,W): 1 on every pixel whose Euclidean distance
    to the contour is at most `radius` px, measured as analysis/exp09_mechanism.py does (outside the
    lesion: distance to the nearest lesion pixel; inside: distance to the nearest background pixel).
    Empty or full masks get an all-zero map. GPU tensors use edt_torch, CPU tensors scipy.'''
    g = target.detach() >= 0.5
    out = torch.zeros(g.shape, dtype=torch.float32, device=g.device)
    if not g.is_cuda:
        from scipy.ndimage import distance_transform_edt
    for b in range(g.shape[0]):
        m = g[b, 0]
        if not (m.any() and not m.all()):
            continue
        # edt_torch(f) = distance to the nearest True pixel of f; scipy's edt(f) = distance of each
        # True pixel of f to the nearest False pixel. Inside the lesion we want the distance to the
        # background, outside the distance to the lesion.
        if g.is_cuda:
            d = torch.where(m, edt_torch(~m), edt_torch(m))
        else:
            mn = m.numpy()
            d = torch.from_numpy(np.where(mn, distance_transform_edt(mn), distance_transform_edt(~mn)))
        out[b, 0] = (d <= radius).float()
    return out


def boundary_ring(target, radius=3):
    '''EXP-11 contour ring of the binarized target (B,1,H,W): dilate - erode with a (2r+1) x (2r+1) square,
    i.e. 1 on the pixels within r px (Chebyshev) of the contour on either side. Same -inf padding as
    boundary_band, so the image frame is never marked. radius=1 gives boundary_band.'''
    r = int(radius)
    if r < 1:
        raise ValueError(f'ring radius must be >= 1, got {radius}')
    m = (target.detach() > 0.5).float()
    k = 2 * r + 1
    return F.max_pool2d(m, k, 1, r) - (-F.max_pool2d(-m, k, 1, r))


def deep_positive_mask(target, radius=10.0, backend=None):
    '''EXP-11: 1 on lesion pixels (target >= 0.5) whose Euclidean distance to the nearest background pixel
    exceeds `radius` px: the "far" FN band of analysis/exp09_mechanism.GroundTruth (fn_code == 2). A mask
    with no background counts as all deep; an empty mask gives zeros. GPU tensors (or backend='torch')
    use edt_torch, CPU tensors scipy.'''
    g = target.detach() >= 0.5
    out = torch.zeros(g.shape, dtype=torch.float32, device=g.device)
    use_torch = backend == 'torch' or (backend is None and g.is_cuda)
    if not use_torch:
        from scipy.ndimage import distance_transform_edt
    for b in range(g.shape[0]):
        m = g[b, 0]
        if not m.any():
            continue
        if m.all():
            out[b, 0] = 1.0
            continue
        if use_torch:
            d = edt_torch(~m)
        else:
            d = torch.from_numpy(distance_transform_edt(m.cpu().numpy())).to(g.device)
        out[b, 0] = (m & (d > radius)).float()
    return out


def edge_masks(target, size, hard_contrast=None, hard_frac=0.2):
    '''EXP-13 edge groups on the propagation grid. target (B,1,H,W) at full resolution -> cells at `size` by 2x2
    average pooling (>= 0.5). Returns dict of (B,4,h,w) bool masks X / LL / BB (+ LL_hard when hard_contrast,
    (B,4,h,w), is given) and the cell labels (B,1,h,w). Edges leaving the grid belong to no group.'''
    from models.propagation import OFFS_HALF, shift
    lab = F.adaptive_avg_pool2d((target.detach() >= 0.5).float(), size) >= 0.5
    ones = torch.ones_like(lab, dtype=torch.float32)
    X, LL, BB = [], [], []
    for dy, dx in OFFS_HALF:
        valid = shift(ones, dy, dx) > 0.5
        ly = shift(lab.float(), dy, dx) > 0.5
        X.append(valid & (lab != ly))
        LL.append(valid & lab & ly)
        BB.append(valid & ~lab & ~ly)
    out = {'X': torch.cat(X, 1), 'LL': torch.cat(LL, 1), 'BB': torch.cat(BB, 1), 'lab': lab}
    if hard_contrast is not None:
        hard = torch.zeros_like(out['LL'])
        for b in range(lab.shape[0]):
            m = out['LL'][b]
            if m.sum() >= 10:
                v = hard_contrast[b][m]
                thr = torch.quantile(v.float(), 1 - hard_frac) if v.numel() <= 16_000_000 else v.float().sort()[0][int((1 - hard_frac) * v.numel())]
                hard[b] = m & (hard_contrast[b] >= thr)
        out['LL_hard'] = hard
    return out


def normalized_depth(target, backend=None):
    '''EXP-14: per image, the distance of every lesion pixel to the nearest background pixel divided by the largest
    such distance in that image (1 at the deepest lesion pixel, -> 0 at the contour); 0 outside the lesion and for
    empty masks; a mask with no background gets 1 on every lesion pixel. GPU tensors (or backend='torch') use
    edt_torch, CPU tensors scipy.'''
    g = target.detach() >= 0.5
    out = torch.zeros(g.shape, dtype=torch.float32, device=g.device)
    use_torch = backend == 'torch' or (backend is None and g.is_cuda)
    if not use_torch:
        from scipy.ndimage import distance_transform_edt
    for b in range(g.shape[0]):
        m = g[b, 0]
        if not m.any():
            continue
        if m.all():
            out[b, 0] = 1.0
            continue
        d = edt_torch(~m) if use_torch else torch.from_numpy(distance_transform_edt(m.cpu().numpy())).float().to(g.device)
        d = d * m
        out[b, 0] = d / d.max()
    return out


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


class HardPositive(nn.Module):
    '''EXP-11: mean over all pixels of 1[d_in > radius] * (1 - u)^gamma * BCE(u, 1) (focal BCE on the deep
    lesion pixels; the (1 - u)^gamma factor is part of the loss, not detached). A pixel the model already
    paints (u -> 1) costs nothing; a confidently missed one (u -> 0) gets the full BCE push.'''
    def __init__(self, radius=10.0, gamma=2.0):
        super().__init__()
        if gamma != 0 and gamma < 1:
            raise ValueError(f'hp gamma must be 0 or >= 1, got {gamma}')
        self.radius, self.gamma = float(radius), float(gamma)

    def forward(self, u, target):
        h = deep_positive_mask(target, self.radius)
        bce = F.binary_cross_entropy(u, torch.ones_like(u), reduction='none')
        w = (1 - u) ** self.gamma if self.gamma else 1.0
        return (h * w * bce).mean()


def affinity_terms(s_logit, masks):
    '''Unweighted mean BCE of the edge logits per group (0 for an empty group) and the combined affinity loss.'''
    def mean_over(t, m):
        n = m.sum()
        return (t * m).sum() / n if n > 0 else t.sum() * 0
    cut = F.softplus(s_logit)          # -log(1 - S)
    keep = F.softplus(-s_logit)        # -log S
    t = {'aff_X': mean_over(cut, masks['X']), 'aff_LL': mean_over(keep, masks['LL']),
         'aff_BB': mean_over(keep, masks['BB']), 'aff_hard': mean_over(keep, masks['LL_hard'])}
    loss = 0.5 * t['aff_X'] + 0.25 * 0.5 * (t['aff_LL'] + t['aff_BB']) + 0.25 * t['aff_hard']
    return loss, t


def ranking_accuracy(s_logit, masks):
    '''Share of images (with both groups) whose mean S on LL_hard exceeds the mean S on X.'''
    with torch.no_grad():
        s = torch.sigmoid(s_logit)
        wins, n = 0, 0
        for b in range(s.shape[0]):
            h, x = masks['LL_hard'][b], masks['X'][b]
            if h.any() and x.any():
                wins += int(s[b][h].mean() > s[b][x].mean())
                n += 1
        return wins / n if n else float('nan')


def completion_term(r, lab, seed, dilate_cells=10):
    '''BCE(R, GT cell labels) over the GT dilated by dilate_cells, seed cells excluded (EXP-13 pair_coupled).'''
    k = 2 * dilate_cells + 1
    region = (F.max_pool2d(lab.float(), k, 1, dilate_cells) > 0.5) & (seed < 0.5)
    if not region.any():
        return r.sum() * 0
    bce = F.binary_cross_entropy(r.clamp(1e-6, 1 - 1e-6), lab.float(), reduction='none')
    return (bce * region).sum() / region.sum()


class InteriorHardPositive(nn.Module):
    '''EXP-14: focal BCE on the lesion pixels that lie in the interior (normalized depth >= tau) and that the model
    currently misses (u < p_low, gate without gradient). Mean over all pixels, like utils.BCELoss.'''
    def __init__(self, tau=0.2, gamma=2.0, p_low=0.5):
        super().__init__()
        if not 0 <= tau < 1 or not 0 < p_low <= 1 or (gamma != 0 and gamma < 1):
            raise ValueError(f'ihp needs 0 <= tau < 1, 0 < p_low <= 1, gamma 0 or >= 1; got {tau}, {p_low}, {gamma}')
        self.tau, self.gamma, self.p_low = float(tau), float(gamma), float(p_low)

    def mask(self, u, target):
        return ((normalized_depth(target) >= self.tau) & (target >= 0.5) & (u.detach() < self.p_low)).float()

    def forward(self, u, target):
        m = self.mask(u, target)
        bce = F.binary_cross_entropy(u, torch.ones_like(u), reduction='none')
        w = (1 - u) ** self.gamma if self.gamma else 1.0
        return (m * w * bce).mean()


class FocalLoss(nn.Module):
    '''EXP-14: binary focal loss on probabilities, mean over pixels (log clamped at -100 like nn.BCELoss).'''
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        if not 0 < alpha < 1 or gamma < 0:
            raise ValueError(f'focal needs 0 < alpha < 1 and gamma >= 0; got {alpha}, {gamma}')
        self.alpha, self.gamma = float(alpha), float(gamma)

    def forward(self, u, target):
        g = (target >= 0.5).float()
        # F.binary_cross_entropy clamps the log at -100 with a finite gradient at u = 0 or 1 (torch.log would give nan)
        pos = F.binary_cross_entropy(u, torch.ones_like(u), reduction='none')
        neg = F.binary_cross_entropy(u, torch.zeros_like(u), reduction='none')
        loss = self.alpha * g * (1 - u) ** self.gamma * pos + (1 - self.alpha) * (1 - g) * u ** self.gamma * neg
        return loss.mean()


class TverskyLoss(nn.Module):
    '''EXP-14: 1 - (TP + 1) / (TP + alpha FP + beta FN + 1) per image with soft counts, mean over the batch.'''
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError(f'tversky needs alpha, beta >= 0; got {alpha}, {beta}')
        self.alpha, self.beta = float(alpha), float(beta)

    def forward(self, u, target):
        g = (target >= 0.5).float()
        tp = _per_image(u * g).sum(1)
        fp = _per_image(u * (1 - g)).sum(1)
        fn = _per_image((1 - u) * g).sum(1)
        return (1 - (tp + 1) / (tp + self.alpha * fp + self.beta * fn + 1)).mean()


class InteriorHardPositiveRegion(nn.Module):
    '''EXP-15: focal BCE on the interior region H = {target = 1, normalized depth >= tau}, summed over H and divided by
    the number of H pixels in the batch; 0 when the batch has no H pixel.'''
    def __init__(self, tau=0.15, gamma=2.0):
        super().__init__()
        self.tau, self.gamma = float(tau), float(gamma)

    def forward(self, u, target):
        h = ((normalized_depth(target) >= self.tau) & (target >= 0.5)).float()
        n = h.sum()
        if n == 0:
            return u.sum() * 0
        bce = F.binary_cross_entropy(u, torch.ones_like(u), reduction='none')
        w = (1 - u) ** self.gamma if self.gamma else 1.0
        return (h * w * bce).sum() / n


TERM_CLASSES = {'fn_dp': FNDistance, 'tv': TVLength, 'tv_match': TVMatch, 'area': LogArea,
                'bl': BoundaryLoss, 'snbl': ScaleNormBoundaryLoss, 'hp': HardPositive,
                'ihp': InteriorHardPositive, 'focal': FocalLoss, 'tversky': TverskyLoss}


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
    are computed under no_grad for the log only and the result is inner's value, unchanged.
    `target_fn(target)` builds the full-resolution target map; the default is the 2-px contour band
    (EXP-9), EXP-10 passes a contour zone of a given radius, EXP-11 a +-r px contour ring.'''
    def __init__(self, inner, weight, stage_weights=None, target_fn=None):
        super().__init__()
        self.inner = inner
        self.weight = float(weight)
        self.stage_weights = dict(stage_weights or BOUNDARY_STAGE_WEIGHTS)
        self.target_fn = target_fn or boundary_band
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
        band = self.target_fn(target)
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


class WithPropagationAux(nn.Module):
    '''EXP-13: inner(gt_pre, out, target) + w_aff * affinity loss + w_comp * completion loss. The edge logits, the
    propagated map, the seed and the edge contrast come from models.propagation.CorePropagation through
    gt_pre.boundary ('aff', 'prop', 'seed', 'contrast'). Keeps running means of every unweighted term and of
    rank_acc for the train_extra_* log columns.'''
    def __init__(self, inner, w_aff, w_comp=0.0, hard_frac=0.2):
        super().__init__()
        self.inner = inner
        self.w_aff, self.w_comp, self.hard_frac = float(w_aff), float(w_comp), float(hard_frac)
        self.own = ['aff_X', 'aff_LL', 'aff_BB', 'aff_hard', 'rank_acc'] + (['comp'] if self.w_comp > 0 else [])
        self.names = list(getattr(inner, 'names', [])) + self.own
        self.reset_stats()

    def reset_stats(self):
        if hasattr(self.inner, 'reset_stats'):
            self.inner.reset_stats()
        self._sums, self._counts = {n: 0.0 for n in self.own}, {n: 0 for n in self.own}

    def epoch_means(self):
        out = dict(self.inner.epoch_means()) if hasattr(self.inner, 'epoch_means') else {}
        out.update({n: (self._sums[n] / self._counts[n] if self._counts[n] else float('nan')) for n in self.own})
        return out

    def epoch_mean(self):
        return self.epoch_means()[self.names[0]]

    def _log(self, name, v):
        if v == v:   # skip nan
            self._sums[name] += float(v)
            self._counts[name] += 1

    def forward(self, gt_pre, out, target):
        bnd = getattr(gt_pre, 'boundary', None)
        need = ('aff', 'prop', 'seed', 'contrast')
        if bnd is None or not all(k in bnd for k in need):
            raise ValueError(f'the propagation loss needs {need} from the propagation module in gt_pre.boundary; '
                             f'got {None if bnd is None else sorted(bnd)}')
        total = self.inner(gt_pre, out, target)
        s_logit = bnd['aff']
        masks = edge_masks(target, s_logit.shape[-2:], bnd['contrast'], self.hard_frac)
        aff, terms = affinity_terms(s_logit, masks)
        for k, v in terms.items():
            self._log(k, float(v.detach()))
        self._log('rank_acc', ranking_accuracy(s_logit, masks))
        total = total + self.w_aff * aff
        if self.w_comp > 0:
            comp = completion_term(bnd['prop'], masks['lab'], bnd['seed'])
            self._log('comp', float(comp.detach()))
            total = total + self.w_comp * comp
        return total


class WithExpertAux(nn.Module):
    '''EXP-15: inner(gt_pre, out, target) + 0.5 * detail aux + 0.3 * completion aux (see the module docstring). Reads
    gt_pre.boundary['lsa_d'] (B,2,h,w), ['lsa_c'] (B,1,h,w) and, in route mode, ['lsa_g'] {stage: (B,3,G,G)}.'''
    def __init__(self, inner, w_detail=0.5, w_completion=0.3, tau=0.15, ring_radius=3, gamma=2.0):
        super().__init__()
        self.inner = inner
        self.w_d, self.w_c, self.ring_radius = float(w_detail), float(w_completion), int(ring_radius)
        self.bd = BceDiceLoss(wb=1, wd=1)
        self.bd_ring = BceDiceLoss(wb=0.5, wd=1)
        self.ihr = InteriorHardPositiveRegion(tau, gamma)
        self.terms = ['aux_d_seg', 'aux_d_ring', 'aux_c_seg', 'aux_c_ihr']
        self.gate_names = [f'{g}_{r}_{s}' for s in ('dec3', 'dec4', 'dec5') for g in ('gC', 'gnull') for r in ('les', 'bg')]
        self.names = list(getattr(inner, 'names', [])) + self.terms + self.gate_names
        self.reset_stats()

    def reset_stats(self):
        if hasattr(self.inner, 'reset_stats'):
            self.inner.reset_stats()
        own = self.terms + self.gate_names
        self._sums, self._counts = {n: 0.0 for n in own}, {n: 0 for n in own}

    def epoch_means(self):
        out = dict(self.inner.epoch_means()) if hasattr(self.inner, 'epoch_means') else {}
        out.update({n: (self._sums[n] / self._counts[n] if self._counts[n] else float('nan')) for n in self._sums})
        return out

    def epoch_mean(self):
        return self.epoch_means()[self.names[0]]

    def _log(self, name, v):
        if v == v:
            self._sums[name] += float(v)
            self._counts[name] += 1

    def aux_terms(self, bnd, target):
        size = target.shape[-2:]
        up = lambda x: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        d = torch.sigmoid(up(bnd['lsa_d']))
        c = torch.sigmoid(up(bnd['lsa_c']))
        ring = boundary_ring(target, self.ring_radius)
        return {'aux_d_seg': self.bd(d[:, 0:1], target), 'aux_d_ring': self.bd_ring(d[:, 1:2], ring),
                'aux_c_seg': self.bd(c, target), 'aux_c_ihr': self.ihr(c, target)}

    def forward(self, gt_pre, out, target):
        bnd = getattr(gt_pre, 'boundary', None)
        if bnd is None or 'lsa_d' not in bnd or 'lsa_c' not in bnd:
            raise ValueError('the expert aux loss needs lsa_d / lsa_c from models.lsa_fusion in gt_pre.boundary; got '
                             f'{None if bnd is None else sorted(bnd)}')
        total = self.inner(gt_pre, out, target)
        t = self.aux_terms(bnd, target)
        for k, v in t.items():
            self._log(k, float(v.detach()))
        total = total + self.w_d * (t['aux_d_seg'] + t['aux_d_ring']) + self.w_c * (t['aux_c_seg'] + t['aux_c_ihr'])
        with torch.no_grad():
            for s, g in (bnd.get('lsa_g') or {}).items():
                les = F.adaptive_avg_pool2d((target >= 0.5).float(), g.shape[-2:]) >= 0.5
                for gi, gname in ((1, 'gC'), (2, 'gnull')):
                    gg = g[:, gi:gi + 1]
                    if les.any():
                        self._log(f'{gname}_les_{s}', float(gg[les].mean()))
                    if (~les).any():
                        self._log(f'{gname}_bg_{s}', float(gg[~les].mean()))
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


def describe(loss='bcedice', extra_term='none', extra_weight=None, boundary_weight=None,
             boundary_stages=None, boundary_radius=None, ring_radius=None, aff_weight=None, comp_weight=0.0,
             hard_frac=0.2, lsa_aux=None, lsa_tau=0.15, **kw):
    terms = parse_terms(extra_term, extra_weight)
    s = loss + ''.join(f' + {w:g} * {n}' for n, w in terms)
    names = [n for n, _ in terms]
    if 'snbl' in names:
        s += f' (tau {kw.get("snbl_tau", 3.0)})'
    if 'area' in names:
        s += f' (delta {kw.get("area_delta", 0.05)})'
    if 'hp' in names:
        s += f' (hp r={float(kw.get("hp_radius", 10.0)):g}px, gamma {float(kw.get("hp_gamma", 2.0)):g})'
    if 'ihp' in names:
        s += (f' (ihp depth>={float(kw.get("ihp_tau", 0.2)):g}, p<{float(kw.get("ihp_plow", 0.5)):g}, '
              f'gamma {float(kw.get("ihp_gamma", 2.0)):g})')
    if 'focal' in names:
        s += f' (focal alpha {float(kw.get("focal_alpha", 0.75)):g}, gamma {float(kw.get("focal_gamma", 2.0)):g})'
    if 'tversky' in names:
        s += f' (tversky alpha {float(kw.get("tversky_alpha", 0.3)):g}, beta {float(kw.get("tversky_beta", 0.7)):g})'
    if boundary_weight is not None:
        bw = float(boundary_weight)
        if ring_radius is not None:   # EXP-11 barrier head of the region completion (ASCII, see below)
            s += (f' + {bw:g} * ring (BceDice(0.5,1) on the +-{int(ring_radius)}px contour ring, barrier head)'
                  if bw != 0 else f' (+ ring +-{int(ring_radius)}px loss logged only, weight 0)')
        elif boundary_radius is not None:   # EXP-10 contour zone on the dec5 head (ASCII: the log line is compared as text)
            where = '/'.join(f'{w:g}*{st}' for st, w in (boundary_stages or {'dec5': 1.0}).items())
            s += (f' + {bw:g} * zone (BceDice(0.5,1) on the r={int(boundary_radius)}px contour zone, {where})'
                  if bw != 0 else f' (+ zone r={int(boundary_radius)}px loss logged only, weight 0)')
        else:
            s += (f' + {bw:g} * bnd (0.1*dec3 + 0.2*dec4 + 0.3*dec5, BceDice(0.5,1) on the contour band per head grid)'
                  if bw != 0 else ' (+ bnd contour-band loss logged only, weight 0)')
    if lsa_aux is not None:   # EXP-15 expert aux losses (ASCII: the log line is compared as text)
        s += (f' + expert aux {lsa_aux} (0.5*[BD(D seg)+BD(.5,1)(D ring +-3px)] + 0.3*[BD(C seg)+focal interior '
              f'depth>={float(lsa_tau):g} /|H|])')
    if aff_weight is not None:   # EXP-13 edge affinity of the propagation (ASCII: the log line is compared as text)
        s += (f' + {float(aff_weight):g} * aff (edge BCE same-label: 0.5 cut / 0.25 LL+BB / 0.25 hard'
              f' {int(round(100 * float(hard_frac)))}% contrast)')
        if float(comp_weight) > 0:
            s += f' + {float(comp_weight):g} * comp (BCE of propagated R vs GT, GT dilated 10 cells)'
    return s


def make_term(name, snbl_tau=3.0, area_delta=0.05, hp_radius=10.0, hp_gamma=2.0, **kw):
    if name == 'snbl':
        return ScaleNormBoundaryLoss(tau=snbl_tau)
    if name == 'area':
        return LogArea(delta=area_delta)
    if name == 'hp':
        return HardPositive(radius=hp_radius, gamma=hp_gamma)
    if name == 'ihp':
        return InteriorHardPositive(kw.get('ihp_tau', 0.2), kw.get('ihp_gamma', 2.0), kw.get('ihp_plow', 0.5))
    if name == 'focal':
        return FocalLoss(kw.get('focal_alpha', 0.75), kw.get('focal_gamma', 2.0))
    if name == 'tversky':
        return TverskyLoss(kw.get('tversky_alpha', 0.3), kw.get('tversky_beta', 0.7))
    return TERM_CLASSES[name]()


def build_criterion(loss='bcedice', extra_term='none', extra_weight=None, snbl_tau=3.0, area_delta=0.05,
                    boundary_weight=None, boundary_stages=None, boundary_radius=None, ring_radius=None,
                    hp_radius=10.0, hp_gamma=2.0, aff_weight=None, comp_weight=0.0, hard_frac=0.2, lsa_aux=None,
                    lsa_tau=0.15, **term_kw):
    '''The training criterion. Defaults return the original GT_BceDiceLoss(wb=1, wd=1); extra terms
    alone return EXP-6/7's WithExtraTerm; a boundary_weight (EXP-9, 0 allowed) wraps either in
    WithBoundaryAux. boundary_stages / boundary_radius (EXP-10) supervise the given heads with the
    contour zone of that radius instead of the 2-px band on dec3/4/5; ring_radius (EXP-11) with the
    +-r px contour ring.'''
    if loss not in LOSS_MODES:
        raise ValueError(f'unknown loss {loss!r}; choose from {LOSS_MODES}')
    unknown = set(term_kw) - {'ihp_tau', 'ihp_gamma', 'ihp_plow', 'focal_alpha', 'focal_gamma', 'tversky_alpha', 'tversky_beta'}
    if unknown:
        raise ValueError(f'unknown loss options {sorted(unknown)}')
    terms = parse_terms(extra_term, extra_weight)
    base = GT_BceDiceLoss(wb=1, wd=1) if loss == 'bcedice' else GT_BceRegionLoss(wb=1, wr=1)
    crit = base if not terms else WithExtraTerm(base, [(n, make_term(n, snbl_tau, area_delta, hp_radius, hp_gamma, **term_kw), w)
                                                        for n, w in terms])
    if lsa_aux is not None:   # EXP-15
        if boundary_weight is not None or aff_weight is not None:
            raise ValueError('the expert aux loss does not combine with a boundary or affinity loss')
        if lsa_aux not in ('probe', 'supervise'):
            raise ValueError(f'lsa_aux must be probe or supervise, got {lsa_aux!r}')
        return WithExpertAux(crit, 0.5, 0.3, lsa_tau, 3)
    if aff_weight is not None:   # EXP-13
        if boundary_weight is not None:
            raise ValueError('the propagation affinity loss does not combine with a boundary loss')
        if float(aff_weight) < 0 or float(comp_weight) < 0 or not 0 < float(hard_frac) < 1:
            raise ValueError(f'aff_weight / comp_weight must be >= 0 and hard_frac in (0, 1); got {aff_weight}, {comp_weight}, {hard_frac}')
        return WithPropagationAux(crit, aff_weight, comp_weight, hard_frac)
    if boundary_weight is None:
        return crit
    if float(boundary_weight) < 0:
        raise ValueError(f'boundary_weight must be >= 0, got {boundary_weight}')
    target_fn = None
    if ring_radius is not None:
        if boundary_radius is not None:
            raise ValueError('give either boundary_radius (EXP-10 zone) or ring_radius (EXP-11 ring), not both')
        target_fn = functools.partial(boundary_ring, radius=int(ring_radius))
    elif boundary_radius is not None:
        if float(boundary_radius) <= 0:
            raise ValueError(f'boundary_radius must be > 0, got {boundary_radius}')
        target_fn = functools.partial(boundary_zone, radius=float(boundary_radius))
    return WithBoundaryAux(crit, boundary_weight, boundary_stages, target_fn)
