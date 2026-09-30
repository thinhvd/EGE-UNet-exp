'''
Shared helpers for the GHPA static-prior analysis scripts
(visualize_hpa.py, eval_per_image.py, compare_runs.py).

Background: in GHPA the gates conv_xy(BI(params_xy)), conv_zx(BI(params_zx)), conv_zy(BI(params_zy))
do not depend on the input image (the learnable tensors have batch dim 1), so after training they are
static per-channel spatial priors shared by every image. These helpers load a checkpoint, capture those
gates with forward hooks, compute spatial/center-bias statistics against the dataset lesion prior, and
provide per-image segmentation metrics for stratified comparisons between runs.
'''
import os
import sys
import re
import json
import glob

import matplotlib
matplotlib.use('Agg')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from models.egeunet import EGEUNet, Grouped_multi_axis_Hadamard_Product_Attention
from configs.config_setting import setting_config, build_transforms

EPS = 1e-8

# Okabe-Ito colorblind-safe palette, assigned to runs/groups in fixed order
CATEGORICAL = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']
SEQ_GATE = 'Blues'      # magnitude of a gate (|g|)
SEQ_PRIOR = 'Oranges'   # lesion prior
DIVERGING = 'RdBu_r'    # signed gate / P around 1


# ----------------------------------------------------------------------------- checkpoints / model

def resolve_checkpoint(path):
    '''
    Accept a checkpoint file or an experiment work_dir. For a work_dir, prefer the checkpoint named in
    test_results.json, else the best-epoch*-loss*.pth with the smallest loss, else best.pth, else latest.pth.
    '''
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f'checkpoint path not found: {path}')
    ckpt_dir = os.path.join(path, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = path
    results = os.path.join(path, 'test_results.json')
    if os.path.isfile(results):
        try:
            name = json.load(open(results)).get('checkpoint')
        except Exception:
            name = None
        if name and os.path.isfile(os.path.join(ckpt_dir, name)):
            return os.path.join(ckpt_dir, name)
    bests = []
    for f in glob.glob(os.path.join(ckpt_dir, 'best-epoch*-loss*.pth')):
        m = re.search(r'loss([0-9]+(?:\.[0-9]+)?)\.pth$', os.path.basename(f))
        if m:
            bests.append((float(m.group(1)), f))
    if bests:
        return sorted(bests)[0][1]
    for name in ('best.pth', 'latest.pth'):
        f = os.path.join(ckpt_dir, name)
        if os.path.isfile(f):
            return f
    raise FileNotFoundError(f'no checkpoint found under {path}')


def infer_work_dir(ckpt_path):
    d = os.path.dirname(os.path.abspath(ckpt_path))
    if os.path.basename(d) == 'checkpoints':
        return os.path.dirname(d)
    return d


def load_state_dict(path):
    '''Load either a bare state_dict (best-epoch*.pth) or a training dict (latest.pth). Returns (sd, meta).'''
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        meta = {k: obj[k] for k in ('epoch', 'min_loss', 'min_epoch', 'loss') if k in obj}
        return obj['model_state_dict'], meta
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        return obj, {}
    raise ValueError(f'unrecognized checkpoint format: {path}')


def infer_model_config(sd):
    '''Recover the EGEUNet constructor arguments from state_dict shapes/keys (any GHPA placement).'''
    c_list, ghpa_stages = [], []
    for i in range(1, 7):
        if f'encoder{i}.0.ldw.2.weight' in sd:      # GHPA stage (its last conv outputs the stage width)
            c_list.append(int(sd[f'encoder{i}.0.ldw.2.weight'].shape[0]))
            ghpa_stages.append(f'enc{i}')
        elif f'encoder{i}.0.weight' in sd:          # plain conv stage
            c_list.append(int(sd[f'encoder{i}.0.weight'].shape[0]))
        else:
            raise ValueError(f'cannot infer encoder{i} from state_dict keys')
    for i in range(1, 6):
        if f'decoder{i}.0.ldw.2.weight' in sd:
            ghpa_stages.append(f'dec{i}')
    return {
        'num_classes': int(sd['final.weight'].shape[0]),
        'input_channels': int(sd['encoder1.0.weight'].shape[1]),
        'c_list': c_list,
        'bridge': any(k.startswith('GAB1.') for k in sd),
        'gt_ds': any(k.startswith('gt_conv1.') for k in sd),
        # frozen_ones is forward-equivalent to learnable once the weights are loaded
        'hpa_mode': 'learnable' if any(k.endswith('params_xy') for k in sd) else 'none',
        'ghpa_stages': ghpa_stages or None,
        **_infer_fusion_config(sd),
        # EXP-10 dec5 boundary residual refinement: only the gated variant has a boundary head
        'refine_mode': ('gate' if any(k.startswith('refine.bnd_head.') for k in sd)
                        else 'plain' if any(k.startswith('refine.') for k in sd) else 'none'),
    }


def _infer_fusion_config(sd):
    '''EXP-4: recover the cross-stage fusion arguments (models/fusion.py) from the checkpoint keys.'''
    if not any(k.startswith('fusion.') for k in sd):
        return {'fusion_mode': 'none', 'fusion_stages': None, 'fusion_dim': 16}
    has_attn = any(k.startswith('fusion.attn.') for k in sd)
    has_sum = any(k.startswith('fusion.sum_w.') for k in sd)
    alpha = [k for k in sd if k.startswith('fusion.bg_alpha.')]
    if alpha:   # EXP-9 boundary-guided fusion: one alpha per source (5) per stage
        n_src = int(sd['fusion.sum_w.' + alpha[0].split('.')[-1]].numel())
        if sd[alpha[0]].numel() != n_src:
            raise ValueError(f'{alpha[0]} has {sd[alpha[0]].numel()} values, expected one per source ({n_src})')
        mode = 'bg_stage'
    else:
        mode = {(True, True): 'sum_attn', (True, False): 'csaa',
                (False, True): 'sum', (False, False): 'concat'}[(has_attn, has_sum)]
    stages = [f'dec{i}' for i in range(1, 6) if f'fusion.heads.dec{i}.weight' in sd]
    return {
        'fusion_mode': mode,
        'fusion_stages': stages,
        'fusion_dim': int(sd['fusion.proj.0.0.weight'].shape[0]),
    }


def build_model(sd, device='cpu', overrides=None, strict=True):
    cfg = infer_model_config(sd)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    model = EGEUNet(**cfg)
    result = model.load_state_dict(sd, strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        print(f'[warn] non-strict load: missing={result.missing_keys} unexpected={result.unexpected_keys}')
    model.eval().to(device)
    return model, cfg


def pick_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        print('[warn] cuda not available, falling back to cpu')
        return 'cpu'
    return name


# ----------------------------------------------------------------------------- GHPA gates

def find_ghpa_modules(model):
    '''[(short_name, full_name, module)] in network order: enc4, enc5, enc6, dec1, dec2, dec3.'''
    out = []
    for name, m in model.named_modules():
        if isinstance(m, Grouped_multi_axis_Hadamard_Product_Attention):
            short = name.replace('encoder', 'enc').replace('decoder', 'dec').replace('.0', '')
            out.append((short, name, m))
    return out


def capture_gates(model, input_size=256, device='cpu', in_channels=3):
    '''
    One forward pass of zeros through the model with hooks on every GHPA module; returns
    {short: {H, W, C, g_xy (C,H,W), g_zx (C,H), g_zy (C,W), p_xy (C,8,8), p_zx (C,8), p_zy (C,8)}}
    where C = dim_in//4 of that module. The gates are the actual tensors the network multiplies
    its features with (independent of the input), taken from the real forward (same align_corners etc.).
    '''
    gates, hooks = {}, []

    def in_hook(rec):
        def h(mod, inp):
            x = inp[0]
            rec['H'], rec['W'] = int(x.shape[2]), int(x.shape[3])
        return h

    def out_hook(rec, key):
        def h(mod, inp, out):
            rec[key] = out.detach().cpu().double().numpy()
        return h

    for short, _, m in find_ghpa_modules(model):
        if m.hpa_mode == 'none':
            continue
        rec = {}
        gates[short] = rec
        hooks.append(m.register_forward_pre_hook(in_hook(rec)))
        hooks.append(m.conv_xy.register_forward_hook(out_hook(rec, 'g_xy')))
        hooks.append(m.conv_zx.register_forward_hook(out_hook(rec, 'g_zx')))
        hooks.append(m.conv_zy.register_forward_hook(out_hook(rec, 'g_zy')))
        rec['p_xy'] = m.params_xy.detach().cpu().double().numpy()[0]      # (C,8,8)
        rec['p_zx'] = m.params_zx.detach().cpu().double().numpy()[0, 0]   # (C,8)
        rec['p_zy'] = m.params_zy.detach().cpu().double().numpy()[0, 0]   # (C,8)
    if not gates:
        return gates
    with torch.no_grad():
        model(torch.zeros(1, in_channels, input_size, input_size, device=device))
    for h in hooks:
        h.remove()
    for rec in gates.values():
        rec['g_xy'] = rec['g_xy'][0]   # (C,H,W)
        rec['g_zx'] = rec['g_zx'][0]   # (C,H)  per-(channel,row) profile, broadcast over width
        rec['g_zy'] = rec['g_zy'][0]   # (C,W)  per-(channel,column) profile, broadcast over height
        rec['C'] = int(rec['g_xy'].shape[0])
    return gates


class GateOverride(nn.Module):
    '''
    Wraps conv_xy / conv_zx / conv_zy of a trained GHPA to knock out the spatial structure of the gate
    post hoc (no retraining):
      'ones'         -> gate = 1 everywhere (no gating at all)
      'spatial_mean' -> gate replaced by its spatial mean per channel (keeps per-channel scaling only)
      'shuffle'      -> spatial positions permuted with a fixed seed (keeps the value distribution,
                        destroys the spatial layout)
    '''
    def __init__(self, inner, mode, spatial_dims, seed=0):
        super().__init__()
        if mode not in ('ones', 'spatial_mean', 'shuffle'):
            raise ValueError(mode)
        self.inner = inner
        self.mode = mode
        self.spatial_dims = tuple(spatial_dims)
        self.seed = seed

    def forward(self, x):
        g = self.inner(x)
        if self.mode == 'ones':
            return torch.ones_like(g)
        if self.mode == 'spatial_mean':
            return g.mean(dim=self.spatial_dims, keepdim=True).expand_as(g)
        lead = g.shape[:len(g.shape) - len(self.spatial_dims)]
        flat = g.reshape(*lead, -1)
        perm = torch.randperm(flat.shape[-1], generator=torch.Generator().manual_seed(self.seed)).to(g.device)
        return flat[..., perm].reshape(g.shape)


def apply_gate_override(model, mode, seed=0):
    n = 0
    for _, _, m in find_ghpa_modules(model):
        if m.hpa_mode == 'none':
            continue
        m.conv_xy = GateOverride(m.conv_xy, mode, (2, 3), seed)
        m.conv_zx = GateOverride(m.conv_zx, mode, (2,), seed)
        m.conv_zy = GateOverride(m.conv_zy, mode, (2,), seed)
        n += 1
    if n == 0:
        raise ValueError('model has no HPA gates to override (hpa_mode=none)')
    return n


# ----------------------------------------------------------------------------- data / lesion prior

def make_config(dataset='isic17', input_size=256):
    '''Reuse the training config so the evaluation pipeline (normalize/to-tensor/resize) is identical.'''
    setting_config.datasets = dataset
    setting_config.input_size_h = input_size
    setting_config.input_size_w = input_size
    build_transforms(setting_config)
    return setting_config


def lesion_prior(data_path, input_size=256, cache_path=None):
    '''
    Mean of the train masks at input_size x input_size, processed exactly like NPY_datasets + myResize
    (PIL 'L' / 255 -> tensor -> bilinear resize, antialias=False). Returns (prior (S,S) float64, n_masks).
    '''
    if cache_path and os.path.isfile(cache_path):
        arr = np.load(cache_path)
        return arr, -1
    masks_dir = os.path.join(data_path, 'train', 'masks')
    files = sorted(os.listdir(masks_dir))
    acc = torch.zeros(input_size, input_size, dtype=torch.float64)
    for f in files:
        msk = np.expand_dims(np.array(Image.open(os.path.join(masks_dir, f)).convert('L')), axis=2) / 255
        t = torch.tensor(msk).permute(2, 0, 1)
        t = TF.resize(t, [input_size, input_size], antialias=False)
        acc += t[0].double()
    prior = (acc / max(len(files), 1)).numpy()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, prior)
    return prior, len(files)


def downsample_prior(prior, hw):
    '''Block-mean the S x S prior down to a module's (H, W) (exact block means when S % H == 0).'''
    t = torch.tensor(prior, dtype=torch.float64)[None, None]
    return F.adaptive_avg_pool2d(t, tuple(hw))[0, 0].numpy()


# ----------------------------------------------------------------------------- geometry / stats

def radial_map(H, W):
    '''Normalized distance from the image center: 0 at center, 1 at the edge midpoints.'''
    yy = (np.arange(H) - (H - 1) / 2) / max((H - 1) / 2, 1e-8)
    xx = (np.arange(W) - (W - 1) / 2) / max((W - 1) / 2, 1e-8)
    return np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)


def radial_1d(L):
    return np.abs(np.arange(L) - (L - 1) / 2) / max((L - 1) / 2, 1e-8)


def center_box_mask(H, W, area_frac=0.5):
    '''Centered square covering ~area_frac of the map (True inside).'''
    sh = max(1, int(round(H * np.sqrt(area_frac))))
    sw = max(1, int(round(W * np.sqrt(area_frac))))
    m = np.zeros((H, W), dtype=bool)
    y0, x0 = (H - sh) // 2, (W - sw) // 2
    m[y0:y0 + sh, x0:x0 + sw] = True
    return m


def center_span_mask(L, frac=0.5):
    m = np.zeros(L, dtype=bool)
    n = max(1, int(round(L * frac)))
    s = (L - n) // 2
    m[s:s + n] = True
    return m


def interior(a):
    '''Strip the 1-px border (where the zero-padded depthwise conv of a constant map differs).'''
    if a.ndim == 1:
        return a[1:-1] if a.shape[0] > 2 else a
    if a.shape[-1] > 2 and a.shape[-2] > 2:
        return a[..., 1:-1, 1:-1]
    return a


def cv(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return float(a.std() / (abs(a.mean()) + EPS))


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def center_ratio(a, box):
    inside, outside = a[box], a[~box]
    if inside.size == 0 or outside.size == 0:
        return float('nan')
    return float(inside.mean() / (outside.mean() + EPS))


def _stats_map(g_signed, prior=None):
    '''Stats of one 2D map (H,W): a = |g| (spatial emphasis), plus signed correlation with the prior.'''
    a = np.abs(g_signed)
    H, W = a.shape
    ai = interior(a)
    rad, radi = radial_map(H, W), radial_map(*ai.shape)
    d = {
        'mean_abs': float(a.mean()),
        'frac_negative': float((g_signed < 0).mean()),
        'cv_full': cv(a),
        'cv_interior': cv(ai),
        'r_radial_full': pearson(a, rad),
        'r_radial_interior': pearson(ai, radi),
        'center_ratio_full': center_ratio(a, center_box_mask(H, W)),
        'center_ratio_interior': center_ratio(ai, center_box_mask(*ai.shape)),
        'r_prior_full': float('nan'), 'r_prior_interior': float('nan'), 'r_prior_signed': float('nan'),
    }
    if prior is not None:
        d['r_prior_full'] = pearson(a, prior)
        d['r_prior_interior'] = pearson(ai, interior(prior))
        d['r_prior_signed'] = pearson(g_signed, prior)
    return d


def _stats_profile(g_signed, marginal=None):
    '''Same as _stats_map for a 1D per-channel profile (length L).'''
    a = np.abs(g_signed)
    L = a.shape[0]
    ai = interior(a)
    d = {
        'mean_abs': float(a.mean()),
        'frac_negative': float((g_signed < 0).mean()),
        'cv_full': cv(a),
        'cv_interior': cv(ai),
        'r_radial_full': pearson(a, radial_1d(L)),
        'r_radial_interior': pearson(ai, radial_1d(ai.shape[0])),
        'center_ratio_full': center_ratio(a, center_span_mask(L)),
        'center_ratio_interior': center_ratio(ai, center_span_mask(ai.shape[0])),
        'r_prior_full': float('nan'), 'r_prior_interior': float('nan'), 'r_prior_signed': float('nan'),
    }
    if marginal is not None:
        d['r_prior_full'] = pearson(a, marginal)
        d['r_prior_interior'] = pearson(ai, interior(marginal))
        d['r_prior_signed'] = pearson(g_signed, marginal)
    return d


STAT_KEYS = ['mean_abs', 'frac_negative', 'cv_full', 'cv_interior', 'r_radial_full', 'r_radial_interior',
             'center_ratio_full', 'center_ratio_interior', 'r_prior_full', 'r_prior_interior', 'r_prior_signed']


def gate_stats(g, p, prior=None):
    '''
    g: (C,H,W) or (C,L) gate; p: raw learnable tensor (C,8,8) or (C,8); prior: (H,W) map or (L,) marginal.
    Returns {'per_channel': [ {channel, ...STAT_KEYS, p_mean_abs_dev_from_one, p_cv} ],
             'meanmap': stats of the channel-mean |g| map (and signed mean map for r_prior_signed),
             'summary': mean/median over channels of the main stats}.
    '''
    fn = _stats_map if g.ndim == 3 else _stats_profile
    per_channel = []
    for c in range(g.shape[0]):
        d = fn(g[c], prior)
        d['channel'] = c
        d['p_mean_abs_dev_from_one'] = float(np.abs(p[c] - 1.0).mean())
        d['p_cv'] = cv(p[c])
        per_channel.append(d)
    mean_abs_map = np.abs(g).mean(axis=0)
    meanmap = fn(mean_abs_map, prior)
    if prior is not None:
        meanmap['r_prior_signed'] = pearson(g.mean(axis=0), prior)
    summary = {}
    for k in ('cv_full', 'cv_interior', 'r_radial_interior', 'center_ratio_interior', 'r_prior_interior',
              'r_prior_signed', 'mean_abs', 'p_mean_abs_dev_from_one'):
        vals = np.array([d[k] for d in per_channel], dtype=np.float64)
        summary[k + '_mean'] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
        summary[k + '_median'] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float('nan')
    return {'per_channel': per_channel, 'meanmap': meanmap, 'summary': summary}


# ----------------------------------------------------------------------------- segmentation metrics

def binary_counts(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return tp, fp, fn, tn


def dsc_from_counts(tp, fp, fn):
    den = 2 * tp + fp + fn
    return float(2 * tp / den) if den > 0 else 1.0


def iou_from_counts(tp, fp, fn):
    den = tp + fp + fn
    return float(tp / den) if den > 0 else 1.0


def pooled_metrics(tp, fp, fn, tn):
    '''Identical formulas to engine.val_one_epoch / test_one_epoch (dataset-level pooled pixels).'''
    total = tp + fp + fn + tn
    return {
        'accuracy': float(tn + tp) / float(total) if total != 0 else 0,
        'sensitivity': float(tp) / float(tp + fn) if float(tp + fn) != 0 else 0,
        'specificity': float(tn) / float(tn + fp) if float(tn + fp) != 0 else 0,
        'f1_or_dsc': float(2 * tp) / float(2 * tp + fp + fn) if float(2 * tp + fp + fn) != 0 else 0,
        'miou': float(tp) / float(tp + fp + fn) if float(tp + fp + fn) != 0 else 0,
    }


def mask_descriptors(gt):
    '''
    Position/size descriptors of a binary GT mask (H,W):
      area_frac, cy, cx (pixel centroid), centroid_offset (distance of the centroid from the image
      center normalized by the half-size: 0 = center, 1 = edge midpoint, ~1.41 = corner),
      touches_border, bbox (y0,x0,y1,x1).
    '''
    gt = gt.astype(bool)
    H, W = gt.shape
    area = int(gt.sum())
    d = {'area_frac': area / float(H * W), 'cy': float('nan'), 'cx': float('nan'),
         'centroid_offset': float('nan'), 'touches_border': 0,
         'bbox_y0': -1, 'bbox_x0': -1, 'bbox_y1': -1, 'bbox_x1': -1}
    if area == 0:
        return d
    ys, xs = np.nonzero(gt)
    cy, cx = float(ys.mean()), float(xs.mean())
    d.update({
        'cy': cy, 'cx': cx,
        'centroid_offset': float(np.sqrt(((cy - (H - 1) / 2) / ((H - 1) / 2)) ** 2 +
                                         ((cx - (W - 1) / 2) / ((W - 1) / 2)) ** 2)),
        'touches_border': int(ys.min() == 0 or ys.max() == H - 1 or xs.min() == 0 or xs.max() == W - 1),
        'bbox_y0': int(ys.min()), 'bbox_x0': int(xs.min()), 'bbox_y1': int(ys.max()), 'bbox_x1': int(xs.max()),
    })
    return d


def shift_tensor(x, dx, dy):
    '''Translate a (C,H,W) tensor: content moves right by dx and down by dy; uncovered area is reflected.'''
    if dx == 0 and dy == 0:
        return x
    _, H, W = x.shape
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))  # left, right, top, bottom
    xp = F.pad(x.unsqueeze(0), pad, mode='reflect')[0]
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return xp[:, y0:y0 + H, x0:x0 + W]


def fmt(v, nd=3):
    if v is None:
        return 'nan'
    try:
        if isinstance(v, float) and not np.isfinite(v):
            return 'nan'
        return f'{v:.{nd}f}'
    except (TypeError, ValueError):
        return str(v)
