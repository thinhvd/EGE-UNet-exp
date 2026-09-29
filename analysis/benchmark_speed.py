'''
Cost benchmark for EGE-UNet variants: parameters, multiply-accumulate operations and measured latency.

Why measured latency and not only GFLOPs: a module can be cheap in MACs and still slow, because
attention adds reshapes, softmaxes and small matrix multiplies that the FLOP count ignores. EXP-4
compares variants that differ by a few thousand parameters, so the honest comparison needs the wall
clock as well as the analytic cost.

What the MAC count covers: every Conv1d/Conv2d/Linear, counted with forward hooks from the real
shapes seen in a forward pass, plus an analytic term for the cross-stage attention's einsums (which
are not modules and so have no hook). Not counted: normalizations, GELU, interpolation, softmax,
elementwise multiplies. Those are identical in kind across all variants, so the differences reported
here are fair; the absolute number is a lower bound.

Sanity anchor and a units warning: the baseline measures 0.0721 GMACs, i.e. 0.1442 GFLOPs counting a
multiply-add as two operations. The EGE-UNet paper reports "0.072 GFLOPs", which is this same number
in the MACs convention that thop/fvcore print (they label MACs as FLOPs). So the column comparable
with the paper and with other papers' tables is GMACs, not GFLOPs; both are reported here to keep
that unambiguous.

Usage:
    # from a trained run (work_dir or .pth) - the variant is inferred from the checkpoint
    python analysis/benchmark_speed.py --checkpoint results/<run> --device cuda --label learnable

    # from flags, no weights needed (speed does not depend on the values)
    python analysis/benchmark_speed.py --fusion csaa --fusion-stages all5 --device cpu

    # several runs into one json
    python analysis/benchmark_speed.py --checkpoint results/<run_a> --checkpoint results/<run_b> \
        --device cuda --out results/benchmark_gpu.json
'''
import os
import sys
import json
import time
import argparse
import platform

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn

from models.egeunet import EGEUNet
from models.fusion import CrossStageAxialAttention, FUSION_STAGE_SETS
from analysis.common import resolve_checkpoint, load_state_dict, infer_model_config, pick_device


# ----------------------------------------------------------------------------- MACs

def count_macs(model, input_size=256, device='cpu'):
    '''MACs for one image. Hooks on conv/linear modules + an analytic term for the CSAA einsums.'''
    total = {'hooked': 0, 'attention': 0}
    hooks = []

    def conv_hook(mod, inp, out):
        # out: (B, C_out, *spatial); each output element costs C_in/groups * prod(kernel) MACs
        k = int(np.prod(mod.kernel_size))
        c_in = mod.in_channels // mod.groups
        n_out = int(np.prod(out.shape[1:]))          # per image
        total['hooked'] += k * c_in * n_out

    def linear_hook(mod, inp, out):
        total['hooked'] += mod.in_features * int(np.prod(out.shape[1:]))

    def attn_hook(mod, inp, out):
        # two axial steps; per step: scores (S*T tokens x S*T keys x dim) + weighted sum, same size
        x = inp[0]
        s, g = x.shape[1], mod.grid
        dim = mod.fdim * g
        per_step = 2 * (s * g) * (s * g) * dim       # q@k^T and att@v
        total['attention'] += 2 * per_step

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, CrossStageAxialAttention):
            hooks.append(m.register_forward_hook(attn_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size, device=device))
    for h in hooks:
        h.remove()
    total['total'] = total['hooked'] + total['attention']
    return total


# ----------------------------------------------------------------------------- latency

def measure_latency(model, batch_size, input_size=256, device='cpu', warmup=20, iters=100):
    model.eval()
    x = torch.zeros(batch_size, 3, input_size, input_size, device=device)
    cuda = device.startswith('cuda')
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if cuda:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))          # ms
            else:
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1e3)
    t = np.array(times)
    med = float(np.median(t))
    return {
        'batch_size': batch_size,
        'median_ms': med,
        'p10_ms': float(np.percentile(t, 10)),
        'p90_ms': float(np.percentile(t, 90)),
        'images_per_s': batch_size / (med / 1e3),
    }


# ----------------------------------------------------------------------------- model construction

def model_from_checkpoint(path, device):
    ckpt = resolve_checkpoint(path)
    sd, _ = load_state_dict(ckpt)
    cfg = infer_model_config(sd)
    model = EGEUNet(**cfg)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), cfg, ckpt


def model_from_flags(args, device):
    cfg = {
        'hpa_mode': args.hpa_mode,
        'fusion_mode': args.fusion,
        'fusion_stages': FUSION_STAGE_SETS[args.fusion_stages] if args.fusion_stages else None,
        'fusion_dim': args.fusion_dim,
    }
    model = EGEUNet(**cfg)
    return model.to(device).eval(), cfg, None


def describe(cfg):
    fusion = cfg.get('fusion_mode', 'none')
    if fusion == 'none':
        return f'hpa={cfg.get("hpa_mode", "learnable")} fusion=none'
    stages = cfg.get('fusion_stages') or []
    return (f'hpa={cfg.get("hpa_mode", "learnable")} fusion={fusion} '
            f'stages={len(stages)}({",".join(stages)}) dim={cfg.get("fusion_dim", 16)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', action='append', default=[],
                    help='work_dir or .pth to benchmark; repeatable. Without it the model is built from flags')
    ap.add_argument('--label', action='append', default=[],
                    help='name for the corresponding --checkpoint (defaults to the folder name)')
    ap.add_argument('--hpa-mode', type=str, default='learnable', choices=['learnable', 'frozen_ones', 'none'])
    ap.add_argument('--fusion', type=str, default='none',
                    choices=['none', 'sum', 'concat', 'csaa', 'sum_attn', 'bg_stage'])
    ap.add_argument('--fusion-stages', type=str, default=None, choices=['deep3', 'all5', 'shallow3'])
    ap.add_argument('--fusion-dim', type=int, default=16)
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--batch-sizes', type=str, default='1,8')
    ap.add_argument('--input-size', type=int, default=256)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--threads', type=int, default=4, help='torch CPU threads (recorded in the json)')
    ap.add_argument('--out', type=str, default=None, help='json file to write (appends to an existing one)')
    args = ap.parse_args()

    device = pick_device(args.device)
    if device == 'cpu':
        torch.set_num_threads(args.threads)
    batch_sizes = [int(b) for b in args.batch_sizes.split(',') if b.strip()]

    env = {
        'device': device,
        'torch': torch.__version__,
        'platform': platform.platform(),
        'cpu_threads': torch.get_num_threads(),
        'input_size': args.input_size,
        'warmup': args.warmup,
        'iters': args.iters,
    }
    if device.startswith('cuda'):
        env['gpu'] = torch.cuda.get_device_name(0)

    targets = []
    if args.checkpoint:
        for i, ck in enumerate(args.checkpoint):
            label = args.label[i] if i < len(args.label) else os.path.basename(os.path.normpath(ck))
            targets.append((label, ck))
    else:
        label = args.label[0] if args.label else (
            f'{args.hpa_mode}' if args.fusion == 'none'
            else f'fuse-{args.fusion}-{args.fusion_stages or "deep3"}-d{args.fusion_dim}')
        targets.append((label, None))

    rows = []
    for label, ck in targets:
        if ck is None:
            model, cfg, ckpt_path = model_from_flags(args, device)
        else:
            model, cfg, ckpt_path = model_from_checkpoint(ck, device)
        macs = count_macs(model, input_size=args.input_size, device=device)
        row = {
            'label': label,
            'checkpoint': ckpt_path,
            'config': describe(cfg),
            'params_total': sum(p.numel() for p in model.parameters()),
            'params_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'gmacs': macs['total'] / 1e9,
            'gmacs_conv': macs['hooked'] / 1e9,
            'gmacs_attention': macs['attention'] / 1e9,
            'gflops': 2 * macs['total'] / 1e9,
            'latency': [measure_latency(model, b, args.input_size, device, args.warmup, args.iters)
                        for b in batch_sizes],
        }
        rows.append(row)
        lat = ' | '.join(f'bs{l["batch_size"]}: {l["median_ms"]:.2f} ms ({l["images_per_s"]:.1f} img/s)'
                         for l in row['latency'])
        print(f'{label:28} {row["params_total"]:7d} params  {row["gmacs"]:.4f} GMACs '
              f'({row["gflops"]:.4f} GFLOPs)  {lat}')
        del model

    if args.out:
        payload = {'env': env, 'runs': rows}
        if os.path.isfile(args.out):
            try:
                old = json.load(open(args.out))
                if old.get('env') == env:
                    keep = [r for r in old.get('runs', []) if r['label'] not in {x['label'] for x in rows}]
                    payload['runs'] = keep + rows
            except Exception:
                pass
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
