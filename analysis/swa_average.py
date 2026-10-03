'''
EXP-15: stochastic weight averaging (SWA) of saved checkpoints, after training.

The author's CosineAnnealingLR(T_max=50) is periodic (the LR climbs back every 100 epochs), and the mask size of one
run oscillates strongly from epoch to epoch (EXP-14 doc, section 6b). The uniform average of the checkpoints saved at
epochs 200, 210, ..., 300 spans one full LR period. EGE-UNet uses only GroupNorm / LayerNorm, so the averaged weights
need no batch-norm recalibration. Floating-point tensors are averaged; integer buffers (provenance ids, step counts)
must agree and are copied.

  python analysis/swa_average.py --run results/.../egeunet_isic17_learnable_s42            # epochs 200..300
  -> <run>/checkpoints/swa-epoch200-300.pth (a bare state_dict, loadable by analysis.common)
'''
import os
import sys
import glob
import argparse

import torch


def average_state_dicts(sds):
    keys = list(sds[0])
    for sd in sds[1:]:
        if list(sd) != keys:
            raise ValueError('checkpoints have different keys')
    out = {}
    for k in keys:
        v0 = sds[0][k]
        if v0.is_floating_point():
            out[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
        else:
            if any(not torch.equal(sd[k], v0) for sd in sds[1:]):
                raise ValueError(f'integer buffer {k} differs between checkpoints')
            out[k] = v0.clone()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--first', type=int, default=200)
    p.add_argument('--last', type=int, default=300)
    p.add_argument('--every', type=int, default=10)
    a = p.parse_args()
    paths = [os.path.join(a.run, 'checkpoints', f'epoch{e:03d}.pth') for e in range(a.first, a.last + 1, a.every)]
    missing = [x for x in paths if not os.path.isfile(x)]
    if missing:
        sys.exit(f'missing checkpoints: {missing}')
    sds = [torch.load(x, map_location='cpu', weights_only=False) for x in paths]
    sds = [sd['model_state_dict'] if 'model_state_dict' in sd else sd for sd in sds]
    out = os.path.join(a.run, 'checkpoints', f'swa-epoch{a.first}-{a.last}.pth')
    torch.save(average_state_dicts(sds), out)
    print(f'averaged {len(sds)} checkpoints -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
