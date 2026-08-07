"""
train_wno_burgers.py — Wavelet Neural Operator (WNO) on the 1D Burgers benchmark
(burgers_data_R10.mat), the WNO counterpart of train_sp2gno_burgers.py.

Static 1D operator map a(x) -> u(x, T=1) on a periodic domain. We reuse the exact
same train/val/test split (Jetson_data/burgers_split.json) and the same per-sample
relative-L2 metric as the Sp2GNO run, so the two operators are directly comparable.

Two experiment axes are exposed:
  --variant {small,base,large}  model-scale variants tuned to ~FNO param counts
                                (Table 6 of the Jetson edge paper): 72k / 235k / 820k.
  --res     {2048,4096,8192}    resolution-scaling: grid size (8192 // res = subsample).
                                The wavelet decomposition level scales with resolution
                                (8 @2048, 9 @4096, 10 @8192) to keep the number of
                                active wavelet modes — and hence the parameter count —
                                roughly constant across resolutions.

Usage:
  conda activate ~/GraphWNO
  python train_wno_burgers.py --variant base --res 2048 --ckpt wno_burgers_base_r2048.pth
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_codes'))
from wavelet_convolution import WaveConv1d                     # noqa: E402
from utils import LpLoss, count_params                          # noqa: E402
from sp2gno_core import set_all_seeds, setup_logging            # noqa: E402

# variant -> (width, layers); level is resolution-dependent (see level_for_res)
VARIANTS = {'small': (22, 4), 'base': (40, 4), 'large': (74, 4)}


def level_for_res(res):
    """Base level 8 @ res 2048; +1 per doubling so the # wavelet modes stays ~const."""
    return 8 + int(round(math.log2(res / 2048)))


class WNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super().__init__()
        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.padding = padding
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        self.fc0 = nn.Linear(in_channel, width)
        for _ in range(layers):
            self.conv.append(WaveConv1d(width, width, level, size, wavelet))
            self.w.append(nn.Conv1d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding])
        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.layers - 1:
                x = F.mish(x)
        if self.padding != 0:
            x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


@torch.no_grad()
def evaluate(model, loader, myloss, device):
    model.eval()
    err, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        b = x.shape[0]
        err += myloss(out.view(b, -1), y.view(b, -1)).item()
        n += b
    return err / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='base', choices=list(VARIANTS))
    ap.add_argument('--res', type=int, default=2048, help='grid size (8192 // res = subsample)')
    ap.add_argument('--data_dir', default='Jetson_data')
    ap.add_argument('--split', default='Jetson_data/burgers_split.json')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wavelet', default='db6')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--ckpt', default=None, help='checkpoint filename under checkpoints/')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    width, layers = VARIANTS[args.variant]
    level = level_for_res(args.res)
    tag = args.ckpt[:-4] if args.ckpt else f'wno_burgers_{args.variant}_r{args.res}'
    args.out_dir = args.out_dir or f'runs/wno_burgers_{args.variant}_r{args.res}'
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    ckpt_path = os.path.join('checkpoints', args.ckpt or f'{tag}.pth')

    logger = setup_logging(args.out_dir, name=tag)
    # also mirror the run log into logs/<tag>.log
    import logging
    fh = logging.FileHandler(os.path.join('logs', f'{tag}.log'), mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    logger.info(f"Args: {vars(args)} | width={width} layers={layers} level={level}")
    set_all_seeds(args.seed)

    # ---- data ----
    import scipy.io as sio
    sub = 8192 // args.res
    mat = sio.loadmat(os.path.join(args.data_dir, 'burgers_data_R10.mat'))
    a_all = mat['a'][:, ::sub].astype(np.float32)
    u_all = mat['u'][:, ::sub].astype(np.float32)
    s = a_all.shape[1]
    with open(args.split) as f:
        split = json.load(f)
    tr_i, va_i, te_i = split['train'], split['val'], split['test']
    logger.info(f"res(grid)={s} sub={sub} train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    def loader(idx, shuffle):
        x = torch.from_numpy(a_all[idx])[:, :, None]
        y = torch.from_numpy(u_all[idx])
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                           batch_size=args.batch_size, shuffle=shuffle)
    train_loader = loader(tr_i, True)
    val_loader = loader(va_i, False)
    test_loader = loader(te_i, False)

    # ---- model ----
    model = WNO1d(width=width, level=level, layers=layers, size=s, wavelet=args.wavelet,
                  in_channel=2, grid_range=1).to(args.device)
    logger.info(f"WNO1d params={count_params(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    myloss = LpLoss(size_average=False)

    best_val, curve, t0 = float('inf'), [], time.time()
    for ep in range(args.epochs):
        model.train()
        tr_l2, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            b = x.shape[0]
            opt.zero_grad()
            out = model(x)
            loss = myloss(out.view(b, -1), y.view(b, -1))
            loss.backward(); opt.step()
            tr_l2 += loss.item(); n += b
        sched.step()
        tr_l2 /= n

        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            val = evaluate(model, val_loader, myloss, args.device)
            te = evaluate(model, test_loader, myloss, args.device)
            improved = val < best_val
            if improved:
                best_val = val
                torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                            'val_rel_l2': val, 'test_rel_l2': te,
                            'config': {'variant': args.variant, 'res': s, 'width': width,
                                       'layers': layers, 'level': level}}, ckpt_path)
            curve.append({'epoch': ep + 1, 'train_rel_l2': tr_l2, 'val_rel_l2': val,
                          'test_rel_l2': te, 'seconds': time.time() - t0})
            with open(os.path.join(args.out_dir, 'curve.json'), 'w') as f:
                json.dump(curve, f, indent=2)
            logger.info(f"  [{tag}] ep {ep+1}/{args.epochs} train {tr_l2:.4f} val {val:.4f} "
                        f"test {te:.4f} (best_val {best_val:.4f}{' *' if improved else ''}, "
                        f"{time.time()-t0:.0f}s)")

    # final test from best-on-val checkpoint
    ck = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ck['model_state_dict'])
    final_test = evaluate(model, test_loader, myloss, args.device)
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump({'model': 'WNO', 'label': tag, 'variant': args.variant, 'res': s,
                   'params': count_params(model), 'final_test_rel_l2': final_test,
                   'best_val_rel_l2': best_val, 'epochs': args.epochs,
                   'ckpt': ckpt_path, 'best_epoch': ck['epoch'] + 1}, f, indent=2)
    logger.info(f"DONE {tag}: final test rel-L2 {final_test:.4f} (best val {best_val:.4f}) "
                f"-> {ckpt_path}")


if __name__ == '__main__':
    main()
