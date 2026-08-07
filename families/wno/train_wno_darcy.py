"""
train_wno_darcy.py — Wavelet Neural Operator (WNO) on the 2D Darcy benchmark
(piececonst_r421_N1024_smooth1/2.mat), the WNO counterpart of train_sp2gno_darcy.py.

Operator map a(x,y) -> u(x,y). Same data protocol as the Sp2GNO run:
  train = smooth1[:900], val = smooth1[900:1000], test = smooth2[:200],
per-sample relative-L2 metric, so the two operators are directly comparable. As in
the reference WNO recipe the targets are unit-Gaussian normalised during training.

Two experiment axes are exposed:
  --variant {small,base,large}  model-scale variants tuned to ~FNO param counts
                                (Table 6 of the Jetson edge paper): 72k / 235k / 820k.
  --res     {141,211,281,421}   resolution-scaling: target grid size, selected from
                                the native 421-grid by evenly spaced indices
                                (141=stride3, 211=stride2, 421=full, 281≈stride1.5).
                                The wavelet level scales with resolution
                                ({141:5,211:6,281:6,421:7}) so the number of active
                                wavelet modes — hence the parameter count — stays
                                roughly constant across resolutions.

Usage:
  conda activate ~/GraphWNO
  python train_wno_darcy.py --variant base --res 141 --ckpt wno_darcy_base_r141.pth
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_codes'))
from wavelet_convolution import WaveConv2d                          # noqa: E402
from utils import LpLoss, count_params, UnitGaussianNormalizer      # noqa: E402
from sp2gno_core import set_all_seeds, setup_logging                # noqa: E402

VARIANTS = {'small': (5, 4), 'base': (8, 4), 'large': (15, 4)}      # (width, layers)
LEVEL_FOR_RES = {141: 5, 211: 6, 281: 6, 421: 7}


class WNO2d(nn.Module):
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
            self.conv.append(WaveConv2d(width, width, level, size, wavelet))
            self.w.append(nn.Conv2d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.layers - 1:
                x = F.mish(x)
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def darcy_indices(res):
    """Evenly-spaced indices into the native 421-grid giving `res` points."""
    return np.round(np.linspace(0, 420, res)).astype(int)


def load_darcy(path, n, idx):
    import scipy.io as sio
    d = sio.loadmat(path)
    coeff = d['coeff'][:n][:, idx][:, :, idx].astype(np.float32)     # (n,s,s)
    sol = d['sol'][:n][:, idx][:, :, idx].astype(np.float32)
    return coeff, sol


@torch.no_grad()
def evaluate(model, loader, myloss, y_norm, s, device):
    model.eval()
    err, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        b = x.shape[0]
        out = model(x).reshape(b, s, s)
        out = y_norm.decode(out)
        err += myloss(out.view(b, -1), y.view(b, -1)).item()
        n += b
    return err / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='base', choices=list(VARIANTS))
    ap.add_argument('--res', type=int, default=141, help='target grid size from the 421-grid')
    ap.add_argument('--data_dir', default='Jetson_data')
    ap.add_argument('--ntrain', type=int, default=900)
    ap.add_argument('--nval', type=int, default=100)
    ap.add_argument('--ntest', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wavelet', default='db6')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    width, layers = VARIANTS[args.variant]
    level = LEVEL_FOR_RES.get(args.res, 5)
    tag = args.ckpt[:-4] if args.ckpt else f'wno_darcy_{args.variant}_r{args.res}'
    args.out_dir = args.out_dir or f'runs/wno_darcy_{args.variant}_r{args.res}'
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    ckpt_path = os.path.join('checkpoints', args.ckpt or f'{tag}.pth')

    logger = setup_logging(args.out_dir, name=tag)
    import logging
    fh = logging.FileHandler(os.path.join('logs', f'{tag}.log'), mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    logger.info(f"Args: {vars(args)} | width={width} layers={layers} level={level}")
    set_all_seeds(args.seed)

    # ---- data ----
    idx = darcy_indices(args.res)
    s = len(idx)
    x_train, y_train = load_darcy(os.path.join(args.data_dir, 'piececonst_r421_N1024_smooth1.mat'),
                                  args.ntrain + args.nval, idx)
    x_test, y_test = load_darcy(os.path.join(args.data_dir, 'piececonst_r421_N1024_smooth2.mat'),
                                args.ntest, idx)
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
    x_val, y_val = x_train[args.ntrain:], y_train[args.ntrain:]
    x_train, y_train = x_train[:args.ntrain], y_train[:args.ntrain]
    logger.info(f"res(grid)={s} train={args.ntrain} val={args.nval} test={args.ntest}")

    x_norm = UnitGaussianNormalizer(x_train)
    x_train = x_norm.encode(x_train); x_val = x_norm.encode(x_val); x_test = x_norm.encode(x_test)
    y_norm = UnitGaussianNormalizer(y_train)
    y_train = y_norm.encode(y_train)              # train targets normalised; val/test raw
    y_norm.to(args.device)

    def loader(x, y, shuffle):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x.reshape(x.shape[0], s, s, 1), y),
            batch_size=args.batch_size, shuffle=shuffle)
    train_loader = loader(x_train, y_train, True)
    val_loader = loader(x_val, y_val, False)
    test_loader = loader(x_test, y_test, False)

    # ---- model ----
    model = WNO2d(width=width, level=level, layers=layers, size=[s, s], wavelet=args.wavelet,
                  in_channel=3, grid_range=[1, 1], padding=1).to(args.device)
    logger.info(f"WNO2d params={count_params(model):,}")

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
            out = model(x).reshape(b, s, s)
            out = y_norm.decode(out)
            y_dec = y_norm.decode(y)
            loss = myloss(out.view(b, -1), y_dec.view(b, -1))
            loss.backward(); opt.step()
            tr_l2 += loss.item(); n += b
        sched.step()
        tr_l2 /= n

        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            val = evaluate(model, val_loader, myloss, y_norm, s, args.device)
            te = evaluate(model, test_loader, myloss, y_norm, s, args.device)
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

    ck = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ck['model_state_dict'])
    final_test = evaluate(model, test_loader, myloss, y_norm, s, args.device)
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump({'model': 'WNO', 'label': tag, 'variant': args.variant, 'res': s,
                   'params': count_params(model), 'final_test_rel_l2': final_test,
                   'best_val_rel_l2': best_val, 'epochs': args.epochs,
                   'ckpt': ckpt_path, 'best_epoch': ck['epoch'] + 1}, f, indent=2)
    logger.info(f"DONE {tag}: final test rel-L2 {final_test:.4f} (best val {best_val:.4f}) "
                f"-> {ckpt_path}")


if __name__ == '__main__':
    main()
