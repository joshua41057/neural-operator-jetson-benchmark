"""
train_sp2gno_burgers.py — Sp2GNO on the 1D Burgers benchmark (burgers_data_R10).

The provided burgers_data_R10.mat is a *static* operator map: initial condition
a(x) (2048 x 8192) -> solution u(x, T=1) (2048 x 8192) on a periodic 1D domain
(there is no intermediate time axis). We therefore treat it as a 1D
operator-learning problem: nodes lie on a line (x-coordinates), and a single
shared Sp2GNO graph (knn over x, inverse-distance edge weights, Lipschitz
embeddings, Laplacian eigenpairs) maps a(x) -> u(x,1).

Spatial subsample sub (default 8 -> s=1024 nodes). Train/val/test indices come
from Jetson_data/burgers_split.json.

Node features per point: the x-coordinate (1) plus the (z-scored) initial
condition a(x) (1) -> in_dim = 2. (Positional information is otherwise carried by
the graph-Laplacian eigenvectors inside the spectral layers.) Targets are the raw
solution; loss is per-sample relative L2.

Usage:
  conda activate ~/GraphWNO
  python train_sp2gno_burgers.py             # train 500 epochs from scratch
  python train_sp2gno_burgers.py --resume    # resume from the last checkpoint
"""

import argparse
import functools
import json
import os

import numpy as np
import scipy.io as sio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sp2gno_core import (set_all_seeds, setup_logging, build_graph_features,
                         Sp2GNO, SharedGraph, train_shared)

# model-scale variants (fixed n_layers=6, num_freq=64; width is the scale knob),
# tuned to ~FNO param counts from Table 6 of the Jetson edge paper: 72k/235k/820k.
VARIANTS = {'small': 13, 'base': 24, 'large': 45}


def build_features(a, u, coord_feat, mean, std):
    """a,u (M,s) -> feats (M,s,2) = [x, a_norm], Y (M,s,1)."""
    M, s = a.shape
    an = ((a - mean) / std)[..., None]                            # (M,s,1)
    feats = torch.cat([coord_feat.unsqueeze(0).expand(M, -1, -1),
                       torch.from_numpy(an)], dim=-1)
    Y = torch.from_numpy(u.reshape(M, s, 1))
    return feats.contiguous(), Y.contiguous()


def plot_burgers(model, graph, epoch, plot_dir, *, test_feats, test_Y, xcoord, tag=''):
    model.eval()
    idx = [0, 1, 2, 3]
    with torch.no_grad():
        feats, U, ei, ew, lips, Y = graph.batch(test_feats, test_Y, idx)
        pred = model(feats, U, ei, ew, lips).cpu().numpy()        # (4,s,1)
    fig, ax = plt.subplots(2, 4, figsize=(24, 9))
    for c, i in enumerate(idx):
        truth = test_Y[i, :, 0].numpy()
        pr = pred[c, :, 0]
        ax[0, c].plot(xcoord, truth, color='k', label='Ground Truth')
        ax[0, c].plot(xcoord, pr, '--', color='r', label='Prediction')
        ax[0, c].legend(fontsize=11); ax[0, c].set_title(f'test sample #{i}', fontsize=13)
        ax[1, c].plot(xcoord, (truth - pr) ** 2, color='b')
        ax[1, c].set_xlabel('x')
    ax[0, 0].set_ylabel('u(x, T=1)', fontsize=14, fontweight='bold')
    ax[1, 0].set_ylabel('Squared Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    name = tag if tag else f'ep{epoch}'
    plt.savefig(os.path.join(plot_dir, f'burgers_pred_{name}.png'), dpi=100)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='base', choices=list(VARIANTS))
    ap.add_argument('--res', type=int, default=2048, help='grid size (s=8192//res subsample)')
    ap.add_argument('--ckpt', default=None, help='checkpoint filename under checkpoints/')
    ap.add_argument('--data_dir', default='Jetson_data')
    ap.add_argument('--split', default='Jetson_data/burgers_split.json')
    ap.add_argument('--sub', type=int, default=None, help='spatial subsample (default 8192//res)')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=20)
    ap.add_argument('--width', type=int, default=None, help='override variant width')
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--num_freq', type=int, default=64)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.sub is None:
        args.sub = 8192 // args.res
    if args.width is None:
        args.width = VARIANTS[args.variant]
    tag = args.ckpt[:-4] if args.ckpt else f'sp2gno_burgers_{args.variant}_s{args.res}'
    args.out_dir = args.out_dir or f'runs/sp2gno_burgers_{args.variant}_s{args.res}'

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True); os.makedirs('logs', exist_ok=True)
    plot_dir = os.path.join(args.out_dir, 'plots'); os.makedirs(plot_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'ckpt'); os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logging(args.out_dir, name=tag)
    import logging
    fh = logging.FileHandler(os.path.join('logs', f'{tag}.log'), mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    logger.info(f"Args: {vars(args)} | variant={args.variant} width={args.width}")
    set_all_seeds(args.seed)
    ckpt_best_path = os.path.join('checkpoints', args.ckpt or f'{tag}.pth')

    # ---- data ----
    mat = sio.loadmat(os.path.join(args.data_dir, 'burgers_data_R10.mat'))
    a_all = mat['a'][:, ::args.sub].astype(np.float32)            # (2048,s)
    u_all = mat['u'][:, ::args.sub].astype(np.float32)
    s = a_all.shape[1]; N = s
    with open(args.split) as f:
        split = json.load(f)
    tr_i, va_i, te_i = split['train'], split['val'], split['test']
    logger.info(f"s={s} N={N}  train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    tr_a, tr_u = a_all[tr_i], u_all[tr_i]
    va_a, va_u = a_all[va_i], u_all[va_i]
    te_a, te_u = a_all[te_i], u_all[te_i]
    mean, std = float(tr_a.mean()), float(tr_a.std())
    logger.info(f"a normalization mean={mean:.4f} std={std:.4f}")

    xcoord = np.linspace(0, 1, s, dtype=np.float32)
    pos = xcoord[:, None]                                         # (N,1)
    coord_feat = torch.from_numpy(pos)                            # (N,1)

    tr_feats, tr_Y = build_features(tr_a, tr_u, coord_feat, mean, std)
    va_feats, va_Y = build_features(va_a, va_u, coord_feat, mean, std)
    te_feats, te_Y = build_features(te_a, te_u, coord_feat, mean, std)

    # ---- shared graph (cached) ----
    os.makedirs('cache', exist_ok=True)
    cache_path = f'cache/burgers_s{s}_k{args.k}_f{args.num_freq}.pt'
    if os.path.exists(cache_path):
        logger.info(f"graph cache hit: {cache_path}")
        gfeat = torch.load(cache_path)
    else:
        logger.info(f"building shared 1D graph (knn k={args.k}, eigh of {N}x{N}) ...")
        gfeat = build_graph_features(pos, k=args.k, num_freq=args.num_freq,
                                     seed=0, eig_device=args.device)
        torch.save(gfeat, cache_path)
        logger.info(f"cached graph -> {cache_path}")
    graph = SharedGraph(gfeat, args.device)

    in_dim = tr_feats.shape[-1]
    model = Sp2GNO(in_dim, width=args.width, n_layers=args.n_layers, N=N,
                   num_freq=args.num_freq, out_dim=1).to(args.device)
    logger.info(f"model in_dim={in_dim} width={args.width} layers={args.n_layers} "
                f"params={sum(p.numel() for p in model.parameters()):,}")

    plot_fn = functools.partial(plot_burgers, test_feats=te_feats, test_Y=te_Y, xcoord=xcoord)

    final_test, best_val, _ = train_shared(
        model, graph, tr_feats, tr_Y, va_feats, va_Y, te_feats, te_Y,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        eval_every=5, plot_every=50, label=tag, logger=logger,
        ckpt_best=ckpt_best_path,
        ckpt_last=os.path.join(ckpt_dir, 'burgers_last.pth'),
        resume=args.resume, plot_fn=plot_fn, plot_dir=plot_dir,
        curve_path=os.path.join(args.out_dir, 'curve.json'))
    n_params = sum(p.numel() for p in model.parameters())
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump({'model': 'Sp2GNO', 'label': tag, 'variant': args.variant,
                   'res': s, 'params': n_params, 'final_test_rel_l2': final_test,
                   'best_val_rel_l2': best_val, 'epochs': args.epochs,
                   's': s, 'N': N, 'ckpt': ckpt_best_path, 'args': vars(args)}, f, indent=2)
    logger.info(f"DONE {tag}: final test rel-L2 {final_test:.4f}, best val {best_val:.4f} "
                f"-> {ckpt_best_path}")


if __name__ == '__main__':
    main()
