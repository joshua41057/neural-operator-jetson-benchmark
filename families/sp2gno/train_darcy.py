"""
train_darcy.py — Sp2GNO on the 2D Darcy-flow benchmark (piececonst_r421_N1024).

Task: learn the operator a(x) -> u(x) (coefficient field -> pressure solution) on
the regular 421x421 grid, subsampled by r (default r=5 -> s=85, N=s*s=7225 nodes).
The grid is identical for every sample, so a single shared Sp2GNO graph (knn,
inverse-distance edge weights, Lipschitz embeddings, Laplacian eigenpairs) is built
once and reused by every sample (cached to ./cache/).

Data:
  train = smooth1[:ntrain]          (default 900)
  val   = smooth1[ntrain:ntrain+nval]  (default 100, used to pick the best checkpoint)
  test  = smooth2[:ntest]           (default 200, the canonical Darcy test set)

Node features per grid point: the (x,y) grid coordinates (2) + the (z-scored)
coefficient value (1) -> in_dim = 3. (Positional information is otherwise carried
by the graph-Laplacian eigenvectors inside the spectral layers.) Targets are the
raw solution; loss is per-sample relative L2 (scale invariant, so no output
normalization is needed).

Usage:
  conda activate BGCN_torch
  python train_darcy.py                 # train 500 epochs from scratch
  python train_darcy.py --resume        # resume from the last checkpoint
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


def load_darcy(path: str, n: int, r: int):
    d = sio.loadmat(path)
    coeff = d['coeff'][:n, ::r, ::r].astype(np.float32)   # (n,s,s)
    sol = d['sol'][:n, ::r, ::r].astype(np.float32)       # (n,s,s)
    return coeff, sol


def build_features(coeff, sol, coord_feat, mean, std):
    """coeff,sol (M,s,s) -> feats (M,N,3) = [x,y,coeff_norm], Y (M,N,1)."""
    M, s, _ = coeff.shape
    N = s * s
    cf = ((coeff.reshape(M, N) - mean) / std)[..., None]                 # (M,N,1)
    feats = torch.cat([coord_feat.unsqueeze(0).expand(M, -1, -1),
                       torch.from_numpy(cf)], dim=-1)                    # (M,N,3)
    Y = torch.from_numpy(sol.reshape(M, N, 1))
    return feats.contiguous(), Y.contiguous()


def plot_darcy(model, graph, epoch, plot_dir, *, test_feats, test_Y, test_coeff, s, tag=''):
    model.eval()
    idx = [0, 1, 2, 3]
    with torch.no_grad():
        feats, U, ei, ew, lips, Y = graph.batch(test_feats, test_Y, idx)
        pred = model(feats, U, ei, ew, lips).cpu().numpy()              # (4,N,1)
    X, Yg = np.meshgrid(np.arange(s), np.arange(s), indexing='ij')
    rows = ['Input a(x)', 'Ground Truth u', 'Prediction u', 'Squared Error']
    fig, ax = plt.subplots(4, 4, figsize=(24, 22))
    for c, i in enumerate(idx):
        inp = test_coeff[i].reshape(s, s)
        truth = test_Y[i, :, 0].numpy().reshape(s, s)
        pr = pred[c, :, 0].reshape(s, s)
        err = (truth - pr) ** 2
        vmin, vmax = truth.min(), truth.max()
        for r, (data, vlim) in enumerate([(inp, None), (truth, (vmin, vmax)),
                                          (pr, (vmin, vmax)), (err, None)]):
            kw = dict(cmap='RdBu_r', shading='gouraud')
            if vlim:
                kw.update(vmin=vlim[0], vmax=vlim[1])
            pic = ax[r, c].pcolormesh(X, Yg, data, **kw)
            fig.colorbar(pic, ax=ax[r, c])
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    for r, t in enumerate(rows):
        ax[r, 0].set_ylabel(t, fontsize=22, fontweight='bold')
    plt.tight_layout()
    name = tag if tag else f'ep{epoch}'
    fn = os.path.join(plot_dir, f'darcy_pred_{name}.png')
    plt.savefig(fn, dpi=100); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Jetson_data')
    ap.add_argument('--r', type=int, default=5, help='grid subsample factor (s=(421-1)//r+1)')
    ap.add_argument('--ntrain', type=int, default=900)
    ap.add_argument('--nval', type=int, default=100)
    ap.add_argument('--ntest', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=10)
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--num_freq', type=int, default=64)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out_dir', default='runs/darcy')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, 'plots'); os.makedirs(plot_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'ckpt'); os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logging(args.out_dir, name='darcy')
    logger.info(f"Args: {vars(args)}")
    set_all_seeds(args.seed)

    # ---- data ----
    coeff1, sol1 = load_darcy(os.path.join(args.data_dir, 'piececonst_r421_N1024_smooth1.mat'),
                              args.ntrain + args.nval, args.r)
    coeff2, sol2 = load_darcy(os.path.join(args.data_dir, 'piececonst_r421_N1024_smooth2.mat'),
                              args.ntest, args.r)
    s = coeff1.shape[1]; N = s * s
    logger.info(f"resolution s={s} N={N}  train={args.ntrain} val={args.nval} test={args.ntest}")

    tr_coeff, tr_sol = coeff1[:args.ntrain], sol1[:args.ntrain]
    va_coeff, va_sol = coeff1[args.ntrain:args.ntrain + args.nval], sol1[args.ntrain:args.ntrain + args.nval]
    te_coeff, te_sol = coeff2, sol2
    mean, std = float(tr_coeff.mean()), float(tr_coeff.std())
    logger.info(f"coeff normalization mean={mean:.4f} std={std:.4f}")

    gx, gy = np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s), indexing='ij')
    pos = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)   # (N,2)
    coord_feat = torch.from_numpy(pos)                                     # (N,2)

    tr_feats, tr_Y = build_features(tr_coeff, tr_sol, coord_feat, mean, std)
    va_feats, va_Y = build_features(va_coeff, va_sol, coord_feat, mean, std)
    te_feats, te_Y = build_features(te_coeff, te_sol, coord_feat, mean, std)

    # ---- shared graph (cached) ----
    os.makedirs('cache', exist_ok=True)
    cache_path = f'cache/darcy_s{s}_k{args.k}_f{args.num_freq}.pt'
    if os.path.exists(cache_path):
        logger.info(f"graph cache hit: {cache_path}")
        gfeat = torch.load(cache_path)
    else:
        logger.info(f"building shared graph (knn k={args.k}, eigh of {N}x{N}) ...")
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

    plot_fn = functools.partial(plot_darcy, test_feats=te_feats, test_Y=te_Y,
                                test_coeff=te_coeff, s=s)

    final_test, best_val, _ = train_shared(
        model, graph, tr_feats, tr_Y, va_feats, va_Y, te_feats, te_Y,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        eval_every=5, plot_every=50, label='darcy', logger=logger,
        ckpt_best=os.path.join(ckpt_dir, 'darcy_best.pth'),
        ckpt_last=os.path.join(ckpt_dir, 'darcy_last.pth'),
        resume=args.resume, plot_fn=plot_fn, plot_dir=plot_dir,
        curve_path=os.path.join(args.out_dir, 'curve.json'))
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump({'label': 'darcy', 'final_test_rel_l2': final_test,
                   'best_val_rel_l2': best_val, 'epochs': args.epochs,
                   's': s, 'N': N, 'args': vars(args)}, f, indent=2)
    logger.info(f"DONE darcy: final test rel-L2 {final_test:.4f}, best val {best_val:.4f}")


if __name__ == '__main__':
    main()
