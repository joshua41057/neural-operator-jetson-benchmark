#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import torch

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from sp2gno_core import (
    knn_graph,
    to_undirected,
    lipschitz_embedding,
    get_laplacian,
)

def build_darcy_cache(s, out, k=20, num_freq=64, lip_k=16, seed=0):
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if os.path.exists(out):
        print(f"[SKIP] {out}")
        return

    n = s * s
    print("=" * 80)
    print(f"[BUILD] Darcy graph cache")
    print(f"  s={s}  N={n}  k={k}  num_freq={num_freq}")
    print(f"  out={out}")
    print("=" * 80)

    x = np.linspace(0, 1, s, dtype=np.float32)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    pos_np = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
    pos = torch.from_numpy(pos_np)

    t0 = time.time()
    print("[1/6] knn_graph")
    edge_index = knn_graph(pos, k=k, loop=False)
    edge_index = to_undirected(edge_index, num_nodes=n)
    print("      edge_index:", tuple(edge_index.shape), "elapsed:", round(time.time() - t0, 1), "s")

    print("[2/6] edge weights")
    dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
    ew = (1.0 / (dist + 1e-6)).clip(max=5000)
    ew = (ew - ew.min()) / (ew.max() - ew.min())

    print("[3/6] Lipschitz embedding")
    lips = lipschitz_embedding(edge_index, dist, n, k=lip_k, seed=seed)
    print("      lips:", tuple(lips.shape))

    print("[4/6] normalized graph Laplacian sparse matrix")
    L_idx, L_val = get_laplacian(edge_index, normalization="sym")
    rows = L_idx[0].cpu().numpy()
    cols = L_idx[1].cpu().numpy()
    vals = L_val.cpu().numpy().astype(np.float64)

    L = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    print("      L shape:", L.shape, "nnz:", L.nnz)

    print("[5/6] sparse eigsh lowest modes")
    try:
        lam, U = eigsh(L, k=num_freq, which="SM", tol=1e-4, maxiter=10000)
    except ArpackNoConvergence as e:
        print("[ERROR] ARPACK did not fully converge")
        print("        converged eigenvalues:", None if e.eigenvalues is None else e.eigenvalues.shape)
        print("        converged eigenvectors:", None if e.eigenvectors is None else e.eigenvectors.shape)
        raise

    order = np.argsort(lam)
    lam = lam[order]
    U = U[:, order]

    gfeat = {
        "edge_index": edge_index.cpu(),
        "edge_weight": ew.float().cpu(),
        "lips": lips.float().cpu(),
        "U": torch.from_numpy(U.astype(np.float32)).cpu(),
        "lambdas": torch.from_numpy(lam.astype(np.float32)).cpu(),
    }

    print("[6/6] save")
    tmp = out + ".tmp"
    torch.save(gfeat, tmp)
    os.replace(tmp, out)

    print("[OK] saved", out)
    for kk, vv in gfeat.items():
        print("   ", kk, tuple(vv.shape), vv.dtype)
    print("elapsed total:", round(time.time() - t0, 1), "s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, required=True, choices=[141, 211])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    build_darcy_cache(args.s, args.out)

if __name__ == "__main__":
    main()
