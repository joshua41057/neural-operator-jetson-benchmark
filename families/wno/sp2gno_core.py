"""
sp2gno_core.py — batched Sp2GNO model + utilities, reused for the Darcy and
Burgers benchmarks.

The model (MLP encoder P(x) + [GraphFourierLayer ∥ FrigateConv] x L) and the
graph-feature preprocessing (knn graph, inverse-distance edge weights, Lipschitz
embeddings via scipy Dijkstra, sym-normalized Laplacian eigenpairs) are ported
verbatim from train_sp2gno_elasticity.py (the batched implementation). The only
additions here are:
  - a *shared-graph* batching path (Darcy/Burgers use one grid graph for every
    sample, unlike elasticity's per-mesh meshes), and
  - a generic trainer with best-on-validation checkpointing and resume support.
"""

import logging
import os
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_undirected, get_laplacian


# ============================================================
# Generic helpers
# ============================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logging(log_dir: str, name: str = 'sp2gno') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'run.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================
# Graph feature preprocessing (verbatim from elasticity)
# ============================================================
def lipschitz_embedding(edge_index: torch.Tensor, dist: torch.Tensor, n: int,
                        k: int = 16, seed: int = 0) -> torch.Tensor:
    """16-dim Lipschitz node embedding: 1/(d+1) of Dijkstra distances to k random
    anchors, z-scored per anchor."""
    rng = np.random.RandomState(seed)
    anchors = rng.choice(n, size=k, replace=False)
    ei = edge_index.cpu().numpy()
    w = dist.cpu().numpy()
    G = sp.csr_matrix((w, (ei[0], ei[1])), shape=(n, n))
    d = dijkstra(G, directed=False, indices=anchors)          # (k, n), inf if unreachable
    emb = 1.0 / (d + 1.0)                                     # inf -> 0
    emb[~np.isfinite(emb)] = 0.0
    emb = (emb - emb.mean(axis=1, keepdims=True)) / emb.std(axis=1, keepdims=True)
    return torch.from_numpy(emb.T.astype(np.float32))         # (n, k)


def build_graph_features(pos_np: np.ndarray, *, k: int = 20, num_freq: int = 64,
                         lip_k: int = 16, seed: int = 0, eig_device: str = 'cuda') -> dict:
    """Per-graph preprocessing: knn(k)+to_undirected, min-max normalized inverse
    distance edge weights, Lipschitz embeddings, eigenpairs of the sym-normalized
    Laplacian (lowest num_freq)."""
    pos = torch.from_numpy(pos_np.astype(np.float32))
    if pos.dim() == 1:
        pos = pos.unsqueeze(-1)
    n = pos.shape[0]
    edge_index = knn_graph(pos, k=k, loop=False)
    edge_index = to_undirected(edge_index, num_nodes=n)
    dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
    ew = (1.0 / (dist + 1e-6)).clip(max=5000)
    ew = (ew - ew.min()) / (ew.max() - ew.min())
    lips = lipschitz_embedding(edge_index, dist, n, k=lip_k, seed=seed)
    L_idx, L_val = get_laplacian(edge_index, normalization='sym')
    L = torch.sparse_coo_tensor(L_idx, L_val, (n, n)).to_dense().to(eig_device)
    lam, U = torch.linalg.eigh(L)
    return {'edge_index': edge_index, 'edge_weight': ew.float(),
            'lips': lips, 'U': U[:, :num_freq].float().cpu(),
            'lambdas': lam[:num_freq].float().cpu()}


# ============================================================
# Model (verbatim batched Sp2GNO from elasticity)
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        return self.mlp2(F.relu(self.mlp1(x)))


class GraphFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, width, N, num_freq):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.num_nodes = N
        self.no_low_freq = num_freq
        self.w = nn.Linear(width, width)
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.no_low_freq))
        self.norm2 = nn.LayerNorm(width)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bmi,iom->bmo", input, weights)

    def forward(self, x, U):
        B = U.shape[0]
        x_g = x.view(B, self.num_nodes, self.in_channels)
        x_ft = torch.bmm(U.transpose(1, 2), x_g)                       # (B, K, C)
        out_ft = self.compl_mul2d(x_ft, self.weights)
        x1 = torch.bmm(U, out_ft).view(-1, self.in_channels)           # (B*N, C)
        return self.norm2(F.gelu(x1 + self.w(x)))


class FrigateConv(MessagePassing):
    def __init__(self, in_channels, out_channels, lip_dim: int = 16):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_ew = nn.Linear(1, 16)
        self.gate = nn.Sequential(
            nn.Linear(16 + 2 * lip_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_weight, lipschitz_embeddings):
        x = self.lin(x)
        out = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight,
                             lipschitz_embeddings=lipschitz_embeddings)
        return F.normalize(out, p=2., dim=-1)

    def message(self, x_j, edge_index_i, edge_index_j, edge_weight, lipschitz_embeddings):
        ew_j = self.lin_ew(edge_weight.view(-1, 1))
        gate_input = torch.cat((ew_j, lipschitz_embeddings[edge_index_i],
                                lipschitz_embeddings[edge_index_j]), dim=1)
        return x_j * self.gate(gate_input)


class Sp2GNO(nn.Module):
    """MLP encoder P(x) -> [GraphFourierLayer ∥ FrigateConv] x L -> projection -> q."""

    def __init__(self, in_dim: int, width: int = 48, n_layers: int = 6,
                 N: int = 972, num_freq: int = 64, out_dim: int = 1):
        super().__init__()
        self.num_nodes = N
        self.no_low_freq = num_freq
        self.out_dim = out_dim
        self.p = nn.Linear(in_dim, width)
        self.graph_fourier_layers = nn.ModuleList([
            GraphFourierLayer(width, width, width, N, num_freq) for _ in range(n_layers)])
        self.spatial_convs = nn.ModuleList([FrigateConv(width, width) for _ in range(n_layers)])
        self.projection = nn.Linear(2 * width, width)
        self.q = MLP(width, out_dim, 128)

    def forward(self, feats, U, edge_index, edge_weight, lips):
        """feats (B,N,F); U (B,N,K) already batched; edge_index (2,E_tot) batch-shifted;
        edge_weight (E_tot,); lips (B*N, 16)."""
        B, N = feats.shape[0], feats.shape[1]
        x = self.p(feats.view(B * N, -1))
        for fourier, conv in zip(self.graph_fourier_layers, self.spatial_convs):
            x_fourier = fourier(x, U)
            x_spatial = conv(x, edge_index, edge_weight, lips)
            x = self.projection(torch.cat([x_fourier, x_spatial], dim=1))
        return self.q(x).view(B, N, self.out_dim)


# ============================================================
# Loss
# ============================================================
def rel_l2(pred: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Per-graph relative L2, mean over batch (== LpLoss.rel)."""
    diff = (pred - Y).flatten(1).norm(dim=1)
    return (diff / (Y.flatten(1).norm(dim=1) + 1e-8)).mean()


# ============================================================
# Shared-graph batching (one grid graph reused by every sample)
# ============================================================
class SharedGraph:
    """Holds the single grid graph's tensors on `device` and assembles the
    batch-shifted inputs the batched Sp2GNO expects."""

    def __init__(self, gfeat: dict, device: str):
        self.device = device
        self.edge_index = gfeat['edge_index'].to(device)          # (2,E)
        self.edge_weight = gfeat['edge_weight'].to(device)        # (E,)
        self.lips = gfeat['lips'].to(device)                      # (N,16)
        self.U = gfeat['U'].to(device)                            # (N,K)
        self.N = self.U.shape[0]
        self.E = self.edge_index.shape[1]

    def batch(self, feats_cpu: torch.Tensor, Y_cpu: torch.Tensor, idx):
        """feats_cpu (M,N,F), Y_cpu (M,N,out); idx -> (feats,U,ei,ew,lips,Y) on device."""
        B = len(idx)
        feats = feats_cpu[idx].to(self.device)
        Y = Y_cpu[idx].to(self.device)
        offs = (torch.arange(B, device=self.device) * self.N).repeat_interleave(self.E)
        ei = self.edge_index.repeat(1, B) + offs
        ew = self.edge_weight.repeat(B)
        lips = self.lips.repeat(B, 1)
        U = self.U.unsqueeze(0).expand(B, -1, -1).contiguous()
        return feats, U, ei, ew, lips, Y


@torch.no_grad()
def evaluate_shared(model, graph: SharedGraph, feats_cpu, Y_cpu, batch_size=8):
    model.eval()
    errs = []
    M = feats_cpu.shape[0]
    for s in range(0, M, batch_size):
        idx = list(range(s, min(s + batch_size, M)))
        feats, U, ei, ew, lips, Y = graph.batch(feats_cpu, Y_cpu, idx)
        pred = model(feats, U, ei, ew, lips)
        d = (pred - Y).flatten(1).norm(dim=1) / (Y.flatten(1).norm(dim=1) + 1e-8)
        errs.append(d.cpu())
    return float(torch.cat(errs).mean())


def train_shared(model, graph: SharedGraph,
                 train_feats, train_Y, val_feats, val_Y, test_feats, test_Y,
                 *, epochs=500, batch_size=10, lr=1e-3, wd=1e-4,
                 step_size=100, gamma=0.65, eval_every=5, plot_every=50,
                 ckpt_best='best.pth', ckpt_last='last.pth', resume=False,
                 logger=None, label='sp2gno', plot_fn=None, plot_dir=None,
                 curve_path=None):
    """Generic trainer: rel-L2 loss, Adam + StepLR, best-on-val checkpoint, resume."""
    log = logger.info if logger else print
    device = graph.device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    start_epoch, best_val, curve = 0, float('inf'), []
    if resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        sched.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_val = ck.get('best_val', float('inf'))
        curve = ck.get('curve', [])
        log(f"  [{label}] resumed from {ckpt_last} at epoch {start_epoch} (best_val {best_val:.4f})")
    else:
        log(f"  [{label}] training from scratch")

    n_tr = train_feats.shape[0]
    t0 = time.time()
    for ep in range(start_epoch, epochs):
        model.train()
        perm = np.random.permutation(n_tr)
        tr_loss, nb = 0.0, 0
        for s in range(0, n_tr, batch_size):
            idx = perm[s:s + batch_size].tolist()
            feats, U, ei, ew, lips, Y = graph.batch(train_feats, train_Y, idx)
            loss = rel_l2(model(feats, U, ei, ew, lips), Y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); nb += 1
        sched.step()

        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            val = evaluate_shared(model, graph, val_feats, val_Y, batch_size)
            te = evaluate_shared(model, graph, test_feats, test_Y, batch_size)
            curve.append({'epoch': ep + 1, 'train_rel_l2': tr_loss / max(1, nb),
                          'val_rel_l2': val, 'test_rel_l2': te, 'lr': opt.param_groups[0]['lr'],
                          'seconds': time.time() - t0})
            improved = val < best_val
            if improved:
                best_val = val
                torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                            'val_rel_l2': val, 'test_rel_l2': te}, ckpt_best)
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': sched.state_dict(),
                        'best_val': best_val, 'curve': curve}, ckpt_last)
            log(f"  [{label}] ep {ep + 1}/{epochs} train {curve[-1]['train_rel_l2']:.4f} "
                f"val {val:.4f} test {te:.4f} (best_val {best_val:.4f}{' *' if improved else ''}, "
                f"{time.time() - t0:.0f}s)")
            if curve_path:
                import json
                with open(curve_path, 'w') as f:
                    json.dump(curve, f, indent=2)

        if plot_fn is not None and ((ep + 1) % plot_every == 0 or ep == epochs - 1):
            plot_fn(model, graph, ep + 1, plot_dir)

    # final report with the best-on-val checkpoint
    if os.path.exists(ckpt_best):
        ck = torch.load(ckpt_best, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        log(f"  [{label}] loaded best-on-val checkpoint from epoch {ck['epoch'] + 1}")
    final_test = evaluate_shared(model, graph, test_feats, test_Y, batch_size)
    log(f"  [{label}] FINAL test rel-L2 (best-on-val model): {final_test:.4f} "
        f"(best_val {best_val:.4f})")
    if plot_fn is not None:
        plot_fn(model, graph, epochs, plot_dir, tag='best')
    return final_test, best_val, curve
