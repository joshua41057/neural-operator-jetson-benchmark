from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        name = activation.lower()
        if name == "gelu":
            act = nn.GELU
        elif name == "relu":
            act = nn.ReLU
        elif name == "tanh":
            act = nn.Tanh
        elif name == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        last = int(in_dim)
        for _ in range(int(n_layers) - 1):
            layers.append(nn.Linear(last, int(hidden_dim)))
            layers.append(act())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            last = int(hidden_dim)

        layers.append(nn.Linear(last, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GridDeepONet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_resolution: int,
        sensor_resolution: int,
        basis_dim: int,
        branch_hidden_dim: int,
        branch_layers: int,
        trunk_hidden_dim: int,
        trunk_layers: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_coord_features: bool = True,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.spatial_dim = 1
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_resolution = int(input_resolution)
        self.sensor_resolution = int(sensor_resolution)
        self.basis_dim = int(basis_dim)
        self.use_coord_features = bool(use_coord_features)
        self.chunk_size = int(chunk_size)

        branch_in_dim = self.sensor_resolution * self.in_channels
        trunk_in_dim = 3 if self.use_coord_features else 1

        self.branch = MLP(
            in_dim=branch_in_dim,
            out_dim=self.out_channels * self.basis_dim,
            hidden_dim=int(branch_hidden_dim),
            n_layers=int(branch_layers),
            activation=activation,
            dropout=float(dropout),
        )
        self.trunk = MLP(
            in_dim=trunk_in_dim,
            out_dim=self.out_channels * self.basis_dim,
            hidden_dim=int(trunk_hidden_dim),
            n_layers=int(trunk_layers),
            activation=activation,
            dropout=float(dropout),
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def _make_coords(self, n: int, ref: torch.Tensor) -> torch.Tensor:
        coord = torch.linspace(
            0.0,
            1.0,
            n,
            device=ref.device,
            dtype=ref.dtype,
        ).unsqueeze(-1)

        if not self.use_coord_features:
            return coord

        # Do not use ref.new_tensor(...): TorchScript on this Jetson build rejects it.
        two_pi = torch.ones((), device=ref.device, dtype=ref.dtype) * 6.283185307179586
        return torch.cat(
            [
                coord,
                torch.sin(two_pi * coord),
                torch.cos(two_pi * coord),
            ],
            dim=-1,
        )

    def _branch_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> [B, C, L]
        z = x.permute(0, 2, 1)
        z = F.interpolate(
            z,
            size=self.sensor_resolution,
            mode="linear",
            align_corners=True,
        )
        z = z.permute(0, 2, 1).reshape(x.shape[0], -1)
        return z

    def _eval_trunk(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [L, trunk_in]
        t = self.trunk(coords)
        return t.view(coords.shape[0], self.out_channels, self.basis_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        n = x.shape[1]

        b = self.branch(self._branch_input(x))
        b = b.view(bsz, self.out_channels, self.basis_dim)

        coords = self._make_coords(n, x)

        if self.chunk_size > 0 and n > self.chunk_size:
            outs = torch.jit.annotate(list[torch.Tensor], [])
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                t = self._eval_trunk(coords[start:end])
                y_chunk = torch.einsum("bck,lck->blc", b, t)
                outs.append(y_chunk)
            y = torch.cat(outs, dim=1)
        else:
            t = self._eval_trunk(coords)
            y = torch.einsum("bck,lck->blc", b, t)

        y = y + self.bias.view(1, 1, -1)
        return y


class GridDeepONet2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_resolution: Sequence[int],
        sensor_resolution: Sequence[int],
        basis_dim: int,
        branch_hidden_dim: int,
        branch_layers: int,
        trunk_hidden_dim: int,
        trunk_layers: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_coord_features: bool = True,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.spatial_dim = 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.input_h = int(input_resolution[0])
        self.input_w = int(input_resolution[1])
        self.sensor_h = int(sensor_resolution[0])
        self.sensor_w = int(sensor_resolution[1])

        self.input_resolution = (self.input_h, self.input_w)
        self.sensor_resolution = (self.sensor_h, self.sensor_w)

        self.basis_dim = int(basis_dim)
        self.use_coord_features = bool(use_coord_features)
        self.chunk_size = int(chunk_size)

        branch_in_dim = self.sensor_h * self.sensor_w * self.in_channels
        trunk_in_dim = 6 if self.use_coord_features else 2

        self.branch = MLP(
            in_dim=branch_in_dim,
            out_dim=self.out_channels * self.basis_dim,
            hidden_dim=int(branch_hidden_dim),
            n_layers=int(branch_layers),
            activation=activation,
            dropout=float(dropout),
        )
        self.trunk = MLP(
            in_dim=trunk_in_dim,
            out_dim=self.out_channels * self.basis_dim,
            hidden_dim=int(trunk_hidden_dim),
            n_layers=int(trunk_layers),
            activation=activation,
            dropout=float(dropout),
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def _make_coords(self, h: int, w: int, ref: torch.Tensor) -> torch.Tensor:
        yy = torch.linspace(0.0, 1.0, h, device=ref.device, dtype=ref.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=ref.device, dtype=ref.dtype)

        # Avoid torch.meshgrid(indexing="ij") for TorchScript portability.
        gy = yy.view(h, 1).expand(h, w)
        gx = xx.view(1, w).expand(h, w)

        coord = torch.stack(
            [
                gx.reshape(-1),
                gy.reshape(-1),
            ],
            dim=1,
        )

        if not self.use_coord_features:
            return coord

        two_pi = torch.ones((), device=ref.device, dtype=ref.dtype) * 6.283185307179586
        xcoord = coord[:, 0:1]
        ycoord = coord[:, 1:2]

        return torch.cat(
            [
                xcoord,
                ycoord,
                torch.sin(two_pi * xcoord),
                torch.cos(two_pi * xcoord),
                torch.sin(two_pi * ycoord),
                torch.cos(two_pi * ycoord),
            ],
            dim=-1,
        )

    def _branch_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C] -> [B, C, H, W]
        z = x.permute(0, 3, 1, 2)
        z = F.interpolate(
            z,
            size=[self.sensor_h, self.sensor_w],
            mode="bilinear",
            align_corners=True,
        )
        z = z.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        return z

    def _eval_trunk(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [H*W, trunk_in]
        t = self.trunk(coords)
        return t.view(coords.shape[0], self.out_channels, self.basis_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        h = x.shape[1]
        w = x.shape[2]
        n = h * w

        b = self.branch(self._branch_input(x))
        b = b.view(bsz, self.out_channels, self.basis_dim)

        coords = self._make_coords(h, w, x)

        if self.chunk_size > 0 and n > self.chunk_size:
            outs = torch.jit.annotate(list[torch.Tensor], [])
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                t = self._eval_trunk(coords[start:end])
                y_chunk = torch.einsum("bck,nck->bnc", b, t)
                outs.append(y_chunk)
            y = torch.cat(outs, dim=1)
        else:
            t = self._eval_trunk(coords)
            y = torch.einsum("bck,nck->bnc", b, t)

        y = y + self.bias.view(1, 1, -1)
        y = y.view(bsz, h, w, self.out_channels)
        return y


def _as_resolution_tuple(res) -> Tuple[int, ...]:
    if isinstance(res, int):
        return (int(res),)
    if isinstance(res, (list, tuple)):
        return tuple(int(v) for v in res)
    return (int(res),)


def build_deeponet_model(cfg, input_channels: int, output_channels: int) -> nn.Module:
    spatial_dim = int(cfg.model.spatial_dim)
    res_tuple = _as_resolution_tuple(cfg.data.resolution)

    if spatial_dim == 1:
        return GridDeepONet1d(
            in_channels=input_channels,
            out_channels=output_channels,
            input_resolution=int(res_tuple[0]),
            sensor_resolution=int(cfg.model.sensor_resolution),
            basis_dim=int(cfg.model.basis_dim),
            branch_hidden_dim=int(cfg.model.branch_hidden_dim),
            branch_layers=int(cfg.model.branch_layers),
            trunk_hidden_dim=int(cfg.model.trunk_hidden_dim),
            trunk_layers=int(cfg.model.trunk_layers),
            activation=str(cfg.model.activation),
            dropout=float(cfg.model.dropout),
            use_coord_features=bool(cfg.model.use_coord_features),
            chunk_size=int(cfg.model.chunk_size),
        )

    if spatial_dim == 2:
        if len(res_tuple) == 1:
            input_resolution = (int(res_tuple[0]), int(res_tuple[0]))
        else:
            input_resolution = (int(res_tuple[0]), int(res_tuple[1]))

        sensor = cfg.model.sensor_resolution
        if isinstance(sensor, int):
            sensor_resolution = (int(sensor), int(sensor))
        elif isinstance(sensor, (list, tuple)):
            if len(sensor) == 1:
                sensor_resolution = (int(sensor[0]), int(sensor[0]))
            else:
                sensor_resolution = (int(sensor[0]), int(sensor[1]))
        else:
            sensor_resolution = (int(sensor), int(sensor))

        return GridDeepONet2d(
            in_channels=input_channels,
            out_channels=output_channels,
            input_resolution=input_resolution,
            sensor_resolution=sensor_resolution,
            basis_dim=int(cfg.model.basis_dim),
            branch_hidden_dim=int(cfg.model.branch_hidden_dim),
            branch_layers=int(cfg.model.branch_layers),
            trunk_hidden_dim=int(cfg.model.trunk_hidden_dim),
            trunk_layers=int(cfg.model.trunk_layers),
            activation=str(cfg.model.activation),
            dropout=float(cfg.model.dropout),
            use_coord_features=bool(cfg.model.use_coord_features),
            chunk_size=int(cfg.model.chunk_size),
        )

    raise ValueError(f"Unsupported spatial_dim={spatial_dim}")
