from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # input: [B, Cin, M], weight: [Cin, Cout, M] -> [B, Cout, M]
        return torch.einsum('bim,iom->bom', input, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weight[:, :, :modes])
        x = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # input: [B, Cin, M1, M2], weight: [Cin, Cout, M1, M2] -> [B, Cout, M1, M2]
        return torch.einsum('bixy,ioxy->boxy', input, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        modes1 = min(self.modes1, x_ft.shape[-2])
        modes2 = min(self.modes2, x_ft.shape[-1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.shape[-2],
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2],
            self.weight1[:, :, :modes1, :modes2],
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2],
            self.weight2[:, :, :modes1, :modes2],
        )
        x = torch.fft.irfft2(out_ft, s=x.shape[-2:])
        return x
