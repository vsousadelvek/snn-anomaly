"""Autoencoder SNN para deteccao de anomalias em eventos.

Encoder de spikes comprime o event frame 2x64x64 numa representacao
latente; decoder reconstrói. Treinado so com cena NORMAL — erro de
reconstrucao alto = padrao nunca visto = anomalia.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_mem, v_th=1.0):
        ctx.save_for_backward(v_mem)
        ctx.v_th = v_th
        return (v_mem >= v_th).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v_mem,) = ctx.saved_tensors
        x = torch.pi * (v_mem - ctx.v_th)
        return grad_out / (1.0 + x * x), None


class LIF(nn.Module):
    def __init__(self, beta=0.85, v_th=1.0):
        super().__init__()
        self.beta, self.v_th = beta, v_th

    def forward(self, x, mem):
        mem = self.beta * mem + x
        s = SpikeFn.apply(mem, self.v_th)
        return s, mem * (1.0 - s)


class SpikingAE(nn.Module):
    """2x64x64 -> latente -> reconstrucao. Bottleneck pequeno forca o
    modelo a aprender so os padroes de movimento normais."""

    def __init__(self, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        # encoder spiking
        self.enc1 = nn.Conv2d(2, 16, 3, stride=2, padding=1)    # 32
        self.lif1 = LIF()
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)   # 16
        self.lif2 = LIF(v_th=0.8)
        self.enc3 = nn.Conv2d(32, 48, 3, stride=2, padding=1)   # 8
        self.lif3 = LIF(v_th=0.6)
        # decoder deterministico
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),  # 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1),   # 64
        )

    def encode(self, x):
        B = x.size(0)
        m1 = torch.zeros(B, 16, 32, 32, device=x.device)
        m2 = torch.zeros(B, 32, 16, 16, device=x.device)
        m3 = torch.zeros(B, 48, 8, 8, device=x.device)
        for _ in range(self.num_steps):
            s, m1 = self.lif1(self.enc1(x), m1)
            s, m2 = self.lif2(self.enc2(s), m2)
            s, m3 = self.lif3(self.enc3(s), m3)
        return m3 / self.num_steps  # latente = membrana media

    def forward(self, x):
        return self.dec(self.encode(x))
