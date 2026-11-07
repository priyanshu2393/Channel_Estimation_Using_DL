"""Pilot-Attention Coefficient Estimator (PACE-Net)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class PACEConfig:
    d: int = 16
    heads: int = 4
    n_enc_blocks: int = 2
    n_lat_blocks: int = 1
    ffn_mult: int = 2
    dropout: float = 0.0
    L: int = 6
    K: int = 3
    delay_step: float = 0.5
    doppler_step: float = 0.1
    learnable_basis: bool = True
    use_basis: bool = True
    use_linear_branch: bool = True
    linear_attn_signed: bool = False
    use_refine: bool = True
    latent_self_attn: bool = False
    input_rms_norm: bool = False
    n_fourier: int = 6
    Nf: int = 72
    Nt: int = 14

    @classmethod
    def preset(cls, name: str) -> "PACEConfig":
        if name == "pace_s":
            return cls()
        if name == "pace_m":
            return cls(d=64, dropout=0.1)
        raise ValueError(f"Unknown PACE-Net preset: {name}")


def fourier_features(coords_norm: torch.Tensor, n_freqs: int = 6) -> torch.Tensor:
    k = (2.0 ** torch.arange(n_freqs, device=coords_norm.device, dtype=coords_norm.dtype)) * math.pi
    x = coords_norm.unsqueeze(-1) * k
    return torch.cat((coords_norm, torch.cat((x.sin(), x.cos()), dim=-1).flatten(-2)), dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult*d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult*d, d))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.ffn(self.ln2(x))


class LatentBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: int, dropout: float, self_attn: bool):
        super().__init__()
        self.qnorm, self.knorm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if self_attn else None
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult*d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult*d, d))

    def forward(self, q: torch.Tensor, z: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        q = q + self.cross(self.qnorm(q), self.knorm(z), self.knorm(z), key_padding_mask=pad, need_weights=False)[0]
        if self.self_attn is not None:
            h = self.qnorm(q)
            q = q + self.self_attn(h, h, h, need_weights=False)[0]
        return q + self.ffn(self.norm2(q))


class DelayDopplerBasis(nn.Module):
    def __init__(self, Nf=72, Nt=14, L=6, K=3, delay_step=0.5, doppler_step=0.1, learnable=True):
        super().__init__()
        f = torch.arange(Nf, dtype=torch.float32)[:, None]
        tau = torch.arange(L, dtype=torch.float32)[None, :] * delay_step
        t = torch.arange(Nt, dtype=torch.float32)[:, None] - (Nt - 1) / 2
        nu = (torch.arange(K, dtype=torch.float32)[None, :] - (K - 1) / 2) * doppler_step
        bf = torch.exp(-2j * math.pi * f * tau / Nf)
        bt = torch.exp(2j * math.pi * t * nu / Nt)
        self.bf_re = nn.Parameter(bf.real, requires_grad=learnable)
        self.bf_im = nn.Parameter(bf.imag, requires_grad=learnable)
        self.bt_re = nn.Parameter(bt.real, requires_grad=learnable)
        self.bt_im = nn.Parameter(bt.imag, requires_grad=learnable)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        device_type = coefficients.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            bf = torch.complex(self.bf_re.float(), self.bf_im.float())
            bt = torch.complex(self.bt_re.float(), self.bt_im.float())
            c = torch.complex(coefficients[..., 0].float(), coefficients[..., 1].float())
            return torch.einsum("fl,blk,tk->bft", bf, c, bt)


class LinearInterp(nn.Module):
    def __init__(self, d: int, heads: int, signed: bool = False):
        super().__init__()
        self.h, self.dh, self.signed = heads, d // heads, signed
        self.wq, self.wk = nn.Linear(d, d), nn.Linear(d, d)
        self.c_re = nn.Parameter(torch.full((heads,), 1.0 / heads))
        self.c_im = nn.Parameter(torch.zeros(heads))

    def forward(self, q, z, h_ls_c, pad):
        B, G, _ = q.shape
        N = z.shape[1]
        device_type = q.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            Q = self.wq(q.float()).view(B, G, self.h, self.dh).transpose(1, 2)
            K = self.wk(z.float()).view(B, N, self.h, self.dh).transpose(1, 2)
            logits = Q @ K.transpose(-1, -2) / math.sqrt(self.dh)
            if pad is not None:
                logits = logits.masked_fill(pad[:, None, None, :], float("-inf"))
            w = torch.tanh(logits) / math.sqrt(N) if self.signed else logits.softmax(-1)
            if self.signed and pad is not None:
                w = w.masked_fill(pad[:, None, None, :], 0.0)
            out_h = torch.einsum("bhgn,bn->bhg", torch.complex(w, torch.zeros_like(w)), h_ls_c)
            c = torch.complex(self.c_re.float(), self.c_im.float())
            return torch.einsum("h,bhg->bg", c, out_h)


def _complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()


class PACENet(nn.Module):
    def __init__(self, config: PACEConfig | None = None):
        super().__init__()
        self.config = config or PACEConfig()
        c = self.config
        if not c.use_basis and not c.use_linear_branch:
            raise ValueError("At least one of use_basis/use_linear_branch must be enabled")
        if c.d % c.heads:
            raise ValueError("d must be divisible by heads")
        self.tokenizer = nn.Sequential(nn.Linear(2 + 2 + 4*c.n_fourier, c.d), nn.GELU(), nn.Linear(c.d, c.d))
        self.encoder = nn.ModuleList([EncoderBlock(c.d, c.heads, c.ffn_mult, c.dropout) for _ in range(c.n_enc_blocks)])
        self.znorm = nn.LayerNorm(c.d)
        self.latent_queries = nn.Parameter(torch.randn(c.L*c.K, c.d) * 0.02)
        self.latent_blocks = nn.ModuleList([LatentBlock(c.d, c.heads, c.ffn_mult, c.dropout, c.latent_self_attn) for _ in range(c.n_lat_blocks)])
        self.coefficient_head = nn.Linear(c.d, 2)
        nn.init.normal_(self.coefficient_head.weight, std=0.01)
        nn.init.zeros_(self.coefficient_head.bias)
        if c.use_basis:
            self.basis = DelayDopplerBasis(c.Nf, c.Nt, c.L, c.K, c.delay_step, c.doppler_step, c.learnable_basis)
        if c.use_linear_branch:
            grid = torch.stack(torch.meshgrid(torch.arange(c.Nf), torch.arange(c.Nt), indexing="ij"), -1).reshape(-1, 2).float()
            grid = grid / torch.tensor([max(c.Nf-1, 1), max(c.Nt-1, 1)])
            self.register_buffer("grid_coords", grid, persistent=True)
            self.grid_embed = nn.Sequential(nn.Linear(2 + 4*c.n_fourier, c.d), nn.GELU(), nn.Linear(c.d, c.d))
            self.ctx_to_q = nn.Linear(c.d, c.d)
            self.linear_interp = LinearInterp(c.d, c.heads, c.linear_attn_signed)
        self.gate = nn.Linear(c.d, 1)
        nn.init.zeros_(self.gate.bias)
        if c.use_refine:
            self.refine = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 2, 3, padding=1))
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    def forward(self, h_ls: torch.Tensor, coords: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        c = self.config
        if h_ls.ndim != 3 or h_ls.shape[-1] != 2:
            raise ValueError("h_ls must have shape [B,N,2]")
        B, N, _ = h_ls.shape
        if coords.ndim == 2:
            coords = coords.unsqueeze(0).expand(B, -1, -1)
        if coords.shape != (B, N, 2):
            raise ValueError("coords must have shape [N,2] or [B,N,2]")
        if pad is not None:
            pad = pad.to(device=h_ls.device, dtype=torch.bool)
            if pad.shape != (B, N):
                raise ValueError("pad must have shape [B,N]")
            if torch.any(pad.all(dim=1)):
                raise ValueError("Each sample must contain at least one unpadded pilot")
        scale = torch.ones((B, 1, 1), device=h_ls.device, dtype=h_ls.dtype)
        values = h_ls
        if c.input_rms_norm:
            valid = torch.ones((B,N,1), device=h_ls.device, dtype=h_ls.dtype) if pad is None else (~pad).unsqueeze(-1)
            scale = ((h_ls.square().sum(-1, keepdim=True) * valid).sum(1, keepdim=True) / (valid.sum(1, keepdim=True)*2).clamp_min(1)).sqrt().clamp_min(1e-8)
            values = h_ls / scale
        norm_coords = coords.to(h_ls.dtype) / torch.tensor([max(c.Nf-1,1), max(c.Nt-1,1)], device=h_ls.device, dtype=h_ls.dtype)
        tokens = self.tokenizer(torch.cat((values, fourier_features(norm_coords, c.n_fourier)), dim=-1))
        z = tokens
        for block in self.encoder:
            z = block(z, pad)
        if pad is None:
            z_mean = z.mean(1)
        else:
            valid = (~pad).to(z.dtype).unsqueeze(-1)
            z_mean = (z * valid).sum(1) / valid.sum(1).clamp_min(1)
        z_mean = self.znorm(z_mean)
        qlat = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.latent_blocks:
            qlat = block(qlat, z, pad)
        coeff = self.coefficient_head(qlat).view(B, c.L, c.K, 2)
        h_basis = self.basis(coeff) if c.use_basis else None
        if c.use_linear_branch:
            grid_feat = self.grid_embed(fourier_features(self.grid_coords.to(h_ls.dtype), c.n_fourier))
            q = grid_feat.unsqueeze(0) + self.ctx_to_q(z_mean).unsqueeze(1)
            h_ls_c = torch.complex(values[..., 0].float(), values[..., 1].float())
            h_lin = self.linear_interp(q, z, h_ls_c, pad).view(B, c.Nf, c.Nt)
        else:
            h_lin = None
        if h_basis is not None and h_lin is not None:
            g = torch.sigmoid(self.gate(z_mean)).view(B, 1, 1, 1)
            h0 = g * _complex_to_ri(h_basis) + (1-g) * _complex_to_ri(h_lin)
        elif h_basis is not None:
            h0 = _complex_to_ri(h_basis)
        else:
            h0 = _complex_to_ri(h_lin)
        if c.use_refine:
            hb = _complex_to_ri(h_basis) if h_basis is not None else torch.zeros_like(h0)
            hl = _complex_to_ri(h_lin) if h_lin is not None else torch.zeros_like(h0)
            h0 = h0 + self.refine(torch.cat((h0, hb, hl), dim=1))
        return h0 * scale.permute(0,2,1).unsqueeze(-1)


def to_pilot_tokens(batch: torch.Tensor, pilot_coords: torch.Tensor | list | None = None, pad: torch.Tensor | None = None):
    """Convert compact [B,2,Pf,Pt] or zero-filled [B,2,72,14] inputs to pilot tokens.

    Coordinates are returned in the caller's order. Compact uniform grids follow
    frequency-major, then time-major ordering, matching manifest and dataset storage.
    """
    if batch.ndim != 4 or batch.shape[1] != 2:
        raise ValueError("batch must have shape [B,2,Pf,Pt] or [B,2,72,14]")
    if pilot_coords is None:
        raise ValueError("pilot_coords are required; obtain them from the manifest or pilot index file")
    coords = torch.as_tensor(pilot_coords, device=batch.device, dtype=torch.long)
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("pilot_coords must be [N,2] as (subcarrier,symbol)")
    if batch.shape[2:] == (72, 14):
        h = batch.permute(0, 2, 3, 1)[:, coords[:, 0], coords[:, 1], :]
    else:
        if batch.shape[2] * batch.shape[3] != len(coords):
            raise ValueError("compact input pilot count does not match coordinates")
        h = batch.permute(0, 2, 3, 1).reshape(batch.shape[0], -1, 2)
    return h, coords, pad


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().sum(dim=1).mean()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
