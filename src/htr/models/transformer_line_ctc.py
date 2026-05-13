"""Трансформерный кодировщик последовательности признаков строки с головкой CTC (§4.3)."""

from __future__ import annotations

from typing import Tuple

import math

import torch
from torch import nn

from htr.models.crnn_ctc import ConvBlock


class PositionalEncoding1D(nn.Module):
    """Синусоидальная разметка вдоль длины T (до входа трансформера)."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [T,B,D]
        t = x.size(0)
        return x + self.pe[:t]


class TransformerEncoderLineCTC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        stride_h = [(2, 2), (2, 2), (2, 1), (2, 1)]
        ch = [(in_channels, 64), (64, 128), (128, 256), (256, 256)]
        blocks: list[nn.Module] = []
        ci = in_channels
        for (_, co), pt in zip(ch, stride_h):
            blocks.append(ConvBlock(ci, co, pool=pt))
            ci = co
        self.cnn = nn.Sequential(*blocks)
        feat_c = ci
        self.proj_in = nn.Linear(feat_c, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding1D(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode_sequence(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """[B,1,H,W] -> feats [T,B,d_model], lengths."""
        self._expect_h32(inputs)
        feats_map = self.cnn(inputs)
        feats = feats_map.mean(dim=2)
        feats = feats.permute(2, 0, 1)
        x = self.proj_in(feats)
        x = self.pos(x)
        b = inputs.shape[0]
        t_src = x.shape[0]
        mask = None
        out = self.encoder(x, mask=mask, src_key_padding_mask=None)
        tlens = torch.full((b,), t_src, device=inputs.device, dtype=torch.long).clamp(min=1, max=t_src)
        return out, tlens

    @staticmethod
    def _expect_h32(inputs: torch.Tensor) -> None:
        h_pix = inputs.shape[2]
        if h_pix != 32:
            raise ValueError(f"ожидается H=32, получено H={h_pix}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq, _ = self.encode_sequence(inputs)
        logits = self.head(seq)
        return torch.log_softmax(logits, dim=-1)


def build_transformer_line_ctc(num_classes: int, model_cfg: dict, in_channels: int = 1) -> TransformerEncoderLineCTC:
    return TransformerEncoderLineCTC(
        in_channels=in_channels,
        num_classes=int(num_classes),
        d_model=int(model_cfg.get("transformer_d_model", 256)),
        nhead=int(model_cfg.get("transformer_nhead", 4)),
        num_layers=int(model_cfg.get("transformer_encoder_layers", 2)),
        dim_feedforward=int(model_cfg.get("transformer_ff_dim", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
