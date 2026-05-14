"""CRNN‑подобный энкодер свёрточными блоками, BiLSTM, линейная головка + CTC (реализация §4.1 без претрейна)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from htr.line_width_limits import MIN_CRNN_RESIZED_WIDTH_PX


class ConvBlock(nn.Sequential):
    def __init__(self, ci: int, co: int, pool: Optional[Tuple[int, int]] = None):
        layers: list[nn.Module] = [
            nn.Conv2d(ci, co, kernel_size=3, padding=1),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        ]
        if pool is not None:
            layers.append(nn.MaxPool2d(pool))
        super().__init__(*layers)


class CRNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        stride_h = [(2, 2), (2, 2), (2, 1), (2, 1)]
        ch = [(in_channels, 64), (64, 128), (128, 256), (256, 256)]
        blocks: list[nn.Module] = []
        ci = in_channels
        for (_, co), pt in zip(ch, stride_h):
            blocks.append(ConvBlock(ci, co, pool=pt))
            ci = co
        self.cnn = nn.Sequential(*blocks)

        features_per_step = ci
        self.lstm = nn.LSTM(
            input_size=features_per_step,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=False,
        )
        self.proj = nn.Linear(lstm_hidden * 2, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Args:
            inputs: [B,1,H,W]

        Returns:
            seq: [T, B, C] временная компонента слева‑направо по изображению
            tlens: длины T до паддинга (по каждому примеру)
        """
        b, _, h_pix, widths = inputs.shape
        if h_pix != 32:
            # модель параметрически рассчитана на высоту 32 (как принято в литературе по CRNN);
            raise ValueError(f"ожидается H=32, получено H={h_pix}")
        if int(widths) < MIN_CRNN_RESIZED_WIDTH_PX:
            # Узкие строки/артефакты бокса дают W=1..2 → max_pool2d по ширине даёт 0.
            pad = MIN_CRNN_RESIZED_WIDTH_PX - int(widths)
            inputs = F.pad(inputs, (0, pad), mode="constant", value=-1.0)
        feats = self.cnn(inputs)
        feats = feats.mean(dim=2)  # среднее по вертикали [b, c, w']
        feats = feats.permute(2, 0, 1)
        lengths = torch.full((b,), feats.shape[0], device=inputs.device, dtype=torch.long)
        lengths = lengths.clamp(min=1, max=feats.shape[0])
        return feats, lengths

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Возвращает log_probs [T, B, num_classes]."""
        seq, _ = self.encode(inputs)
        seq, _ = self.lstm(seq)
        logits = self.proj(seq)
        return torch.log_softmax(logits, dim=-1)


def build_crnn(num_classes: int, model_cfg: dict, in_channels: int = 1) -> CRNN:
    return CRNN(
        in_channels=in_channels,
        num_classes=int(num_classes),
        lstm_hidden=int(model_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(model_cfg.get("lstm_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0)),
    )


def predict_greedy(log_probs: torch.Tensor, charset_blank: int = 0) -> list[list[int]]:
    """
    Args:
        log_probs: [T, B, C]
    Returns:
        индексные последовательности без слияния blank (наружнее при необходимости)
    """
    pred = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    return pred
