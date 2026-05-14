"""Общая геометрия линии после нормализованного ресайза (высота = data.img_height)."""

from __future__ import annotations

from typing import Optional

# CRNN: два подряд MaxPool2d(2×2) по ширине (~÷4). При исходной ширине 1–2 px после пулов W=0
# и PyTorch падает: «Output size is too small». 8 px — с запасом на округление.
MIN_CRNN_RESIZED_WIDTH_PX = 8


def clamp_line_width_px(w: float | int, *, max_width: Optional[int] = None) -> int:
    """Гарантия минимума под CRNN; ограничение сверху data.max_width (если задано); снова минимум."""
    ow = max(MIN_CRNN_RESIZED_WIDTH_PX, int(round(float(w))))
    if max_width is not None:
        mw = int(max_width)
        if mw < MIN_CRNN_RESIZED_WIDTH_PX:
            return MIN_CRNN_RESIZED_WIDTH_PX
        ow = min(ow, mw)
    return max(MIN_CRNN_RESIZED_WIDTH_PX, ow)
