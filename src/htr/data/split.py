from __future__ import annotations

import random
from typing import TypeVar

from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)


class Subset(Dataset[T_co]):
    """Тонкая обёртка над индексами без копирования базового набора."""

    def __init__(self, ds: Dataset, indices: list[int]):
        self.ds = ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.ds[self.indices[i]]


def random_split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    nv = max(1, min(n - 1, int(round(n * val_fraction)))) if n > 5 else max(1, n // 5) if n > 2 else 0
    val = idx[:nv]
    train = idx[nv:]
    if not train:
        train = val
        val = []
    return train, val


def texts_for_charset_from_coco(samples: list[tuple[str, str, tuple]], train_idx: list[int]) -> list[str]:
    return [samples[i][1] for i in train_idx]

