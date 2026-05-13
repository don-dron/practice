from __future__ import annotations

from typing import List, Optional


class Charset:
    """Отображение символ → индекс; индекс 0 зарезервирован под CTC-blank."""

    blank_idx = 0

    def __init__(self, chars: list[str]):
        seen: set[str] = set()
        ordered: list[str] = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        self.itos = ["<blk>"] + ordered
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def from_itos(cls, itos: list[str]) -> Charset:
        if not itos or itos[0] != "<blk>":
            raise ValueError("itos должен начинаться с псевдо-символа <blk>")
        obj = cls.__new__(cls)
        obj.itos = list(itos)
        obj.stoi = {s: i for i, s in enumerate(obj.itos)}
        return obj

    @property
    def num_classes(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        out = []
        for ch in text:
            if ch not in self.stoi:
                raise KeyError(f"символ {ch!r} отсутствует в алфавите модели")
            out.append(self.stoi[ch])
        return out

    def decode_indices(self, indices: list[int], collapse_blank: bool = True) -> str:
        res: list[str] = []
        prev = None
        for idx in indices:
            if idx == self.blank_idx:
                prev = None
                continue
            if collapse_blank and idx == prev:
                continue
            if 0 <= idx < len(self.itos):
                res.append(self.itos[idx])
            prev = idx
        return "".join(res)


def charset_from_strings(texts: List[str], extra: Optional[str] = None) -> Charset:
    s = sorted(set("".join(texts)))
    if extra:
        for ch in sorted(set(extra)):
            if ch not in s:
                s.append(ch)
    return Charset(s)
