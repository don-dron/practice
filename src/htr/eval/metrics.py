"""Инструментарий метрик качества символьного уровня (набросок протокола §2.4 без весов)."""

from __future__ import annotations


def lev_ratio(hyp: str, ref: str) -> tuple[int, float]:
    """Подсчитывает расстояние Левенштейна между строками символов без нормализации пробелов."""

    dp = [[0] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i in range(len(hyp) + 1):
        dp[i][0] = i
    for j in range(len(ref) + 1):
        dp[0][j] = j
    for i in range(1, len(hyp) + 1):
        for j in range(1, len(ref) + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    err = dp[len(hyp)][len(ref)]
    denom = max(1, len(ref))
    return err, err / denom
