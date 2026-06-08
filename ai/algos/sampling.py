"""对手采样策略的纯数学公式，不依赖 PoolEntry 等数据结构。

每个函数接收数值数组，返回概率分布或采样结果。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 概率分布计算
# ---------------------------------------------------------------------------

def logistic_softmax_probs(values: np.ndarray, lam: float, s: float, D: float) -> np.ndarray:
    """Logistic-Softmax: 将任意 scores 转为概率分布。

    Step 1 — Logistic centering（以中位数为基准做尺度归一化）:
        f(x_i) = s / (1 + 10^(-(x_i - median(x)) / D))

    Step 2 — Softmax with adaptive temperature τ = (N+1)/λ:
        P(i) = softmax(λ/(N+1) · f(x))

    Args:
        values: 原始优先级信号 (N,)
        lam: 温度系数 λ，越大分布越尖锐
        s: logistic 缩放因子
        D: logistic 尺度（ELO 标准 = 400）
    Returns:
        probs: 概率分布 (N,), sum=1
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)
    median = np.median(values)
    scores = s / (1.0 + 10.0 ** (-(values - median) / D))
    k = float(n + 1)  # adaptive temperature τ = k / λ
    logits = lam / k * scores
    logits -= np.max(logits)  # numerically stable softmax
    probs = np.exp(logits)
    probs /= np.sum(probs)
    return probs


def uniform_probs(n: int) -> np.ndarray:
    """均匀概率分布。P(i) = 1/n。"""
    if n == 0:
        return np.array([], dtype=np.float64)
    return np.full(n, 1.0 / n, dtype=np.float64)
