from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeTracker:
    """无框架依赖的回合指标统计器。

    用法（任意框架）：
        tracker = EpisodeTracker(window=100)
        tracker.push(win=1.0)       # 每局结束时调用
        tracker.win_rate_global     # -> float | None
        tracker.win_rate_window     # -> float | None（窗口未满返回 None）
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._history: list[float] = []

    def push(self, win: float) -> None:
        self._history.append(win)

    @property
    def win_rate_global(self) -> float | None:
        if not self._history:
            return None
        return sum(self._history) / len(self._history)

    @property
    def win_rate_window(self) -> float | None:
        if len(self._history) < self._window:
            return None
        return sum(self._history[-self._window:]) / self._window

    @property
    def window(self) -> int:
        return self._window

    def reset(self) -> None:
        self._history.clear()


class WinRateCallback(BaseCallback):
    """SB3 适配层：从 episode info 提取数据喂给 EpisodeTracker，并写入 logger。

    要求 VecMonitor 初始化时传入 info_keywords=("win",)，且 env.step() 在
    episode 结束时返回 info["win"] = 1.0（胜）/ 0.0（负/平）。
    """

    def __init__(self, window: int = 100) -> None:
        super().__init__()
        self._tracker = EpisodeTracker(window=window)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None and "win" in ep:
                self._tracker.push(ep["win"])
                if (v := self._tracker.win_rate_global) is not None:
                    self.logger.record("rollout/win_rate_global", v)
                if (v := self._tracker.win_rate_window) is not None:
                    self.logger.record(f"rollout/win_rate_{self._tracker.window}", v)
        return True
