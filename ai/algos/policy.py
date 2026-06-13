from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv


class Policy(Protocol):
    def predict(self, obs: np.ndarray, mask: np.ndarray) -> int: ...


class SB3Policy:
    """MaskablePPO 封装，统一训练与推理接口。

    三种初始化模式均通过关键字参数区分：
        SB3Agent(env=env, args=args, tb_log_dir=...)  # 新训练
        SB3Agent(path=ckpt, env=env)                  # 续训
        SB3Agent(path=ckpt)                           # 推理
    """

    def __init__(
        self,
        *,
        env: Optional[VecEnv] = None,
        args=None,
        path: Optional[str] = None,
        policy_kwargs: Optional[dict] = None,
        tb_log_dir: Optional[str] = None,
    ) -> None:
        if path is not None:
            self._model = MaskablePPO.load(path, env=env)
        else:
            self._model = MaskablePPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs or {"net_arch": args.net_arch},
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                learning_rate=args.lr,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                verbose=1,
                seed=args.seed,
                tensorboard_log=tb_log_dir,
            )
        self._first = path is None

    def predict(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=deterministic)
        return int(action)

    @property
    def obs_dim(self) -> int:
        return int(self._model.observation_space.shape[0])

    def learn(self, steps: int, *, callback: Any = None) -> None:
        self._model.learn(steps, reset_num_timesteps=self._first, callback=callback)
        self._first = False

    def save(self, path: str) -> None:
        self._model.save(path)

    @property
    def num_timesteps(self) -> int:
        return self._model.num_timesteps
