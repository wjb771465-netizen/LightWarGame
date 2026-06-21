from __future__ import annotations

import logging

from ai.algos.opponent_pool import OpponentPool
from ai.train.sb3_trainer import Sb3Trainer
from ai.train.utils import checkpoint_path, extract_ckpt_step


class SelfPlayTrainer(Sb3Trainer):

    def __init__(self, args) -> None:
        super().__init__(args)
        self._pool = OpponentPool(max_size=self.args.self_play_pool_size)
        self._agent_elo = 1200.0
        self._warmup = True

    def train(self) -> None:
        self._init_logging()

        agent_id: int = self.env.get_attr("agent_id")[0]
        num_players: int = self.env.get_attr("config")[0].game.num_players
        opponent_id = next(p for p in range(1, num_players + 1) if p != agent_id)

        n_opponents = self.args.n_opponents or self.args.n_envs
        n_envs = self.args.n_envs
        if n_envs % n_opponents != 0:
            raise ValueError(
                f"--n-envs ({n_envs}) 须为 --n-opponents ({n_opponents}) 的整数倍"
            )
        per_opponent = n_envs // n_opponents

        warmup_spec = {"type": self.args.self_play_initial_opponent, "player_id": opponent_id}
        self.env.env_method("set_opponent", warmup_spec)
        logging.info("[SelfPlay] 冷启动对手: %s", self.args.self_play_initial_opponent)

        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq
        while self.agent.num_timesteps < total:
            self.agent.learn(min(chunk, total - self.agent.num_timesteps), callback=[self._win_cb])
            self.agent._model._custom_logger = True
            step = self.agent.num_timesteps

            results = self.eval(step) if self.args.use_eval else None

            if self._warmup:
                wr = self._win_cb._tracker.win_rate_window
                if wr is not None and wr >= self.args.curriculum_win_rate:
                    self._warmup = False
                    self.save(step)
                    logging.info("[SelfPlay] 热身结束 step=%d wr=%.1f%%，转入自博弈", step, wr * 100)
                elif wr is not None:
                    logging.info("[SelfPlay] 热身 step=%d wr=%.1f%% (阈值 %.1f%%)",
                                 step, wr * 100, self.args.curriculum_win_rate * 100)
            else:
                prev_elo = self._agent_elo
                evicted, self._agent_elo, accepted = self._pool.add(
                    step, elo=self._agent_elo, outcomes=results)
                if accepted:
                    self.save(step)
                    if evicted is not None:
                        logging.info("[SelfPlay] 池满，淘汰: step=%d", evicted.step)
                    if results is not None:
                        logging.info("[SelfPlay] step=%d, ELO %.1f -> %.1f, 入池",
                                     step, prev_elo, self._agent_elo)
                self.log_eval_metrics({"elo": self._agent_elo}, step)

            specs = self.choose_opponents(n_opponents, opponent_id)
            steps = []
            for i, spec in enumerate(specs):
                indices = list(range(i * per_opponent, (i + 1) * per_opponent))
                self.env.env_method("set_opponent", spec, indices=indices)
                if spec["type"] == "policy":
                    steps.append(int(extract_ckpt_step(spec["path"])))
                else:
                    steps.append(spec["type"])
            steps_str = ", ".join(str(s) for s in steps[:8])
            if len(steps) > 8:
                steps_str += ", ..."
            logging.info("[SelfPlay] step=%d, opponents=[%s], envs=%d, per=%d",
                         step, steps_str, n_envs, per_opponent)

        path = self.save()
        self.render(path, save_dir=self.save_dir)

    def choose_opponents(self, n_types: int, opponent_id: int) -> list[dict]:
        """热身或池空 → 冷启动；否则池采样。"""
        ft = self.args.self_play_initial_opponent

        if self._warmup or len(self._pool) == 0:
            return [{"type": ft, "player_id": opponent_id}] * n_types

        strategy = self.args.pool_sampling_strategy
        lam = self.args.sampling_lam
        scale = self.args.sampling_scale
        specs = []
        for _ in range(n_types):
            if strategy == "uniform":
                entry = self._pool.sample_uniform()
            elif strategy == "progress":
                entry = self._pool.sample_progress(lam=lam, s=scale, D=self.args.progress_D)
            elif strategy == "elo":
                entry = self._pool.sample_elo(lam=lam, s=scale)
            else:
                entry = self._pool.latest()
            if entry is not None:
                specs.append({
                    "type": "policy", "player_id": opponent_id,
                    "path": checkpoint_path(self.save_dir, entry.step),
                })
            else:
                specs.append({"type": ft, "player_id": opponent_id})
        return specs

    def choose_eval_opponents(self, include_fixed: bool = True, region: int | None = None) -> list[dict]:
        n_envs = self.args.eval_n_envs or self.args.n_envs
        n_opps = self.args.eval_n_opponents or self.args.n_opponents or self.args.n_envs
        oid = self._opponent_id()

        specs = self.choose_opponents(n_opps, oid)
        per = max(1, n_envs // n_opps)
        result = []
        for s in specs:
            result.extend([s] * per)
        result = result[:n_envs]

        if include_fixed and self.args.eval_opponent:
            half = max(1, n_envs // 2)
            result = result[:half]
            result += self._fixed_opponent_specs(n_envs - half, oid)
        return result
