"""训练冒烟测试：全流程覆盖——训练、保存、评估、渲染，检查梯度/参数/产物。

用法:
    conda run -n chinese_war_game python -m tests.smoke.train \\
      --scenario duel/vsbaseline_no_adj --use-gnn

    conda run -n chinese_war_game python -m tests.smoke.train \\
      --scenario duel/vsbaseline

输出: ai/train/results/<scenario>/smoke_<timestamp>.log
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

from ai.train.args import get_config
from ai.train.sb3_trainer import Sb3Trainer

# ── 写死冒烟超参（覆盖命令行传参）──────────────────────────────────────────
SMOKE_DEFAULTS = dict(
    total_timesteps=32768,
    n_steps=2048,
    batch_size=512,
    n_epochs=40,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.97,
    clip_range=0.2,
    seed=42,
    n_envs=1,
    checkpoint_freq=16384,
    use_eval=True,
    wandb=False,
    save_dir=None,
    eval_opponent="random",
    eval_opponent_path="",
    win_rate_window=200,
    eval_opponent_freq=1,
    eval_episodes=4,
    eval_n_envs=1,
    exp_name=None,
    wandb_project=None,
    resume_from=None,
)

# ── 模块级日志 ─────────────────────────────────────────────────────────────
_LOG_STREAM: io.StringIO | None = None
_LOG_PATH: str | None = None

# ── 全局状态（hooks 写入）──────────────────────────────────────────────────
_FWD_STATS: dict[str, list[float]] = {}
_BWD_STATS: dict[str, list[float]] = {}
_BWD_HANDLES: list = []
_REF_SNAPSHOT: dict[str, float] | None = None
_VERDICT: list[str] = []  # 收集 FAIL 条目


def _start_log(filepath: str) -> None:
    global _LOG_STREAM, _LOG_PATH
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    _LOG_STREAM = io.StringIO()
    _LOG_PATH = filepath


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG_STREAM is not None:
        _LOG_STREAM.write(msg + "\n")


def _stop_log() -> None:
    global _LOG_STREAM, _LOG_PATH
    if _LOG_STREAM is not None and _LOG_PATH is not None:
        with open(_LOG_PATH, "a") as f:
            f.write(_LOG_STREAM.getvalue())
    _LOG_STREAM = None


def _fail(reason: str) -> None:
    _VERDICT.append(reason)
    _log(f"  FAIL: {reason}")


# ── 参数树 / 快照 ─────────────────────────────────────────────────────────

def _param_tree(module: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {}
    for name, p in module.named_parameters():
        if name.startswith("features_extractor.backbone."):
            key = "gnn." + name.removeprefix("features_extractor.backbone.")
        elif name.startswith("features_extractor."):
            key = "extractor." + name.removeprefix("features_extractor.")
        elif name.startswith("mlp_extractor."):
            key = "mlp." + name.removeprefix("mlp_extractor.")
        elif name.startswith("action_net."):
            key = "action." + name.removeprefix("action_net.")
        elif name.startswith("value_net."):
            key = "value." + name.removeprefix("value_net.")
        else:
            key = "other." + name
        groups.setdefault(key, []).append(p)
    return groups


def _snapshot(params: dict[str, list[nn.Parameter]]) -> dict[str, float]:
    def _norm(x):
        return float(x.data.norm().item())
    return {k: sum(_norm(p) for p in plist) for k, plist in params.items()}


def _delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: abs(after[k] - before[k]) for k in before}


# ── forward hooks ──────────────────────────────────────────────────────────

def _fwd_hook(name: str):
    def hook(_module, _input, output):
        if isinstance(output, torch.Tensor):
            _FWD_STATS.setdefault(name, []).append(float(output.detach().std().item()))
    return hook


def _install_fwd_hooks(module: nn.Module) -> list:
    handles = []
    for n, m in module.named_modules():
        if n == "":
            continue
        if any(k in n for k in ("conv1", "conv2", "norm1", "norm2", "head", "res_proj",
                                  "action_net", "value_net")):
            handles.append(m.register_forward_hook(_fwd_hook(n)))
    return handles


# ── backward hooks ─────────────────────────────────────────────────────────

def _bwd_hook(name: str):
    def hook(grad):
        if grad is not None:
            _BWD_STATS.setdefault(name, []).append(float(grad.norm().item()))
    return hook


def _install_bwd_hooks(module: nn.Module) -> None:
    _BWD_STATS.clear()
    _BWD_HANDLES.clear()
    for n, p in module.named_parameters():
        if p.requires_grad:
            _BWD_HANDLES.append(p.register_hook(_bwd_hook(n)))


def _remove_bwd_hooks() -> None:
    for h in _BWD_HANDLES:
        h.remove()
    _BWD_HANDLES.clear()


# ── callback ───────────────────────────────────────────────────────────────

class SmokeCallback(BaseCallback):
    """每个 rollout 结束后记录训练指标。"""

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass  # 指标通过 _install_bwd_hooks 收集，callback 仅占位


# ── 诊断阶段 ───────────────────────────────────────────────────────────────

def _phase1_structure(model) -> None:
    """Phase 1: 检查 optimizer 是否包含所有参数。"""
    _log("\n── Phase 1: Optimizer check ──")
    try:
        opt = model.policy.optimizer
    except AttributeError:
        _fail("model.policy.optimizer not found")
        return

    opt_ids = set()
    for pg in opt.param_groups:
        for p in pg["params"]:
            opt_ids.add(id(p))

    all_params = list(model.policy.parameters())
    missing = [p for p in all_params if id(p) not in opt_ids]
    if missing:
        _fail(f"{len(missing)}/{len(all_params)} params NOT in optimizer")
        for p in missing[:5]:
            _log(f"    shape={tuple(p.shape)} requires_grad={p.requires_grad}")
    else:
        _log(f"  ✓ all {len(all_params)} params in optimizer")


def _phase2_forward(model, env) -> None:
    """Phase 2: 前向激活检查——有无零方差层。"""
    _log("\n── Phase 2: Forward activation check ──")
    _FWD_STATS.clear()

    policy = model.policy
    handles = _install_fwd_hooks(policy)
    obs = env.reset()
    for _ in range(256):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, _ = env.step(action)
        if np.any(dones):
            obs = env.reset()
    for h in handles:
        h.remove()

    if not _FWD_STATS:
        _fail("no forward activation stats collected")
        return

    dead = [n for n, v in _FWD_STATS.items() if np.array(v).mean() < 1e-8]
    if dead:
        for n in dead:
            _fail(f"zero-variance layer: {n}")
    else:
        _log(f"  ✓ {len(_FWD_STATS)} layers, all have non-zero variance")


def _phase3_diagnostics(model) -> None:
    """Phase 3: 梯度 + 参数变化检查。"""
    # 3a. 梯度
    _log("\n── 3a. Gradient check ──")
    if not _BWD_STATS:
        _fail("no backward gradient stats collected")
    else:
        zero_grads = [n for n, v in _BWD_STATS.items() if np.array(v).mean() < 1e-15]
        if zero_grads:
            for n in zero_grads:
                _fail(f"zero gradient: {n}")
        else:
            n_params = len(_BWD_STATS)
            _log(f"  ✓ all {n_params} params received non-zero gradients")

    # 3b. 参数变化
    _log("\n── 3b. Parameter Δ check ──")
    cur = _snapshot(_param_tree(model.policy))
    if _REF_SNAPSHOT is None:
        _fail("no reference snapshot")
        return

    d = _delta(_REF_SNAPSHOT, cur)
    unchanged = [k for k, v in d.items() if v < 1e-12]
    if unchanged:
        for k in unchanged:
            _fail(f"zero parameter change: {k}")
    else:
        _log(f"  ✓ all {len(d)} param groups changed")

    # 明细
    for k in sorted(d):
        _log(f"    {k:40s} Δ={d[k]:.6e}")


def _phase4_artifacts(save_dir: str) -> None:
    """Phase 4: 检查保存、评估、渲染产物。"""
    from ai.train.utils import checkpoint_path, final_model_path

    _log("\n── Phase 4: Artifact check (save + eval + render) ──")

    # 4a. checkpoint
    freq = SMOKE_DEFAULTS["checkpoint_freq"]
    total = SMOKE_DEFAULTS["total_timesteps"]
    for step in range(freq, total + 1, freq):
        ckpt = checkpoint_path(save_dir, step) + ".zip"
        if os.path.isfile(ckpt):
            _log(f"  ✓ checkpoint: {os.path.relpath(ckpt)}")
        else:
            _fail(f"checkpoint missing: {os.path.relpath(ckpt)}")

    # 4b. final model
    final = final_model_path(save_dir) + ".zip"
    if os.path.isfile(final):
        _log(f"  ✓ final model: {os.path.relpath(final)}")
    else:
        _fail(f"final model missing: {os.path.relpath(final)}")

    # 4c. tensorboard log
    tb_dirs = [d for d in os.listdir(save_dir) if d.startswith("MaskablePPO")]
    if tb_dirs:
        _log(f"  ✓ tensorboard log: {tb_dirs[0]}")
    else:
        _fail("tensorboard log missing")

    # 4d. render videos (nested under eval_videos/<type>/ep00/)
    videos_dir = os.path.join(save_dir, "eval_videos")
    if not os.path.isdir(videos_dir):
        _fail(f"render videos dir missing: {os.path.relpath(videos_dir)}")
        return

    for opp_type in ["random"]:
        opp_dir = os.path.join(videos_dir, opp_type)
        if not os.path.isdir(opp_dir):
            _fail(f"render video dir missing: {os.path.relpath(opp_dir)}")
            continue
        mp4s = []
        for root, _dirs, files in os.walk(opp_dir):
            mp4s.extend(f for f in files if f.endswith(".mp4"))
        if mp4s:
            _log(f"  ✓ render video vs {opp_type}: {len(mp4s)} mp4")
        else:
            _fail(f"no mp4 found in {os.path.relpath(opp_dir)}")


def _verdict() -> None:
    _log(f"\n{'='*60}")
    if _VERDICT:
        _log(f"SMOKE TEST FAILED ({len(_VERDICT)} issues)")
        for v in _VERDICT:
            _log(f"  - {v}")
        _log(f"{'='*60}")
        sys.exit(1)
    else:
        _log("SMOKE TEST PASSED")
        _log(f"{'='*60}")


# ── SmokeTrainer ───────────────────────────────────────────────────────────

class SmokeTrainer(Sb3Trainer):
    """继承 Sb3Trainer，超参写死，跑全流程诊断后判定 PASS/FAIL。"""

    def __init__(self, args) -> None:
        for k, v in SMOKE_DEFAULTS.items():
            setattr(args, k, v)
        super().__init__(args)

    def train(self) -> None:
        global _REF_SNAPSHOT

        model = self.agent._model
        env = self.env

        _phase1_structure(model)
        _phase2_forward(model, env)

        # 训前快照 + backward hooks
        _REF_SNAPSHOT = _snapshot(_param_tree(model.policy))
        _install_bwd_hooks(model.policy)

        # 全流程训练：save + eval + render
        _log("\n── Phase 3: Training loop (save + eval + render) ──")
        self._init_logging()
        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq
        while self.agent.num_timesteps < total:
            steps = min(chunk, total - self.agent.num_timesteps)
            self.agent.learn(steps, callback=[self._win_cb])
            self.agent._model._custom_logger = True
            step = self.agent.num_timesteps
            self.save(step)
            if self.args.use_eval:
                self.eval(step)
        path = self.save()
        self.render(path, save_dir=self.save_dir)

        _remove_bwd_hooks()

        _phase3_diagnostics(model)
        _phase4_artifacts(self.save_dir)

        _verdict()
        _stop_log()
        env.close()


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = get_config()
    args = parser.parse_args()

    _start_log(_log_path(args))
    _log(f"# Smoke test log — {datetime.now().isoformat()}")
    _log(f"# Scenario: {args.scenario}  use_gnn: {args.use_gnn}")
    _log(f"# PID: {os.getpid()}\n")

    SmokeTrainer(args).train()


def _log_path(args) -> str:
    from ai.train.utils import resolve_save_dir
    d = resolve_save_dir(args.scenario, args.save_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(d, f"smoke_{ts}.log")


if __name__ == "__main__":
    main()
