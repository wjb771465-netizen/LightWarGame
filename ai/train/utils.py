from __future__ import annotations

import os
import random
from datetime import datetime

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_save_dir(scenario: str, save_dir: str | None = None) -> str:
    """训练结果根目录。不传 save_dir 则自动生成带时间戳的目录。"""
    if save_dir is not None:
        return save_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("ai", "train", "results", scenario, f"run_{ts}")


def checkpoint_path(save_dir: str, step: int) -> str:
    """checkpoint 路径（不含 .zip 后缀）。"""
    return os.path.join(save_dir, f"ckpt_{step}")


def final_model_path(save_dir: str) -> str:
    """最终模型路径（不含 .zip 后缀）。"""
    return os.path.join(save_dir, "final")


def region_dir(save_dir: str, region_id: int) -> str:
    """地区自博弈的子目录。"""
    return os.path.join(save_dir, f"region_{region_id}")


def extract_ckpt_step(path: str) -> str:
    """从 checkpoint 路径中提取 step 号字符串。"""
    return path.rsplit("ckpt_", 1)[-1]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_eval_specs(specs: list[dict]) -> str:
    """将 eval spec 列表格式化为可读的对手摘要字符串。"""
    def _label(s: dict) -> str:
        if s["type"] == "policy":
            step = extract_ckpt_step(s.get("path", ""))
            return f"s{step}"
        return s["type"]
    return ", ".join(_label(s) for s in specs[:8]) + (", ..." if len(specs) > 8 else "")
