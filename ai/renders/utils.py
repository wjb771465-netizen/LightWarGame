from __future__ import annotations

import glob
import os
from datetime import datetime


def latest_model_dir(scenario: str) -> str:
    """返回 ai/train/results/<scenario>/ 下按名称最新的 run_* 目录。"""
    pattern = os.path.join("ai", "train", "results", scenario, "run_*")
    dirs = sorted(glob.glob(pattern))
    assert dirs, f"找不到训练结果，请先训练：{os.path.dirname(pattern)}"
    return dirs[-1]


def model_path(model_dir: str) -> str:
    return os.path.join(model_dir, "final")


def render_out_dir(scenario: str) -> str:
    """渲染输出目录，按时间戳区分每次渲染。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("ai", "renders", "results", scenario, ts)
