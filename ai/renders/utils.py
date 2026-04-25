from __future__ import annotations

import glob
import os
import subprocess
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


def make_video(png_dir: str, video_path: str, fps: int) -> None:
    """将 png_dir 下的帧按名称排序合成视频；fps=0 时跳过。"""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", os.path.join(png_dir, "turn_*.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg 合成视频失败（png_dir={png_dir}, video_path={video_path}）。\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e
