"""将 GameState 渲染为省份归属地图 PNG（依赖 matplotlib）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from matplotlib import font_manager

from game.datatypes.state import GameState

_SUFFIXES = ["壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "特别行政区", "市", "省"]

_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

_COLORS = {
    0: "#cccccc",
    1: "#e74c3c",
    2: "#3498db",
    3: "#2ecc71",
    4: "#f39c12",
}


def _normalize_name(name: str) -> str:
    for s in _SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def _geojson_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "map" / "china.json"


def render_map(state: GameState, path: str) -> None:
    """渲染当前回合地图并保存到 path。"""
    import matplotlib
    if _CJK_FONT:
        font_manager.fontManager.addfont(_CJK_FONT)
        prop = font_manager.FontProperties(fname=_CJK_FONT)
        matplotlib.rcParams["font.family"] = prop.get_name()

    with open(_geojson_path(), encoding="utf-8") as f:
        geo = json.load(f)

    regions = state.game_map.regions
    name_to_id: Dict[str, int] = {
        regions[i].name: i
        for i in range(1, len(regions))
        if regions[i] is not None
    }

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for feature in geo["features"]:
        raw_name = feature["properties"]["name"]
        short_name = _normalize_name(raw_name)
        rid = name_to_id.get(short_name)
        owner = regions[rid].owner if rid is not None else 0
        color = _COLORS.get(owner, "#888888")

        geom = feature["geometry"]
        polys = (
            geom["coordinates"]
            if geom["type"] == "Polygon"
            else [p[0] for p in geom["coordinates"]]
        )
        for ring in polys:
            coords = np.array(ring)
            if coords.ndim == 2 and len(coords) >= 3:
                patch = MplPolygon(coords, closed=True)
                ax.add_patch(patch)
                patch.set_facecolor(color)
                patch.set_edgecolor("#ffffff")
                patch.set_linewidth(0.4)

        cx = feature["properties"].get("centroid") or feature["properties"].get("center")
        if cx and rid is not None:
            if regions[rid].is_capital:
                ax.plot(cx[0], cx[1], "*", color="gold", markersize=8, zorder=5)
            ax.text(
                cx[0], cx[1], str(rid),
                ha="center", va="center", fontsize=5.5, color="white",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.45, linewidth=0),
                zorder=6,
            )

    ax.autoscale_view()

    handles = [
        mpatches.Patch(color=_COLORS.get(p, "#888"), label=f"玩家{p}")
        for p in range(1, state.num_players + 1)
    ]
    handles.append(mpatches.Patch(color=_COLORS[0], label="中立"))
    ax.legend(handles=handles, loc="lower left", framealpha=0.7,
              facecolor="#2c2c54", labelcolor="white", fontsize=9)

    ax.set_title(f"第 {state.turn} 回合", color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
