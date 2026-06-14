"""游戏初始化：返回可直接传给 GameRunner 的 GameState。"""

from __future__ import annotations

import random
from pathlib import Path

import yaml  # type: ignore

from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState
from game.campaign.save_load import load_game

NUM_PLAYERS = 2
MAP_CONFIG = "cn"

CAMPAIGN_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = CAMPAIGN_DIR / "sessions"


def random_capitals(num_players: int = NUM_PLAYERS, map_config: str = MAP_CONFIG) -> GameState:
    """随机选取首都，各玩家起点80兵。"""
    m = GameMap(map_config)
    region_ids = [i for i in range(1, len(m.regions)) if m.valid_id(i)]
    capitals = random.sample(region_ids, num_players)
    m.assign_capitals(capitals)
    return GameState(m, num_players=num_players)


def fixed_capitals(capitals: list[int], map_config: str = MAP_CONFIG) -> GameState:
    """指定首都 id 列表开局，capitals[i] 对应玩家 i+1。"""
    m = GameMap(map_config)
    m.assign_capitals(capitals)
    return GameState(m, num_players=len(capitals))


def load_session_config(session_dir: Path) -> dict:
    with open(session_dir / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def from_session(session_dir: Path, save_path: Path | None = None) -> GameState:
    """按 config.yaml 初始化新局；save_path 不为 None 且存在时读档。
    随机首都时将实际 ID 写回 config.yaml 以保证后续读档一致。
    """
    if save_path is not None and save_path.exists():
        return load_game(str(save_path))
    cfg = load_session_config(session_dir)
    capitals = cfg.get("capitals", "random")
    num_players = cfg["num_players"]
    if capitals == "random":
        state = random_capitals(num_players=num_players)
        cfg["capitals"] = list(state.game_map.capitals)
        with open(session_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True)
        return state
    return fixed_capitals(list(capitals))


def list_sessions() -> list[Path]:
    """返回 sessions/ 下所有含 config.yaml 的 session 目录，按名称排序。"""
    if not SESSIONS_DIR.exists():
        return []
    return sorted(
        p for p in SESSIONS_DIR.iterdir()
        if p.is_dir() and (p / "config.yaml").exists()
    )
