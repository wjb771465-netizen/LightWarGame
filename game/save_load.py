"""存档读档：GameState ↔ JSON 文件。"""

from __future__ import annotations

import json
from typing import Any, Dict

from game.datatypes.game_map import GameMap
from game.datatypes.state import GameState


def save_game(state: GameState, path: str = "save.json") -> None:
    regions: list[Any] = [None]
    for r in state.game_map.regions[1:]:
        if r is None:
            regions.append(None)
        else:
            regions.append({
                "owner": r.owner,
                "troops": r.troops,
                "is_capital": r.is_capital,
                "base_growth": r.base_growth,
                "is_special": r.is_special,
                "growth_multiplier": r.growth_multiplier,
            })
    data: Dict[str, Any] = {
        "map_config": state.game_map._config_name,
        "num_players": state.num_players,
        "turn": state.turn,
        "regions": regions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_game(path: str = "save.json") -> GameState:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    m = GameMap(data["map_config"])
    for i, entry in enumerate(data["regions"][1:], 1):
        if entry is None:
            continue
        r = m.regions[i]
        assert r is not None
        r.owner = entry["owner"]
        r.troops = entry["troops"]
        r.is_capital = entry["is_capital"]
        r.base_growth = entry["base_growth"]
        r.is_special = entry["is_special"]
        r.growth_multiplier = entry["growth_multiplier"]
    state = GameState(m, num_players=data["num_players"], turn=data["turn"])
    state.active_players = sorted(
        {r.owner for r in m.regions[1:] if r is not None and 1 <= r.owner <= state.num_players}
    )
    return state
