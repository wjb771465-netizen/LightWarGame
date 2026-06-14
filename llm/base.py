from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from game.campaign.chat import ChatRoom
from game.datatypes.state import GameState

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


class BaseLLMAgent:
    def __init__(self, system_prompt: str, model: str | None = None) -> None:
        config = _load_config()
        self._system_prompt = system_prompt
        self._model = model or config["default_model"]
        self._client = OpenAI(
            api_key=os.getenv(config["api_key_env"], ""),
            base_url=config["base_url"],
        )

    def _call(self, user_content: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content or ""

    def _render_state(self, state: GameState, player_id: int) -> str:
        # TODO(director): 接入 LLMDirector 后补充兵力、中立区、形势摘要等详细信息
        own, enemy = [], []
        for r in state.game_map.regions[1:]:
            if r is None:
                continue
            cap = "（首都）" if r.is_capital else ""
            if r.owner == player_id:
                own.append(r.name + cap)
            elif r.owner != 0:
                enemy.append(r.name + cap)
        return (
            f"回合：{state.turn}  "
            f"我方：{' '.join(own) or '无'}  "
            f"对方：{' '.join(enemy) or '无'}"
        )

    def _render_chat(self, chat_room: ChatRoom, player_id: int, max_turns: int = 5) -> str:
        if not chat_room._messages:
            return "(无外交记录)"
        cutoff = max(m.turn for m in chat_room._messages) - max_turns + 1
        recent = [m for m in chat_room._messages if m.turn >= cutoff]
        return "\n".join(
            f"[回合{m.turn}] {m.sender_name}{'（我）' if m.sender_id == player_id else ''}: {m.text}"
            for m in reversed(recent)
        )
