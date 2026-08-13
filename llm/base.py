from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from game.campaign.chat import ChatRoom
from game.datatypes.state import GameState

load_dotenv()

_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-8B"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 128


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class BaseLLMAgent:
    def __init__(self, system_prompt: str, model: str | None = None) -> None:
        self._system_prompt = system_prompt
        self._model = model or os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)
        self._temperature = float(os.getenv("OPENAI_TEMPERATURE", _DEFAULT_TEMPERATURE))
        self._max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", _DEFAULT_MAX_TOKENS))
        self._enable_thinking = _env_bool("OPENAI_ENABLE_THINKING", False)
        # OPENAI_API_KEY / OPENAI_BASE_URL：SDK 原生命名；未设 BASE_URL 时默认硅基流动
        self._client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", _DEFAULT_BASE_URL),
        )

    def _call(self, user_content: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            extra_body={"enable_thinking": self._enable_thinking},
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
