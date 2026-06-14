from __future__ import annotations

from game.campaign.chat import ChatRoom
from game.datatypes.state import GameState
from llm.base import BaseLLMAgent


class LLMDiplomat(BaseLLMAgent):
    def __init__(self, system_prompt: str, model: str | None = None) -> None:
        super().__init__(system_prompt, model)

    def generate_message(self, state: GameState, chat_room: ChatRoom, player_id: int) -> str:
        state_text = self._render_state(state, player_id)
        chat_text = self._render_chat(chat_room, player_id)
        user_content = (
            f"你是玩家{player_id}。\n\n"
            f"=== 当前战局 ===\n{state_text}\n\n"
            f"=== 近期外交记录（最新在前）===\n{chat_text}\n\n"
            f"直接输出你对其他玩家说的外交消息（1-3句，符合你的角色）："
        )
        return self._call(user_content)
