"""外交聊天室：ChatMessage + ChatRoom，无 LLM/ML 依赖。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class ChatMessage:
    sender_id: int
    sender_name: str
    text: str
    turn: int


class ChatRoom:
    """游戏内外交消息的中心存储，主循环和 HTTP server 共享同一实例。"""

    def __init__(self) -> None:
        self._messages: List[ChatMessage] = []

    def add_message(self, msg: ChatMessage) -> None:
        self._messages.append(msg)

    def get_history(self, since_turn: int = 0) -> List[ChatMessage]:
        if since_turn <= 0:
            return list(self._messages)
        return [m for m in self._messages if m.turn >= since_turn]

    def get_history_text(self, max_turns: int = 10) -> str:
        """将最近 max_turns 回合的消息格式化为纯文本，供 LLM prompt 注入。"""
        if not self._messages:
            return "(无外交记录)"
        cutoff = max(m.turn for m in self._messages) - max_turns + 1
        recent = [m for m in self._messages if m.turn >= cutoff]
        return "\n".join(
            f"[回合{m.turn}] {m.sender_name}: {m.text}" for m in reversed(recent)
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self._messages], f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self._messages = [ChatMessage(**d) for d in data]
