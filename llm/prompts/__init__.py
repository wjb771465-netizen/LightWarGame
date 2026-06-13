from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def build_diplomat_system_prompt(persona: str = "default") -> str:
    template = (_DIR / "diplomat.txt").read_text(encoding="utf-8")
    persona_text = (_DIR / "personas" / f"{persona}.txt").read_text(encoding="utf-8")
    return template.format(persona=persona_text.strip())
