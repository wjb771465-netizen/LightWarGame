from typing import List, Dict, Any

class Command:
    def __init__(self, source: int, target: int, troops: int, player: int):
        self.source = source
        self.target = target
        self.troops = troops
        self.player = player


class CommandResult:
    def __init__(self, command: Command, success: bool, reason: str = ""):
        self.command = command
        self.success = success
        self.reason = reason

