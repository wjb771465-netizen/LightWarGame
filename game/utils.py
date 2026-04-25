import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_project_root() -> Path:
    """仓库根目录（含 data/ 的 ChineseWarGame 根）。"""
    return Path(__file__).resolve().parent.parent


def parse_map_config(config_name: str) -> Dict[str, Any]:
    """加载 data/map_configs/<config_name>.json（不含扩展名）。"""
    json_path = get_project_root() / "data" / "map_configs" / f"{config_name}.json"
    assert json_path.is_file(), f"Map config not found: {json_path}"
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)
