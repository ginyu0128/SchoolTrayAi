from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


FOOD_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "food_catalog.json"


@lru_cache(maxsize=1)
def load_food_catalog() -> Dict[str, Any]:
    with FOOD_CATALOG_PATH.open("r", encoding="utf-8") as catalog_file:
        return json.load(catalog_file)


def get_foods() -> List[Dict[str, Any]]:
    return load_food_catalog()["foods"]


def get_food_map() -> Dict[str, Dict[str, Any]]:
    return {food["food_key"]: food for food in get_foods()}


def get_food_keys() -> List[str]:
    return [food["food_key"] for food in get_foods()]


def normalize_food_key(value: str | None) -> str:
    if not value:
        return "unknown"

    lowered = value.strip().lower()
    for food in get_foods():
        names = [
            food["food_key"].lower(),
            str(food.get("display_name", "")).lower(),
            *[alias.lower() for alias in food.get("aliases", [])],
        ]
        if lowered in names:
            return food["food_key"]
    return "unknown"


def get_food_profile(food_key: str) -> Dict[str, Any]:
    return get_food_map().get(food_key, get_food_map()["unknown"])


def get_density_g_per_ml(food_key: str) -> float:
    food = get_food_profile(food_key)
    return float(food.get("density_g_per_ml", load_food_catalog()["default_density_g_per_ml"]))
