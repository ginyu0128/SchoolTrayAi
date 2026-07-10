from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.cell_splitter import draw_cell_overlay, split_tray_cells
from src.food_area import estimate_food_area_ratio
from src.food_catalog import get_density_g_per_ml, get_food_profile
from src.food_classifier import classify_food
from src.nutrition_api import get_nutrition_per_100g
from src.tray_detector import detect_tray, draw_tray_detection
from src.volume_estimator import (
    estimate_volume_ml,
    get_food_area_cm2,
    infer_flat_piece_cell_ids,
)


def detect_tray_contour(image: np.ndarray) -> List[tuple[int, int, int, int]]:
    detection = detect_tray(image)
    return [detection.bbox] if detection.found else []


def split_cells(image: np.ndarray, rect: tuple[int, int, int, int]) -> List[Dict[str, Any]]:
    return split_tray_cells(rect)


def estimate_area_ratio(image: np.ndarray) -> float:
    return estimate_food_area_ratio(image)


def estimate_volume_from_ratio(cell_spec: Dict[str, Any], ratio: float) -> float:
    return estimate_volume_ml(cell_spec, ratio, volume_profile="standard")


def estimate_weight_from_volume(volume_ml: float, food_type: str) -> float:
    density = get_density_g_per_ml(food_type)
    return volume_ml * density


def estimate_nutrition(
    food_type: str,
    weight_g: float,
    search_name: str | None = None,
) -> Dict[str, float]:
    nutrition = get_nutrition_per_100g(food_type, search_name=search_name)
    grams = max(weight_g, 1.0)
    return {
        "calories": grams * nutrition["calories"] / 100.0,
        "carbs_g": grams * nutrition["carbs_g"] / 100.0,
        "protein_g": grams * nutrition["protein_g"] / 100.0,
        "fat_g": grams * nutrition["fat_g"] / 100.0,
        "nutrition_source": nutrition.get("source", "unknown"),
        "nutrition_detail": nutrition.get("detail", ""),
    }


def estimate_tray_nutrition(image: np.ndarray) -> Dict[str, Any]:
    tray_detection = detect_tray(image)
    tray_rect = tray_detection.bbox

    cells = split_cells(image, tray_rect)
    pending_cells = []
    for cell in cells:
        x, y, w, h = cell["bbox"]
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        ratio = estimate_food_area_ratio(crop, cell_id=cell["cell_id"])
        spec = cell["spec"]
        classification = classify_food(crop, cell_id=cell["cell_id"])

        pending_cells.append({
            "cell": cell,
            "area_ratio": ratio,
            "classification": classification,
        })

    area_ratios = {
        item["cell"]["cell_id"]: item["area_ratio"]
        for item in pending_cells
    }
    flat_piece_cell_ids = infer_flat_piece_cell_ids(area_ratios)

    cell_results = []
    for item in pending_cells:
        cell = item["cell"]
        spec = cell["spec"]
        ratio = item["area_ratio"]
        classification = item["classification"]
        food_type = classification["food_key"]
        food_profile = get_food_profile(food_type)
        raw_food_name = classification.get("raw_food_name") or food_profile["display_name"]
        search_name = classification.get("search_name") or raw_food_name
        if cell["cell_id"] == "rice":
            volume_profile = "rice_mound"
        elif cell["cell_id"] in flat_piece_cell_ids:
            volume_profile = "flat_piece"
        else:
            volume_profile = food_profile["volume_profile"]
        food_area_cm2 = get_food_area_cm2(spec, ratio)
        volume_ml = estimate_volume_ml(spec, ratio, volume_profile)
        weight_g = estimate_weight_from_volume(volume_ml, food_type)
        nutrition = estimate_nutrition(food_type, weight_g, search_name=search_name)

        cell_results.append({
            "cell_id": cell["cell_id"],
            "display_name": cell["display_name"],
            "shape": cell["shape"],
            "bbox": cell["bbox"],
            "food_type": food_type,
            "food_name": raw_food_name,
            "nutrition_search_name": search_name,
            "classification": classification,
            "volume_profile": volume_profile,
            "area_ratio": round(ratio, 4),
            "food_area_cm2": round(food_area_cm2, 2),
            "volume_ml": round(volume_ml, 2),
            "weight_g": round(weight_g, 2),
            "nutrition": nutrition,
        })

    volume_profiles = {
        item["cell_id"]: item["volume_profile"]
        for item in cell_results
    }

    nutrition_summary = []
    for item in cell_results:
        nutrition_summary.append({
            "cell_id": str(item["cell_id"]),
            "cell_name": item["display_name"],
            "food_type": item["food_type"],
            "food_name": item["food_name"],
            "nutrition_search_name": item["nutrition_search_name"],
            "classification_source": item["classification"].get("source", "unknown"),
            "classification_confidence": round(float(item["classification"].get("confidence", 0.0)), 2),
            "classification_reason": item["classification"].get("reason", ""),
            "volume_profile": item["volume_profile"],
            "food_area_cm2": round(item["food_area_cm2"], 2),
            "volume_ml": round(item["volume_ml"], 2),
            "weight_g": round(item["weight_g"], 2),
            "calories": round(item["nutrition"]["calories"], 2),
            "carbs_g": round(item["nutrition"]["carbs_g"], 2),
            "protein_g": round(item["nutrition"]["protein_g"], 2),
            "fat_g": round(item["nutrition"]["fat_g"], 2),
            "nutrition_source": item["nutrition"]["nutrition_source"],
            "nutrition_detail": item["nutrition"]["nutrition_detail"],
        })

    summary_df = pd.DataFrame(nutrition_summary)
    total = {
        "cell_id": "TOTAL",
        "cell_name": "Total",
        "food_type": "mixed",
        "food_name": "Total",
        "nutrition_search_name": "-",
        "classification_source": "-",
        "classification_confidence": None,
        "classification_reason": "-",
        "volume_profile": "-",
        "food_area_cm2": round(float(summary_df["food_area_cm2"].sum()), 2),
        "volume_ml": round(float(summary_df["volume_ml"].sum()), 2),
        "weight_g": round(float(summary_df["weight_g"].sum()), 2),
        "calories": round(float(summary_df["calories"].sum()), 2),
        "carbs_g": round(float(summary_df["carbs_g"].sum()), 2),
        "protein_g": round(float(summary_df["protein_g"].sum()), 2),
        "fat_g": round(float(summary_df["fat_g"].sum()), 2),
        "nutrition_source": "-",
        "nutrition_detail": "-",
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total])], ignore_index=True)

    return {
        "cell_estimates": cell_results,
        "nutrition_summary": summary_df,
        "flat_piece_cells": sorted(flat_piece_cell_ids),
        "tray_detection": {
            "bbox": tray_detection.bbox,
            "found": tray_detection.found,
            "area_ratio": round(tray_detection.area_ratio, 4),
        },
        "debug_images": {
            "tray_detection": draw_tray_detection(image, tray_detection),
            "cell_split": draw_cell_overlay(image, cells, volume_profiles),
        },
    }
