from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any, Dict

import cv2
import numpy as np
import requests
from PIL import Image

from src.food_catalog import get_food_keys, get_food_profile, normalize_food_key
from src.settings import get_setting


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def classify_food(image: np.ndarray, cell_id: str | None = None) -> Dict[str, Any]:
    """Classify food with OpenAI Vision when configured, otherwise fallback."""
    api_key = get_setting("OPENAI_API_KEY")
    if api_key:
        return _classify_with_openai(image, cell_id, api_key)
    return _classify_with_rules(image, cell_id)


def _classify_with_openai(image: np.ndarray, cell_id: str | None, api_key: str) -> Dict[str, Any]:
    image_data_url = _image_to_data_url(image)
    candidates = ", ".join(get_food_keys())
    cell_hint = _get_cell_hint(cell_id)
    prompt = (
        "너는 한국 학교 급식판 사진의 한 칸 이미지를 보고 음식명을 분류하는 모델이다. "
        f"칸 ID: {cell_id or 'unknown'}. "
        f"{cell_hint} "
        "가능하면 구체적인 한국어 음식명으로 판단해라. "
        f"food_key는 다음 내부 후보 중 가장 가까운 값으로 골라라: {candidates}. "
        "search_name은 공공데이터포털 식품영양성분 DB에서 검색하기 좋은 한국어 음식명으로 작성해라. "
        "자주 나오는 후보는 밥, 카레, 국, 김치, 치킨너겟, 떡갈비, 소시지볶음, 어묵, 오징어, 과일, 채소, 김, 계란, 두부, 고기반찬이다. "
        "눈에 보이는 음식이 있으면 unknown을 쓰지 말고 가장 가까운 음식명으로 추정해라. "
        "unknown은 빈 칸, 음식이 아닌 물체, 정말 식별 불가능한 경우에만 사용해라. "
        "reason은 반드시 한국어 한 문장으로 써라. "
        "JSON만 반환해라. 키는 raw_food_name, food_key, search_name, confidence, reason만 사용해라."
    )

    payload = {
        "model": get_setting("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=12)
        response.raise_for_status()
        text = _extract_response_text(response.json())
        parsed = _parse_openai_json(text)
        raw_food_name = str(parsed.get("raw_food_name") or parsed.get("search_name") or "").strip()
        search_name = str(parsed.get("search_name") or raw_food_name or parsed.get("food_key") or "unknown").strip()
        food_key = _resolve_food_key(
            parsed.get("food_key"),
            raw_food_name=raw_food_name,
            search_name=search_name,
            cell_id=cell_id,
        )
        if cell_id in {"rice", "soup"} and food_key == "unknown":
            fallback = _classify_with_rules(image, cell_id)
            fallback["source"] = "openai_vision_cell_fallback"
            fallback["confidence"] = max(float(parsed.get("confidence", 0.0)), fallback["confidence"])
            fallback["reason"] = (
                f"OpenAI returned unknown for fixed {cell_id} cell; "
                f"raw={raw_food_name or 'unknown'}; reason={parsed.get('reason', '')}"
            )
            return fallback

        if food_key == "unknown" and _has_visible_food(image):
            food_key = _fallback_food_key_for_cell(cell_id)
            if search_name.lower() == "unknown":
                search_name = get_food_profile(food_key)["display_name"]
            raw_food_name = raw_food_name if raw_food_name.lower() != "unknown" else search_name

        return {
            "food_key": food_key,
            "raw_food_name": raw_food_name or get_food_profile(food_key)["display_name"],
            "search_name": search_name or get_food_profile(food_key)["display_name"],
            "confidence": float(parsed.get("confidence", 0.0)),
            "source": "openai_vision" if food_key != "unknown" else "openai_vision_uncertain",
            "reason": str(parsed.get("reason", "")),
        }
    except Exception as exc:
        return {
            "food_key": "unknown",
            "raw_food_name": "unknown",
            "search_name": "unknown",
            "confidence": 0.0,
            "source": "openai_vision_failed",
            "reason": _format_openai_error(exc),
        }


def _format_openai_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        body = exc.response.text[:300].replace("\n", " ")
        return f"OpenAI HTTP {exc.response.status_code}: {body}"
    return f"OpenAI classification failed: {type(exc).__name__}"


def _get_cell_hint(cell_id: str | None) -> str:
    if cell_id == "rice":
        return "이 칸은 밥칸이며 보통 흰밥이다."
    if cell_id == "soup":
        return "이 칸은 국그릇이며 국, 찌개, 카레 같은 액상 음식을 분류한다."
    if cell_id in {"side_left", "side_right"}:
        return "이 칸은 반찬칸이며 넓은 고기반찬, 소시지, 튀김류가 자주 나온다."
    if cell_id in {"side_middle_left", "side_middle_right"}:
        return "이 칸은 작은 반찬칸이며 김치, 과일, 김, 채소, 작은 소시지 조각이 자주 나온다."
    return "보이는 음식의 형태와 급식판 맥락을 함께 사용한다."


def _resolve_food_key(food_key: str | None, raw_food_name: str, search_name: str, cell_id: str | None = None) -> str:
    """Map model output to an internal food key using every useful text field."""
    for candidate in (food_key, raw_food_name, search_name):
        normalized = normalize_food_key(candidate)
        if normalized != "unknown":
            return normalized

    combined = " ".join(
        part.strip().lower()
        for part in (str(food_key or ""), raw_food_name, search_name)
        if part and str(part).strip()
    )
    inferred = _infer_food_key_from_text(combined)
    if inferred != "unknown":
        return inferred
    return "unknown"


def _infer_food_key_from_text(text: str) -> str:
    if not text or text == "unknown":
        return "unknown"

    keyword_groups = [
        ("rice", ["rice", "\ubc25", "\ubc31\ubbf8", "\uc300\ubc25", "\ud770\ubc25"]),
        ("soup", ["soup", "\uad6d", "\ud0d5", "\ucc0c\uac1c", "\uce74\ub808", "curry", "sauce"]),
        ("kimchi", ["kimchi", "\uae40\uce58", "\uae4d\ub450\uae30"]),
        ("sausage", ["sausage", "\uc18c\uc2dc\uc9c0", "\uc18c\uc138\uc9c0", "\ube44\uc5d4\ub098", "\ud584"]),
        ("fried_chicken", ["fried", "\ud280\uae40", "\uce58\ud0a8", "\ub108\uac9f", "nugget", "\ud0d5\uc218\uc721"]),
        ("tteokgalbi", ["\ub5a1\uac08\ube44", "\ub3d9\uadf8\ub791\ub561", "\ub108\ube44\uc544\ub2c8", "\uc804"]),
        ("fruit", ["fruit", "\uacfc\uc77c", "\ubcf5\uc22d\uc544", "\uc0ac\uacfc", "\ubc30", "\uade4", "\ud30c\uc778\uc560\ud50c", "peach", "apple"]),
        ("fish", ["fish", "\uc0dd\uc120", "\uc624\uc9d5\uc5b4", "squid", "\uc5b4\ubb35", "\ud574\uc0b0\ubb3c"]),
        ("vegetable", ["vegetable", "\ucc44\uc18c", "\uc57c\ucc44", "\ub098\ubb3c", "\ubb34\uce68", "\uc624\uc774", "\ucf69\ub098\ubb3c", "\uc2dc\uae08\uce58", "\uae40", "seaweed"]),
        ("meat", ["meat", "\uace0\uae30", "\ub3fc\uc9c0", "\ubd88\uace0\uae30", "\uc81c\uc721", "\ub2ed", "pork", "beef", "chicken"]),
    ]
    for key, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return key
    return "unknown"


def _fallback_food_key_for_cell(cell_id: str | None) -> str:
    if cell_id == "rice":
        return "rice"
    if cell_id == "soup":
        return "soup"
    return "vegetable"


def _has_visible_food(image: np.ndarray) -> bool:
    if image.size == 0:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))
    value = float(np.mean(hsv[:, :, 2]))
    return saturation > 20 or value < 230


def _classify_with_rules(image: np.ndarray, cell_id: str | None) -> Dict[str, Any]:
    if cell_id == "rice":
        return _result("rice", 0.75, "rule_based", "rice cell default")
    if cell_id == "soup":
        return _result("soup", 0.75, "rule_based", "soup cell default")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_value = float(np.mean(hsv[:, :, 2]))
    if mean_value < 80:
        return _result("kimchi", 0.35, "rule_based", "dark crop fallback")
    return _result("unknown", 0.2, "rule_based", "no API key or uncertain crop")


def _result(food_key: str, confidence: float, source: str, reason: str) -> Dict[str, Any]:
    food = get_food_profile(food_key)
    return {
        "food_key": food_key,
        "raw_food_name": food["display_name"],
        "search_name": food["display_name"],
        "confidence": confidence,
        "source": source,
        "reason": reason,
    }


def _image_to_data_url(image: np.ndarray) -> str:
    buffer = BytesIO()
    pil_image = Image.fromarray(image)
    pil_image.thumbnail((512, 512))
    pil_image.save(buffer, format="JPEG", quality=75, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_response_text(payload: Dict[str, Any]) -> str:
    if "output_text" in payload:
        return str(payload["output_text"]).strip()

    parts = []
    for output in payload.get("output", []):
        for content in output.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text", "")))
    return "".join(parts).strip()


def _parse_openai_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object even if the model wrapped it in markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            preview = cleaned[:160].replace("\n", " ")
            raise ValueError(f"OpenAI response did not contain JSON: {preview}") from None
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("OpenAI response JSON was not an object")
    return parsed
