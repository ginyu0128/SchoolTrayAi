from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Dict, Tuple
from urllib.parse import quote

import requests

from src.food_catalog import get_food_profile, normalize_food_key
from src.settings import get_setting

DATA_GO_KR_DEFAULT_URLS = [
    "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02",
    "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo/getFoodNtrCpntDbInq",
]
FOODSAFETYKOREA_SERVICE_ID = "I2790"
FOODSAFETYKOREA_BASE_URL = "https://openapi.foodsafetykorea.go.kr/api"
UNKNOWN_QUERY = "\uc54c \uc218 \uc5c6\uc74c"
DATA_GO_KR_BACKOFF_SECONDS = 120
_DATA_GO_KR_BACKOFF_UNTIL = 0.0
_DATA_GO_KR_BACKOFF_DETAIL = ""


def get_nutrition_per_100g(food_key: str, search_name: str | None = None) -> Dict[str, Any]:
    """Load nutrition from public APIs, falling back to the local catalog."""
    return dict(_get_nutrition_per_100g_cached(food_key, search_name or "", _cache_token()))


@lru_cache(maxsize=128)
def _get_nutrition_per_100g_cached(
    food_key: str,
    search_name: str = "",
    cache_token: tuple[str, ...] = (),
) -> Tuple[tuple[str, Any], ...]:
    api_result, data_go_kr_detail = _fetch_from_data_go_kr(food_key, search_name=search_name)
    if api_result is not None:
        return tuple(api_result.items())

    api_result, foodsafety_detail = _fetch_from_foodsafetykorea(food_key, search_name=search_name)
    if api_result is not None:
        return tuple(api_result.items())

    api_result, public_detail = _fetch_from_public_api(food_key, search_name=search_name)
    if api_result is not None:
        return tuple(api_result.items())

    nutrition = _local_nutrition(food_key)
    detail = data_go_kr_detail or foodsafety_detail or public_detail or "using local fallback nutrition"
    nutrition["detail"] = _friendly_api_failure_detail(detail)
    return tuple(nutrition.items())


def _cache_token() -> tuple[str, ...]:
    return (
        str(get_setting("DATA_GO_KR_API_KEY") or ""),
        str(get_setting("FOODSAFETYKOREA_API_KEY") or ""),
        str(get_setting("FOODSAFETYKOREA_NATIVE_API_KEY") or ""),
        str(get_setting("NUTRITION_API_URL") or ""),
        str(get_setting("DATA_GO_KR_API_URL") or ""),
        str(get_setting("NUTRITION_API_TIMEOUT_SECONDS", "8")),
        str(get_setting("NUTRITION_API_RETRIES", "2")),
        str(id(requests.get)),
    )


def _fetch_from_data_go_kr(food_key: str, search_name: str | None = None) -> Tuple[Dict[str, Any] | None, str]:
    global _DATA_GO_KR_BACKOFF_DETAIL
    global _DATA_GO_KR_BACKOFF_UNTIL

    api_key = (
        get_setting("DATA_GO_KR_API_KEY")
        or get_setting("FOODSAFETYKOREA_API_KEY")
        or get_setting("NUTRITION_API_KEY")
    )
    if not api_key:
        return None, "DATA_GO_KR_API_KEY or FOODSAFETYKOREA_API_KEY is not set"

    now = time.monotonic()
    if now < _DATA_GO_KR_BACKOFF_UNTIL:
        return None, _DATA_GO_KR_BACKOFF_DETAIL or _friendly_api_failure_detail("data.go.kr is temporarily unavailable")

    timeout = float(get_setting("NUTRITION_API_TIMEOUT_SECONDS", "8") or "8")
    limit = get_setting("DATA_GO_KR_RESULT_LIMIT", "5")
    query_param = get_setting("DATA_GO_KR_QUERY_PARAM", "DESC_KOR")
    details = []
    transient_failure_seen = False

    for api_url in _data_go_kr_urls():
        for query in _query_candidates(food_key, search_name):
            params = {
                "serviceKey": api_key,
                "pageNo": "1",
                "numOfRows": limit,
                "type": "json",
                query_param: query,
            }
            try:
                payload, error_detail = _get_json_payload(api_url, params=params, timeout=timeout, source="data.go.kr")
                if error_detail:
                    details.append(error_detail)
                    if _is_transient_data_go_kr_failure(error_detail):
                        transient_failure_seen = True
                        break
                    continue

                nutrition = _parse_nutrition_payload(payload)
                if nutrition is not None:
                    _DATA_GO_KR_BACKOFF_UNTIL = 0.0
                    _DATA_GO_KR_BACKOFF_DETAIL = ""
                    nutrition["source"] = "data_go_kr_api"
                    nutrition["detail"] = f"data.go.kr query matched: {query}"
                    return nutrition, ""

                api_message = _extract_api_message(payload)
                detail = f"{query}: no nutrition rows"
                if api_message:
                    detail = f"{detail} ({api_message})"
                details.append(detail)
            except requests.RequestException as exc:
                details.append(_request_exception_detail(exc, "data.go.kr"))

    if transient_failure_seen:
        _DATA_GO_KR_BACKOFF_UNTIL = time.monotonic() + DATA_GO_KR_BACKOFF_SECONDS
        _DATA_GO_KR_BACKOFF_DETAIL = _friendly_api_failure_detail("; ".join(details))
        return None, _DATA_GO_KR_BACKOFF_DETAIL

    return None, f"data.go.kr returned no nutrition rows; tried {_shorten_detail('; '.join(details))}"


def _fetch_from_foodsafetykorea(food_key: str, search_name: str | None = None) -> Tuple[Dict[str, Any] | None, str]:
    api_key = get_setting("FOODSAFETYKOREA_NATIVE_API_KEY")
    if not api_key:
        return None, "FOODSAFETYKOREA_NATIVE_API_KEY is not set"

    service_id = get_setting("FOODSAFETYKOREA_SERVICE_ID", FOODSAFETYKOREA_SERVICE_ID)
    base_url = get_setting("FOODSAFETYKOREA_BASE_URL", FOODSAFETYKOREA_BASE_URL)
    limit = get_setting("FOODSAFETYKOREA_RESULT_LIMIT", "5")
    timeout = float(get_setting("NUTRITION_API_TIMEOUT_SECONDS", "8") or "8")
    details = []

    for query in _query_candidates(food_key, search_name):
        encoded_query = quote(str(query))
        url = f"{base_url}/{api_key}/{service_id}/json/1/{limit}/DESC_KOR={encoded_query}"
        try:
            payload, error_detail = _get_json_payload(url, timeout=timeout, source="FoodSafetyKorea")
            if error_detail:
                return None, error_detail
            nutrition = _parse_nutrition_payload(payload)
            if nutrition is not None:
                nutrition["source"] = "foodsafetykorea_api"
                nutrition["detail"] = f"FoodSafetyKorea query matched: {query}"
                return nutrition, ""

            api_message = _extract_api_message(payload)
            detail = f"{query}: no nutrition rows"
            if api_message:
                detail = f"{detail} ({api_message})"
            details.append(detail)
        except requests.RequestException as exc:
            return None, _request_exception_detail(exc, "FoodSafetyKorea")
        except (ValueError, KeyError, TypeError) as exc:
            return None, f"FoodSafetyKorea response parse failed: {type(exc).__name__}"

    return None, f"FoodSafetyKorea returned no nutrition rows; tried {_shorten_detail('; '.join(details))}"


def _fetch_from_public_api(food_key: str, search_name: str | None = None) -> Tuple[Dict[str, Any] | None, str]:
    api_url = get_setting("NUTRITION_API_URL")
    if not api_url:
        return None, "NUTRITION_API_URL is not set"

    query_param = get_setting("NUTRITION_API_QUERY_PARAM", "food_name")
    key_param = get_setting("NUTRITION_API_KEY_PARAM", "serviceKey")
    timeout = float(get_setting("NUTRITION_API_TIMEOUT_SECONDS", "8") or "8")

    for query in _query_candidates(food_key, search_name):
        params = {query_param: query, "type": "json"}
        api_key = get_setting("NUTRITION_API_KEY")
        if api_key:
            params[key_param] = api_key

        try:
            payload, error_detail = _get_json_payload(api_url, params=params, timeout=timeout, source="Public nutrition API")
            if error_detail:
                return None, error_detail
            nutrition = _parse_nutrition_payload(payload)
            if nutrition is not None:
                nutrition["detail"] = f"Public nutrition API query matched: {query}"
                return nutrition, ""
        except requests.RequestException as exc:
            return None, f"Public nutrition API request failed: {type(exc).__name__}"
        except (ValueError, KeyError, TypeError) as exc:
            return None, f"Public nutrition API response parse failed: {type(exc).__name__}"

    return None, "Public nutrition API returned no nutrition rows"


def _parse_nutrition_payload(payload: Any) -> Dict[str, Any] | None:
    record = _first_record(payload)
    if not record:
        return None

    calories = _first_number(record, ["calories", "kcal", "enerc_kcal", "ENERC", "NUTR_CONT1", "AMT_NUM1"])
    carbs = _first_number(record, ["carbs_g", "carbohydrate", "chocdf", "CHOCDF", "NUTR_CONT2", "AMT_NUM7"])
    protein = _first_number(record, ["protein_g", "protein", "prot", "PROT", "NUTR_CONT3", "AMT_NUM3"])
    fat = _first_number(record, ["fat_g", "fat", "fatce", "FATCE", "NUTR_CONT4", "AMT_NUM4"])

    if calories is None:
        return None

    return {
        "calories": calories,
        "carbs_g": carbs or 0.0,
        "protein_g": protein or 0.0,
        "fat_g": fat or 0.0,
        "source": "public_api",
    }


def _first_record(payload: Any) -> Dict[str, Any] | None:
    if isinstance(payload, list):
        return payload[0] if payload and isinstance(payload[0], dict) else None
    if not isinstance(payload, dict):
        return None

    for key in ("row", "items", "item", "data", "results", "records", "body"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return value[0] if isinstance(value[0], dict) else None
        if isinstance(value, dict):
            nested = _first_record(value)
            if nested:
                return nested

    for value in payload.values():
        if isinstance(value, dict):
            nested = _first_record(value)
            if nested:
                return nested
    return payload


def _first_number(record: Dict[str, Any], keys: list[str]) -> float | None:
    normalized = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        value = normalized.get(key.lower())
        if value is None:
            continue
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            continue
    return None


def _local_nutrition(food_key: str) -> Dict[str, Any]:
    food = get_food_profile(normalize_food_key(food_key))
    nutrition = dict(food["nutrition_per_100g"])
    nutrition["source"] = "local_catalog"
    nutrition["detail"] = "using local fallback nutrition"
    return nutrition


def _data_go_kr_urls() -> list[str]:
    configured = get_setting("DATA_GO_KR_API_URL")
    if configured:
        return [url.strip() for url in configured.split(",") if url.strip()]
    return DATA_GO_KR_DEFAULT_URLS


def _clean_query(query: str | None) -> str:
    cleaned = str(query or "").strip()
    return cleaned if cleaned and cleaned.lower() != "unknown" else UNKNOWN_QUERY


def _query_candidates(food_key: str, search_name: str | None = None) -> list[str]:
    food = get_food_profile(food_key)
    candidates = [search_name, food.get("display_name"), *food.get("aliases", []), food_key]
    max_queries = int(get_setting("NUTRITION_API_MAX_QUERY_CANDIDATES", "3") or "3")

    cleaned_candidates = []
    seen = set()
    for candidate in candidates:
        cleaned = _clean_query(candidate)
        if cleaned in seen or cleaned == UNKNOWN_QUERY:
            continue
        seen.add(cleaned)
        cleaned_candidates.append(cleaned)
        if len(cleaned_candidates) >= max_queries:
            break
    return cleaned_candidates or [UNKNOWN_QUERY]


def _extract_api_message(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    result = payload.get("RESULT") or payload.get("response")
    if isinstance(result, dict):
        header = result.get("header")
        if isinstance(header, dict):
            message = header.get("resultMsg") or header.get("resultCode")
            if message:
                return str(message)
        message = result.get("MSG") or result.get("message")
        if message:
            return str(message)

    for key in ("header", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            message = value.get("resultMsg") or value.get("resultCode") or value.get("message")
            if message:
                return str(message)

    for value in payload.values():
        if isinstance(value, dict):
            message = _extract_api_message(value)
            if message:
                return message
    return ""


def _shorten_detail(detail: str, limit: int = 220) -> str:
    return detail if len(detail) <= limit else f"{detail[:limit]}..."


def _is_transient_data_go_kr_failure(detail: str) -> bool:
    transient_markers = (
        "status=500",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectionError",
        "Unexpected errors",
    )
    return any(marker in detail for marker in transient_markers)


def _friendly_api_failure_detail(detail: str) -> str:
    if _is_transient_data_go_kr_failure(detail):
        return "공공데이터 영양 API가 일시적으로 불안정해 임시 로컬 영양값을 사용했습니다."
    if "not set" in detail:
        return "영양 API 키가 설정되지 않아 임시 로컬 영양값을 사용했습니다."
    if "non-JSON" in detail or "invalid service key" in detail:
        return "영양 API 응답을 해석하지 못해 임시 로컬 영양값을 사용했습니다. API 키 형식을 확인하세요."
    return detail


def _get_json_payload(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float,
    source: str,
) -> tuple[Any | None, str]:
    retries = int(get_setting("NUTRITION_API_RETRIES", "2") or "2")
    last_detail = ""
    headers = {"User-Agent": "SchoolTrayAI/1.0"}

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout, headers=headers)
            response.raise_for_status()
            try:
                return response.json(), ""
            except ValueError:
                return None, _non_json_response_detail(response, source)
        except requests.RequestException as exc:
            last_detail = _request_exception_detail(exc, source)
            if _should_retry(exc) and attempt < retries:
                time.sleep(0.35 * (attempt + 1))
                continue
            return None, last_detail

    return None, last_detail or f"{source} request failed"


def _should_retry(exc: requests.RequestException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True

    response = getattr(exc, "response", None)
    if response is None:
        return False
    return int(response.status_code) >= 500


def _request_exception_detail(exc: requests.RequestException, source: str) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        preview = response.text[:160].replace("\n", " ").replace("\r", " ").strip()
        return (
            f"{source} request failed: {type(exc).__name__} "
            f"status={response.status_code}, body={preview}"
        )
    return f"{source} request failed: {type(exc).__name__}"


def _non_json_response_detail(response: requests.Response, source: str) -> str:
    content_type = response.headers.get("content-type", "unknown")
    preview = response.text[:180].replace("\n", " ").replace("\r", " ").strip()
    return (
        f"{source} returned non-JSON response: "
        f"status={response.status_code}, content_type={content_type}, body={preview}"
    )
