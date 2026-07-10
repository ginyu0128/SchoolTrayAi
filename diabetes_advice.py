from __future__ import annotations

from typing import Any

import pandas as pd


HIGH_RISK_FOOD_TYPES = {"rice", "soup", "fried_chicken", "sausage", "tteokgalbi"}
BETTER_FOOD_TYPES = {"vegetable", "kimchi", "fish", "meat"}
ALWAYS_EXCLUDE_FROM_BETTER = {"rice", "soup"}


def build_diabetes_advice(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Build simple diabetes-oriented meal guidance from the result table."""
    food_rows = summary_df[summary_df["cell_id"] != "TOTAL"].copy()
    if food_rows.empty:
        return {
            "risk_items": [],
            "better_items": [],
            "summary": "분석된 음식이 없어 당뇨 관련 식사 조언을 만들 수 없습니다.",
        }

    risk_items = _risk_items(food_rows)
    risk_names = {item["food_name"] for item in risk_items}
    better_items = _better_items(food_rows, excluded_names=risk_names)

    if risk_items:
        risk_text = " ".join(item["message"] for item in risk_items[:3])
    else:
        risk_text = "이번 식판에서는 당뇨 관점에서 특별히 줄여야 할 고위험 항목이 뚜렷하지 않습니다."

    if better_items:
        better_text = " ".join(item["message"] for item in better_items[:2])
    else:
        better_text = "채소류나 단백질 반찬이 부족하면 다음 식사에서 나물, 채소, 생선, 두부, 살코기 반찬을 보충하는 것이 좋습니다."

    return {
        "risk_items": risk_items,
        "better_items": better_items,
        "summary": (
            f"{risk_text} {better_text} "
            "개인 혈당 반응은 다를 수 있으므로 실제 당뇨가 있거나 약을 복용 중이면 의료진 또는 영양사와 상의하세요."
        ),
    }


def _risk_items(food_rows: pd.DataFrame) -> list[dict[str, str]]:
    items = []
    for _, row in food_rows.iterrows():
        food_type = str(row.get("food_type", "unknown"))
        food_name = _food_name(row)
        calories = _float(row.get("calories"))
        carbs = _float(row.get("carbs_g"))
        fat = _float(row.get("fat_g"))
        protein = _float(row.get("protein_g"))

        reasons = []
        if food_type == "rice":
            reasons.append("흰쌀밥류라 식후 혈당을 빠르게 올릴 수 있는 탄수화물 중심 음식입니다")
        if food_type == "soup" and _contains_any(food_name, ["카레", "짜장", "소스"]):
            reasons.append("소스류라 밥과 함께 먹으면 탄수화물과 열량 부담이 함께 커질 수 있습니다")
        if food_type in {"fried_chicken", "sausage", "tteokgalbi"}:
            reasons.append("튀김 또는 가공육 계열이라 지방과 열량 부담이 있습니다")
        if carbs >= 45:
            reasons.append(f"탄수화물이 {carbs:.1f}g로 높은 편입니다")
        if calories >= 350:
            reasons.append(f"칼로리가 {calories:.0f}kcal로 높은 편입니다")
        if fat >= 15 and protein < 10:
            reasons.append("지방은 많은데 단백질은 상대적으로 적습니다")

        if food_type in HIGH_RISK_FOOD_TYPES or reasons:
            items.append(
                {
                    "food_name": food_name,
                    "message": (
                        f"{food_name}은(는) "
                        f"{', '.join(reasons) if reasons else '혈당 부담이 있을 수 있습니다'}. "
                        f"{_risk_action(food_type, food_name, calories, carbs)}"
                    ),
                }
            )

    return sorted(items, key=lambda item: _risk_sort_value(food_rows, item["food_name"]), reverse=True)


def _better_items(food_rows: pd.DataFrame, excluded_names: set[str]) -> list[dict[str, str]]:
    items = []
    for _, row in food_rows.iterrows():
        food_type = str(row.get("food_type", "unknown"))
        food_name = _food_name(row)
        if food_name in excluded_names or food_type in ALWAYS_EXCLUDE_FROM_BETTER:
            continue

        carbs = _float(row.get("carbs_g"))
        protein = _float(row.get("protein_g"))
        calories = _float(row.get("calories"))
        fat = _float(row.get("fat_g"))

        reasons = []
        if food_type in {"vegetable", "kimchi"}:
            reasons.append("채소 반찬이라 탄수화물 부담이 비교적 낮습니다")
        if food_type in {"fish", "meat"}:
            reasons.append("탄수화물보다 단백질 비중이 높아 포만감 유지에 도움이 됩니다")
        if protein >= 10 and carbs <= 20 and fat <= 15:
            reasons.append("단백질은 충분하고 탄수화물 부담은 낮은 편입니다")
        if carbs <= 15 and calories <= 250 and food_type not in ALWAYS_EXCLUDE_FROM_BETTER:
            reasons.append("현재 추정치 기준으로 탄수화물과 열량이 낮은 편입니다")

        if (food_type in BETTER_FOOD_TYPES or reasons) and reasons:
            items.append(
                {
                    "food_name": food_name,
                    "message": f"{food_name}은(는) {', '.join(reasons)}. {_better_action(food_type)}",
                }
            )

    return items


def _risk_action(food_type: str, food_name: str, calories: float, carbs: float) -> str:
    if food_type == "rice":
        return "당뇨가 걱정된다면 밥 양을 조금 덜어 먹고, 채소나 단백질 반찬을 먼저 먹는 편이 낫습니다."
    if food_type == "soup" and _contains_any(food_name, ["카레", "짜장", "소스"]):
        return "밥에 많이 비비기보다 필요한 만큼만 곁들이면 혈당 부담을 줄이는 데 도움이 됩니다."
    if food_type in {"fried_chicken", "sausage", "tteokgalbi"}:
        return "한 번에 많이 먹기보다 몇 조각만 덜어 먹고 채소 반찬과 함께 먹는 쪽이 좋습니다."
    if carbs >= 45:
        return "절반 정도만 먹거나 다른 반찬과 나눠 먹는 방식이 더 안정적입니다."
    if calories >= 350:
        return "이번 식사에서는 우선순위를 낮추고 포만감이 생기면 남기는 편이 좋습니다."
    return "당뇨가 걱정된다면 한 번에 많이 먹기보다 조금씩 나눠 먹는 편이 좋습니다."


def _better_action(food_type: str) -> str:
    if food_type in {"vegetable", "kimchi"}:
        return "식사 초반에 먼저 먹으면 포만감을 만들고 밥 섭취량을 조절하는 데 도움이 됩니다."
    if food_type in {"fish", "meat"}:
        return "밥보다 먼저 챙겨 먹으면 포만감 유지에 도움이 됩니다."
    return "당뇨가 걱정될 때 상대적으로 먼저 먹거나 조금 더 챙겨 먹기 좋은 편입니다."


def _risk_sort_value(food_rows: pd.DataFrame, food_name: str) -> float:
    row = food_rows[food_rows.apply(_food_name, axis=1) == food_name].iloc[0]
    return _float(row.get("calories")) + _float(row.get("carbs_g")) * 5 + _float(row.get("fat_g")) * 2


def _food_name(row: pd.Series) -> str:
    return str(row.get("food_name") or row.get("nutrition_search_name") or "해당 음식")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)
