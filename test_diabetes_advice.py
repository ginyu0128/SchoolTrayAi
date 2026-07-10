import pandas as pd

from src.diabetes_advice import build_diabetes_advice


def test_build_diabetes_advice_flags_rice_and_fried_foods():
    summary = pd.DataFrame([
        {
            "cell_id": "rice",
            "food_type": "rice",
            "food_name": "흰밥",
            "calories": 460,
            "carbs_g": 60,
            "protein_g": 8,
            "fat_g": 1,
        },
        {
            "cell_id": "side_left",
            "food_type": "fried_chicken",
            "food_name": "치킨너겟",
            "calories": 260,
            "carbs_g": 20,
            "protein_g": 12,
            "fat_g": 14,
        },
        {
            "cell_id": "TOTAL",
            "food_type": "mixed",
            "food_name": "Total",
            "calories": 720,
            "carbs_g": 80,
            "protein_g": 20,
            "fat_g": 15,
        },
    ])

    advice = build_diabetes_advice(summary)

    assert "흰밥" in advice["summary"]
    assert "치킨너겟" in advice["summary"]
    assert "밥 양을 조금 덜어 먹고" in advice["summary"]
    assert "몇 조각만 덜어 먹고" in advice["summary"]


def test_build_diabetes_advice_recommends_vegetables_and_protein():
    summary = pd.DataFrame([
        {
            "cell_id": "side_middle_left",
            "food_type": "vegetable",
            "food_name": "나물",
            "calories": 40,
            "carbs_g": 8,
            "protein_g": 2,
            "fat_g": 1,
        },
        {
            "cell_id": "side_right",
            "food_type": "fish",
            "food_name": "생선구이",
            "calories": 180,
            "carbs_g": 0,
            "protein_g": 22,
            "fat_g": 8,
        },
        {
            "cell_id": "TOTAL",
            "food_type": "mixed",
            "food_name": "Total",
            "calories": 220,
            "carbs_g": 8,
            "protein_g": 24,
            "fat_g": 9,
        },
    ])

    advice = build_diabetes_advice(summary)

    assert "나물" in advice["summary"]
    assert "생선구이" in advice["summary"]
    assert "식사 초반에 먼저 먹으면" in advice["summary"]
    assert "포만감 유지에 도움이 됩니다" in advice["summary"]


def test_build_diabetes_advice_does_not_recommend_rice_as_better_food():
    summary = pd.DataFrame([
        {
            "cell_id": "rice",
            "food_type": "rice",
            "food_name": "흰쌀밥",
            "calories": 461,
            "carbs_g": 1.1,
            "protein_g": 45.5,
            "fat_g": 35,
        },
        {
            "cell_id": "side_middle_left",
            "food_type": "vegetable",
            "food_name": "시금치나물",
            "calories": 50,
            "carbs_g": 7,
            "protein_g": 3,
            "fat_g": 1,
        },
        {
            "cell_id": "TOTAL",
            "food_type": "mixed",
            "food_name": "Total",
            "calories": 511,
            "carbs_g": 8.1,
            "protein_g": 48.5,
            "fat_g": 36,
        },
    ])

    advice = build_diabetes_advice(summary)

    assert "흰쌀밥" in advice["summary"]
    assert any(item["food_name"] == "흰쌀밥" for item in advice["risk_items"])
    assert all(item["food_name"] != "흰쌀밥" for item in advice["better_items"])
    assert "흰쌀밥은(는) 단백질" not in advice["summary"]
