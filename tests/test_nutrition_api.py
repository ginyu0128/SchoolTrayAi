import pytest
import requests

import src.nutrition_api as nutrition_api
from src.nutrition_api import _get_nutrition_per_100g_cached, get_nutrition_per_100g


def _requested_query(params):
    return params.get("FOOD_NM_KR") or params.get("DESC_KOR")


@pytest.fixture(autouse=True)
def clear_nutrition_cache():
    _get_nutrition_per_100g_cached.cache_clear()
    nutrition_api._DATA_GO_KR_BACKOFF_UNTIL = 0.0
    nutrition_api._DATA_GO_KR_BACKOFF_DETAIL = ""
    yield
    _get_nutrition_per_100g_cached.cache_clear()
    nutrition_api._DATA_GO_KR_BACKOFF_UNTIL = 0.0
    nutrition_api._DATA_GO_KR_BACKOFF_DETAIL = ""


def test_get_nutrition_per_100g_falls_back_to_local_catalog(monkeypatch):
    monkeypatch.delenv("DATA_GO_KR_API_KEY", raising=False)
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)

    nutrition = get_nutrition_per_100g("rice")

    assert nutrition["source"] == "local_catalog"
    assert nutrition["detail"]
    assert nutrition["calories"] == 130


def test_get_nutrition_per_100g_uses_latest_data_go_kr_query_param(monkeypatch):
    requested = []

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": {
                    "body": {
                        "items": [
                            {
                                "FOOD_NM_KR": "rice",
                                "AMT_NUM1": "130",
                                "AMT_NUM7": "28",
                                "AMT_NUM3": "2.5",
                                "AMT_NUM4": "0.3",
                            }
                        ]
                    }
                }
            }

    def fake_get(url, params=None, timeout=None, headers=None):
        requested.append((url, params))
        assert timeout == 8
        return FakeResponse()

    monkeypatch.setenv("DATA_GO_KR_API_KEY", "test-key")
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")

    assert "FoodNtrCpntDbInfo02" in requested[0][0]
    assert requested[0][1]["serviceKey"] == "test-key"
    assert requested[0][1]["type"] == "json"
    assert requested[0][1]["FOOD_NM_KR"]
    assert nutrition["source"] == "data_go_kr_api"
    assert nutrition["calories"] == 130
    assert nutrition["carbs_g"] == 28
    assert nutrition["protein_g"] == 2.5
    assert nutrition["fat_g"] == 0.3


def test_get_nutrition_per_100g_uses_search_name_for_data_go_kr(monkeypatch):
    requested = []

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "body": {
                    "items": {
                        "item": {
                            "FOOD_NM_KR": "pork stir fry",
                            "AMT_NUM1": "250",
                            "AMT_NUM7": "5",
                            "AMT_NUM3": "18",
                            "AMT_NUM4": "15",
                        }
                    }
                }
            }

    def fake_get(url, params=None, timeout=None, headers=None):
        requested.append((url, params))
        return FakeResponse()

    monkeypatch.setenv("DATA_GO_KR_API_KEY", "test-key")
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("unknown", search_name="pork stir fry")

    assert requested[0][1]["FOOD_NM_KR"] == "pork stir fry"
    assert nutrition["source"] == "data_go_kr_api"
    assert nutrition["calories"] == 250


def test_get_nutrition_per_100g_retries_aliases(monkeypatch):
    requested_queries = []

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, params=None, timeout=None, headers=None):
        query = _requested_query(params)
        requested_queries.append(query)
        if len(requested_queries) < 2:
            return FakeResponse({"response": {"body": {"items": []}}})
        return FakeResponse({
            "response": {
                "body": {
                    "items": [
                        {
                            "FOOD_NM_KR": "steamed rice",
                            "AMT_NUM1": "145",
                            "AMT_NUM7": "32",
                            "AMT_NUM3": "2.7",
                            "AMT_NUM4": "0.3",
                        }
                    ]
                }
            }
        })

    monkeypatch.setenv("DATA_GO_KR_API_KEY", "test-key")
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice", search_name="plain rice")

    assert requested_queries[:2] == ["plain rice", "밥"]
    assert nutrition["source"] == "data_go_kr_api"
    assert nutrition["calories"] == 145


def test_get_nutrition_per_100g_limits_query_retries(monkeypatch):
    requested_queries = []

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": {"body": {"items": []}}}

    def fake_get(url, params=None, timeout=None, headers=None):
        requested_queries.append(_requested_query(params))
        return FakeResponse()

    monkeypatch.setenv("DATA_GO_KR_API_KEY", "test-key")
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")

    assert len(requested_queries) >= 3
    assert nutrition["source"] == "local_catalog"


def test_get_nutrition_per_100g_reports_data_go_kr_non_json(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html>invalid service key</html>"

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    def fake_get(url, params=None, timeout=None, headers=None):
        return FakeResponse()

    monkeypatch.setenv("DATA_GO_KR_API_KEY", "test-key")
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")

    assert nutrition["source"] == "local_catalog"
    assert nutrition["detail"]
    assert "HTTPError" not in nutrition["detail"]


def test_get_nutrition_per_100g_summarizes_data_go_kr_500(monkeypatch):
    request_count = 0

    class FakeResponse:
        status_code = 500
        headers = {"content-type": "text/plain"}
        text = "Unexpected errors"

        def raise_for_status(self):
            raise requests.HTTPError("server error", response=self)

    def fake_get(url, params=None, timeout=None, headers=None):
        nonlocal request_count
        request_count += 1
        return FakeResponse()

    monkeypatch.setenv("DATA_GO_KR_API_KEY", "test-key")
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")
    second_nutrition = get_nutrition_per_100g("kimchi")

    assert nutrition["source"] == "local_catalog"
    assert second_nutrition["source"] == "local_catalog"
    assert "HTTPError" not in nutrition["detail"]
    assert request_count <= 6


def test_get_nutrition_per_100g_tries_next_data_go_kr_endpoint_after_500(monkeypatch):
    requested_urls = []

    class ErrorResponse:
        status_code = 500
        headers = {"content-type": "text/plain"}
        text = "Unexpected errors"

        def raise_for_status(self):
            raise requests.HTTPError("server error", response=self)

    class SuccessResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": {
                    "body": {
                        "items": [
                            {
                                "DESC_KOR": "rice",
                                "NUTR_CONT1": "130",
                                "NUTR_CONT2": "28",
                                "NUTR_CONT3": "2.5",
                                "NUTR_CONT4": "0.3",
                            }
                        ]
                    }
                }
            }

    def fake_get(url, params=None, timeout=None, headers=None):
        requested_urls.append(url)
        if len(set(requested_urls)) == 1:
            return ErrorResponse()
        return SuccessResponse()

    monkeypatch.setenv("DATA_GO_KR_API_KEY", "test-key")
    monkeypatch.delenv("FOODSAFETYKOREA_API_KEY", raising=False)
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")

    assert nutrition["source"] == "data_go_kr_api"
    assert len(set(requested_urls)) == 2
