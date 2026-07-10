from urllib.parse import unquote

import pytest
import requests

import src.nutrition_api as nutrition_api
from src.nutrition_api import _get_nutrition_per_100g_cached, get_nutrition_per_100g


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


def test_get_nutrition_per_100g_uses_data_go_kr_api(monkeypatch):
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
                                "DESC_KOR": "밥",
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
        requested.append((url, params))
        assert timeout == 8
        return FakeResponse()

    monkeypatch.setenv("FOODSAFETYKOREA_API_KEY", "test-key")
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")

    assert "apis.data.go.kr" in requested[0][0]
    assert requested[0][1]["serviceKey"] == "test-key"
    assert requested[0][1]["type"] == "json"
    assert requested[0][1]["DESC_KOR"] == "밥"
    assert nutrition["source"] == "data_go_kr_api"
    assert nutrition["detail"] == "data.go.kr query matched: 밥"
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
                            "DESC_KOR": "제육볶음",
                            "NUTR_CONT1": "250",
                            "NUTR_CONT2": "5",
                            "NUTR_CONT3": "18",
                            "NUTR_CONT4": "15",
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

    nutrition = get_nutrition_per_100g("unknown", search_name="제육볶음")

    assert requested[0][1]["DESC_KOR"] == "제육볶음"
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
        requested_queries.append(params["DESC_KOR"])
        if params["DESC_KOR"] != "쌀밥":
            return FakeResponse({"response": {"body": {"items": []}}})
        return FakeResponse({
            "response": {
                "body": {
                    "items": [
                        {
                            "DESC_KOR": "쌀밥",
                            "NUTR_CONT1": "145",
                            "NUTR_CONT2": "32",
                            "NUTR_CONT3": "2.7",
                            "NUTR_CONT4": "0.3",
                        }
                    ]
                }
            }
        })

    monkeypatch.setenv("FOODSAFETYKOREA_API_KEY", "test-key")
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")

    assert requested_queries == ["밥", "쌀밥"]
    assert nutrition["source"] == "data_go_kr_api"
    assert nutrition["detail"] == "data.go.kr query matched: 쌀밥"
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
        requested_queries.append(params["DESC_KOR"])
        return FakeResponse()

    monkeypatch.setenv("FOODSAFETYKOREA_API_KEY", "test-key")
    monkeypatch.delenv("NUTRITION_API_URL", raising=False)
    monkeypatch.setattr("src.nutrition_api.requests.get", fake_get)

    nutrition = get_nutrition_per_100g("rice")

    assert requested_queries[:3] == ["밥", "쌀밥", "흰밥"]
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

    monkeypatch.setenv("FOODSAFETYKOREA_API_KEY", "test-key")
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
