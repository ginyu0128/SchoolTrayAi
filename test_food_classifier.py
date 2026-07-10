import numpy as np

from src.food_classifier import classify_food


def test_classify_food_defaults_rice_cell_to_rice(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    image = np.full((40, 40, 3), 240, dtype=np.uint8)

    result = classify_food(image, cell_id="rice")

    assert result["food_key"] == "rice"
    assert result["source"] == "rule_based"


def test_classify_food_defaults_soup_cell_to_soup(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    image = np.full((40, 40, 3), 100, dtype=np.uint8)

    result = classify_food(image, cell_id="soup")

    assert result["food_key"] == "soup"


def test_rice_and_soup_call_openai_when_api_key_is_set(monkeypatch):
    posted_payloads = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": (
                    '{"raw_food_name":"rice",'
                    '"food_key":"rice",'
                    '"search_name":"rice",'
                    '"confidence":0.91,'
                    '"reason":"white rice appearance"}'
                )
            }

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.food_classifier.requests.post", fake_post)
    image = np.full((40, 40, 3), 240, dtype=np.uint8)

    rice_result = classify_food(image, cell_id="rice")
    soup_result = classify_food(image, cell_id="soup")

    assert rice_result["source"] == "openai_vision"
    assert soup_result["source"] == "openai_vision"
    assert len(posted_payloads) == 2
    assert "rice" in posted_payloads[0]["input"][0]["content"][0]["text"]
    assert "soup" in posted_payloads[1]["input"][0]["content"][0]["text"]


def test_classify_food_uses_rule_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    image = np.full((40, 40, 3), 200, dtype=np.uint8)

    result = classify_food(image, cell_id="side_left")

    assert result["food_key"] == "unknown"
    assert result["source"] == "rule_based"


def test_classify_food_accepts_openai_raw_and_search_names(monkeypatch):
    posted_payloads = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": (
                    '{"raw_food_name":"spicy pork stir fry",'
                    '"food_key":"meat",'
                    '"search_name":"jeyuk bokkeum",'
                    '"confidence":0.82,'
                    '"reason":"red stir-fried pork appearance"}'
                )
            }

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        assert timeout == 12
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.food_classifier.requests.post", fake_post)
    image = np.full((40, 40, 3), 160, dtype=np.uint8)

    result = classify_food(image, cell_id="side_left")

    prompt = posted_payloads[0]["input"][0]["content"][0]["text"]
    assert "raw_food_name" in prompt
    assert result["food_key"] == "meat"
    assert result["raw_food_name"] == "spicy pork stir fry"
    assert result["search_name"] == "jeyuk bokkeum"
    assert result["source"] == "openai_vision"


def test_classify_food_accepts_openai_json_inside_markdown(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": (
                    "```json\n"
                    '{"raw_food_name":"curry rice sauce",'
                    '"food_key":"soup",'
                    '"search_name":"curry",'
                    '"confidence":0.76,'
                    '"reason":"brown curry sauce in bowl"}'
                    "\n```"
                )
            }

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.food_classifier.requests.post", fake_post)
    image = np.full((40, 40, 3), 160, dtype=np.uint8)

    result = classify_food(image, cell_id="soup")

    assert result["food_key"] == "soup"
    assert result["raw_food_name"] == "curry rice sauce"
    assert result["search_name"] == "curry"
    assert result["source"] == "openai_vision"


def test_classify_food_maps_sausage_candidate(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": (
                    '{"raw_food_name":"소시지볶음",'
                    '"food_key":"sausage",'
                    '"search_name":"소시지볶음",'
                    '"confidence":0.88,'
                    '"reason":"smooth rounded sausage slices"}'
                )
            }

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.food_classifier.requests.post", fake_post)
    image = np.full((40, 40, 3), 160, dtype=np.uint8)

    result = classify_food(image, cell_id="side_right")

    assert result["food_key"] == "sausage"
    assert result["search_name"] == "소시지볶음"
    assert result["source"] == "openai_vision"


def test_classify_food_recovers_food_key_from_search_name(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": (
                    '{"raw_food_name":"복숭아",'
                    '"food_key":"unknown",'
                    '"search_name":"복숭아",'
                    '"confidence":0.87,'
                    '"reason":"sliced peach in a side compartment"}'
                )
            }

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.food_classifier.requests.post", fake_post)
    image = np.full((40, 40, 3), 160, dtype=np.uint8)

    result = classify_food(image, cell_id="side_middle_right")

    assert result["food_key"] == "fruit"
    assert result["search_name"] == "복숭아"
    assert result["source"] == "openai_vision"


def test_rice_cell_falls_back_when_openai_returns_unknown(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": (
                    '{"raw_food_name":"unknown",'
                    '"food_key":"unknown",'
                    '"search_name":"unknown",'
                    '"confidence":0.2,'
                    '"reason":"unclear crop"}'
                )
            }

    def fake_post(url, headers, json, timeout):
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.food_classifier.requests.post", fake_post)
    image = np.full((40, 40, 3), 240, dtype=np.uint8)

    result = classify_food(image, cell_id="rice")

    assert result["food_key"] == "rice"
    assert result["source"] == "openai_vision_cell_fallback"


def test_classify_food_reports_openai_failure_when_api_key_is_set(monkeypatch):
    def fail_post(*args, **kwargs):
        raise RuntimeError("network failure")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.food_classifier.requests.post", fail_post)
    image = np.full((40, 40, 3), 160, dtype=np.uint8)

    result = classify_food(image, cell_id="side_left")

    assert result["food_key"] == "unknown"
    assert result["source"] == "openai_vision_failed"
