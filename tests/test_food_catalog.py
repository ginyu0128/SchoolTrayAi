from src.food_catalog import get_density_g_per_ml, get_food_profile, normalize_food_key


def test_normalize_food_key_uses_aliases():
    assert normalize_food_key("\ubc25") == "rice"
    assert normalize_food_key("\ub5a1\uac08\ube44") == "tteokgalbi"
    assert normalize_food_key("\uc18c\uc138\uc9c0") == "sausage"
    assert normalize_food_key("not-real") == "unknown"


def test_soup_density_is_lower_than_default_food_density():
    assert get_density_g_per_ml("soup") == 0.75
    assert get_density_g_per_ml("rice") == 0.85
    assert get_density_g_per_ml("unknown") == 0.85


def test_food_profile_contains_volume_profile():
    assert get_food_profile("tteokgalbi")["volume_profile"] == "flat_piece"


def test_food_profile_contains_school_meal_candidates():
    assert get_food_profile("sausage")["display_name"] == "\uc18c\uc2dc\uc9c0"
    assert get_food_profile("fruit")["display_name"] == "\uacfc\uc77c"
    assert get_food_profile("fried_chicken")["display_name"] == "\ub2ed\uace0\uae30\ud280\uae40"
