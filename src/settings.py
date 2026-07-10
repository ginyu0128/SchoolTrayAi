from __future__ import annotations

import os
from typing import Any


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read configuration from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value: Any = st.secrets.get(name)
    except Exception:
        return default

    if secret_value is None:
        return default
    return str(secret_value)
