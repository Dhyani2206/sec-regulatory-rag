from __future__ import annotations

def as_list(value) -> list:
    return value if isinstance(value, list) else []
def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}