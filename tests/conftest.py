"""Pytest hooks: project imports and local `.env` for integration tests."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Resolve repo root (parent of tests/)
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")
