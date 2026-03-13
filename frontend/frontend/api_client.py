from __future__ import annotations
import requests
from frontend.config import (
    OPTIONS_ENDPOINT,
    QUERY_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
)

class APIClientError(Exception):
    """Raised when the frontend cannot successfully communicate with the backend."""

def _raise_for_bad_response(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        raise APIClientError(f"{exc}. Response: {payload}") from exc

def get_options() -> dict:
    try:
        response = requests.get(
            OPTIONS_ENDPOINT,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise APIClientError(str(exc)) from exc
    _raise_for_bad_response(response)

    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("Options endpoint did not return valid JSON.") from exc

def submit_query(
    query: str,
    company: str | None,
    form_folder: str | None,
    year: int | None,
) -> dict:
    payload = {
        "query": query,
        "company": company,
        "form_folder": form_folder,
        "year": year,
    }

    try:
        response = requests.post(
            QUERY_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise APIClientError(str(exc)) from exc
    _raise_for_bad_response(response)

    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("Query endpoint did not return valid JSON.") from exc