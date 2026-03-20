from __future__ import annotations
import requests
from frontend.config import (
    AGENT_QUERY_ENDPOINT,
    BROWSE_ENDPOINT,
    CORPUS_STATS_ENDPOINT,
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


def submit_agent_query(
    query: str,
    company: str | None,
    form_folder: str | None,
    year: int | None,
) -> dict:
    """
    Submit a query to the LangChain Deep Agent endpoint.

    Accepts the same parameters as ``submit_query``; routes to the
    ``/api/v1/agent-query`` endpoint instead of the deterministic engine.
    """
    payload = {
        "query": query,
        "company": company,
        "form_folder": form_folder,
        "year": year,
    }

    try:
        response = requests.post(
            AGENT_QUERY_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise APIClientError(str(exc)) from exc
    _raise_for_bad_response(response)

    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("Agent query endpoint did not return valid JSON.") from exc


def browse_sections(company: str, form_folder: str, year: int) -> dict:
    """Return sections available in a specific filing."""
    try:
        response = requests.get(
            BROWSE_ENDPOINT,
            params={"company": company, "form_folder": form_folder, "year": year},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise APIClientError(str(exc)) from exc
    _raise_for_bad_response(response)
    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("Browse endpoint did not return valid JSON.") from exc


def browse_chunks(
    company: str,
    form_folder: str,
    year: int,
    section: str,
    page: int = 1,
    page_size: int = 10,
) -> dict:
    """Return paginated chunks for a specific filing section."""
    try:
        response = requests.get(
            f"{BROWSE_ENDPOINT}/chunks",
            params={
                "company": company,
                "form_folder": form_folder,
                "year": year,
                "section": section,
                "page": page,
                "page_size": page_size,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise APIClientError(str(exc)) from exc
    _raise_for_bad_response(response)
    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("Browse chunks endpoint did not return valid JSON.") from exc


def get_corpus_stats() -> dict:
    """Return aggregate statistics for the indexed corpus."""
    try:
        response = requests.get(
            CORPUS_STATS_ENDPOINT,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise APIClientError(str(exc)) from exc
    _raise_for_bad_response(response)
    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("Corpus stats endpoint did not return valid JSON.") from exc