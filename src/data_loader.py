"""Functions that load Google Sheets data into pandas DataFrames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT_SECONDS = 20
RESPONSE_PREVIEW_LENGTH = 500


@dataclass
class SheetFetchPlan:
    """Describes how one sheet URL will be fetched."""

    original_url: str
    detected_type: str
    fetch_url: str


@dataclass
class SheetFetchDebug:
    """Detailed fetch diagnostics for one dataset."""

    request_method: str
    original_url: str
    detected_type: str
    final_fetch_url: str
    response_url: str = ""
    http_status_code: int | None = None
    content_type: str = ""
    response_preview: str = ""
    failure_type: str = ""
    exception_class: str = ""
    exception_message: str = ""


@dataclass
class SheetLoadResult:
    """The raw DataFrame plus detailed fetch diagnostics."""

    dataframe: pd.DataFrame | None
    fetch_debug: SheetFetchDebug

    @property
    def succeeded(self) -> bool:
        return self.dataframe is not None and not self.fetch_debug.exception_class


class SheetLoadError(Exception):
    """Structured exception that carries fetch diagnostics."""

    def __init__(self, fetch_debug: SheetFetchDebug):
        message = fetch_debug.exception_message or "Sheet loading failed."
        super().__init__(message)
        self.fetch_debug = fetch_debug


def _extract_spreadsheet_id(sheet_url: str) -> str:
    """Pull the Google Sheet ID out of a normal Google Sheets URL."""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError(
            "Could not find a Google Sheets file ID in the link you provided."
        )
    return match.group(1)


def _extract_gid(sheet_url: str) -> str:
    """
    Find the sheet tab ID (gid).

    Google Sheets links sometimes keep `gid` in the query string
    and sometimes in the `#gid=...` part at the end of the URL.
    """
    gid_match = re.search(r"(?:[?#&])gid=([0-9]+)", sheet_url)
    if gid_match:
        return gid_match.group(1)
    return "0"


def _is_direct_csv_link(parsed_url, query_params: dict[str, list[str]]) -> bool:
    """Detect links that already point directly to CSV content."""
    path = parsed_url.path.lower()
    return (
        query_params.get("output") == ["csv"]
        or query_params.get("format") == ["csv"]
        or path.endswith(".csv")
    )


def build_sheet_fetch_plan(sheet_url: str) -> SheetFetchPlan:
    """
    Decide whether to use the URL directly or convert it first.

    Supported examples:
    - normal Google Sheets edit/view links
    - published Google Sheets CSV links
    - direct CSV export links
    """
    clean_url = sheet_url.strip()
    if not clean_url:
        raise ValueError("The Google Sheets link is empty.")

    parsed_url = urlparse(clean_url)
    query_params = parse_qs(parsed_url.query)
    is_google_sheets_url = (
        "docs.google.com" in parsed_url.netloc and "/spreadsheets/" in parsed_url.path
    )

    if _is_direct_csv_link(parsed_url, query_params):
        detected_type = (
            "direct published CSV link" if is_google_sheets_url else "direct CSV link"
        )
        return SheetFetchPlan(
            original_url=clean_url,
            detected_type=detected_type,
            fetch_url=clean_url,
        )

    if not is_google_sheets_url:
        return SheetFetchPlan(
            original_url=clean_url,
            detected_type="direct URL",
            fetch_url=clean_url,
        )

    spreadsheet_id = _extract_spreadsheet_id(clean_url)
    gid = _extract_gid(clean_url)
    fetch_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export"
        f"?format=csv&gid={gid}"
    )
    return SheetFetchPlan(
        original_url=clean_url,
        detected_type="normal Google Sheets link",
        fetch_url=fetch_url,
    )


def build_google_sheet_csv_url(sheet_url: str) -> str:
    """Return the final fetch URL for one Google Sheet input URL."""
    return build_sheet_fetch_plan(sheet_url).fetch_url


def _build_fetch_debug(fetch_plan: SheetFetchPlan) -> SheetFetchDebug:
    """Create the initial debug object for one fetch attempt."""
    return SheetFetchDebug(
        request_method="GET",
        original_url=fetch_plan.original_url,
        detected_type=fetch_plan.detected_type,
        final_fetch_url=fetch_plan.fetch_url,
    )


def _raise_sheet_load_error(
    fetch_debug: SheetFetchDebug,
    failure_type: str,
    exception: Exception | None,
    exception_message: str,
) -> None:
    """Attach failure metadata and raise a structured loader exception."""
    fetch_debug.failure_type = failure_type
    fetch_debug.exception_class = (
        exception.__class__.__name__ if exception is not None else "SheetLoadError"
    )
    fetch_debug.exception_message = exception_message
    raise SheetLoadError(fetch_debug) from exception


def _fetch_csv_text(fetch_plan: SheetFetchPlan) -> tuple[str, SheetFetchDebug]:
    """
    Fetch CSV text with requests so HTTP behavior is explicit and debuggable.
    """
    fetch_debug = _build_fetch_debug(fetch_plan)

    try:
        response = requests.get(
            fetch_plan.fetch_url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout as error:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="timeout",
            exception=error,
            exception_message=str(error),
        )
    except requests.exceptions.ConnectionError as error:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="network connection failure",
            exception=error,
            exception_message=str(error),
        )
    except requests.exceptions.RequestException as error:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="request failure",
            exception=error,
            exception_message=str(error),
        )

    response_text = response.content.decode("utf-8-sig", errors="replace")
    fetch_debug.http_status_code = response.status_code
    fetch_debug.response_url = str(response.url)
    fetch_debug.content_type = response.headers.get("Content-Type", "").strip()
    fetch_debug.response_preview = response_text[:RESPONSE_PREVIEW_LENGTH]

    if response.status_code == 403:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="403 forbidden",
            exception=None,
            exception_message=(
                "The server returned HTTP 403 Forbidden. "
                "The published CSV link may not be publicly accessible."
            ),
        )

    if response.status_code == 404:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="404 not found",
            exception=None,
            exception_message=(
                "The server returned HTTP 404 Not Found. "
                "Check whether the published CSV URL is correct."
            ),
        )

    if response.status_code >= 400:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="http error",
            exception=None,
            exception_message=f"The server returned HTTP {response.status_code}.",
        )

    stripped_response_text = response_text.strip()
    if not stripped_response_text:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="empty response",
            exception=None,
            exception_message=(
                "The request succeeded, but the response body was empty."
            ),
        )

    lower_text = stripped_response_text.lower()
    content_type_lower = fetch_debug.content_type.lower()
    if "<html" in lower_text or "<!doctype html" in lower_text or "text/html" in content_type_lower:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="non-CSV HTML response",
            exception=None,
            exception_message=(
                "The request returned HTML instead of CSV. "
                "The published link may be wrong or inaccessible."
            ),
        )

    return stripped_response_text, fetch_debug


def _read_csv_text(
    csv_text: str,
    fetch_debug: SheetFetchDebug,
) -> pd.DataFrame:
    """Parse downloaded CSV text into a DataFrame."""
    try:
        dataframe = pd.read_csv(StringIO(csv_text), dtype=str)
    except pd.errors.EmptyDataError as error:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="empty response",
            exception=error,
            exception_message=(
                "The response body did not contain readable CSV rows."
            ),
        )
    except pd.errors.ParserError as error:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="CSV parsing failure",
            exception=error,
            exception_message=(
                "CSV parsing failed. The response was fetched but could not be parsed as CSV."
            ),
        )

    return dataframe


def _read_csv_text_raw(
    csv_text: str,
    fetch_debug: SheetFetchDebug,
) -> pd.DataFrame:
    """Parse downloaded CSV text without applying post-load normalization."""
    try:
        dataframe = pd.read_csv(
            StringIO(csv_text),
            dtype=str,
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError as error:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="empty response",
            exception=error,
            exception_message=(
                "The response body did not contain readable CSV rows."
            ),
        )
    except pd.errors.ParserError as error:
        _raise_sheet_load_error(
            fetch_debug=fetch_debug,
            failure_type="CSV parsing failure",
            exception=error,
            exception_message=(
                "CSV parsing failed. The response was fetched but could not be parsed as CSV."
            ),
        )

    return dataframe


@st.cache_data(show_spinner=False)
def load_google_sheet_with_debug(sheet_url: str) -> SheetLoadResult:
    """Load one sheet and always return debug information about the fetch."""
    try:
        fetch_plan = build_sheet_fetch_plan(sheet_url)
        csv_text, fetch_debug = _fetch_csv_text(fetch_plan)
        dataframe = _read_csv_text(csv_text, fetch_debug)
    except SheetLoadError as error:
        return SheetLoadResult(dataframe=None, fetch_debug=error.fetch_debug)
    except Exception as error:
        fallback_debug = SheetFetchDebug(
            request_method="GET",
            original_url=sheet_url,
            detected_type="unresolved",
            final_fetch_url=sheet_url,
            failure_type="unexpected loader failure",
            exception_class=error.__class__.__name__,
            exception_message=str(error),
        )
        return SheetLoadResult(dataframe=None, fetch_debug=fallback_debug)

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    dataframe = dataframe.replace(r"^\s*$", pd.NA, regex=True)
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)
    return SheetLoadResult(dataframe=dataframe, fetch_debug=fetch_debug)


def load_google_sheet(sheet_url: str) -> pd.DataFrame:
    """Backward-compatible loader that returns only the DataFrame."""
    result = load_google_sheet_with_debug(sheet_url)
    if result.dataframe is None:
        raise ValueError(result.fetch_debug.exception_message)
    return result.dataframe


@st.cache_data(show_spinner=False)
def load_google_sheet_raw_with_debug(sheet_url: str) -> SheetLoadResult:
    """Load one sheet as a raw CSV table without post-load cleanup."""
    try:
        fetch_plan = build_sheet_fetch_plan(sheet_url)
        csv_text, fetch_debug = _fetch_csv_text(fetch_plan)
        dataframe = _read_csv_text_raw(csv_text, fetch_debug)
    except SheetLoadError as error:
        return SheetLoadResult(dataframe=None, fetch_debug=error.fetch_debug)
    except Exception as error:
        fallback_debug = SheetFetchDebug(
            request_method="GET",
            original_url=sheet_url,
            detected_type="unresolved",
            final_fetch_url=sheet_url,
            failure_type="unexpected loader failure",
            exception_class=error.__class__.__name__,
            exception_message=str(error),
        )
        return SheetLoadResult(dataframe=None, fetch_debug=fallback_debug)

    return SheetLoadResult(dataframe=dataframe, fetch_debug=fetch_debug)


def load_pos_data_with_debug(sheet_url: str) -> SheetLoadResult:
    """Load the POS / offline transaction sheet with diagnostics."""
    return load_google_sheet_with_debug(sheet_url)


def load_delivery_financials_with_debug(sheet_url: str) -> SheetLoadResult:
    """Load the delivery financial summary sheet with diagnostics."""
    return load_google_sheet_with_debug(sheet_url)


def load_delivery_orders_with_debug(sheet_url: str) -> SheetLoadResult:
    """Load the delivery partner order-level sheet with diagnostics."""
    return load_google_sheet_with_debug(sheet_url)


def load_delivery_partner_raw_with_debug(sheet_url: str) -> SheetLoadResult:
    """Load the raw published delivery partner CSV source for Jan to Mar."""
    return load_google_sheet_raw_with_debug(sheet_url)


def load_pos_data(sheet_url: str) -> pd.DataFrame:
    """Load the POS / offline transaction sheet."""
    return load_google_sheet(sheet_url)


def load_delivery_financials(sheet_url: str) -> pd.DataFrame:
    """Load the delivery financial summary sheet."""
    return load_google_sheet(sheet_url)


def load_delivery_orders(sheet_url: str) -> pd.DataFrame:
    """Load the delivery partner order-level sheet."""
    return load_google_sheet(sheet_url)


def load_delivery_partner_raw(sheet_url: str) -> pd.DataFrame:
    """Load the raw published delivery partner CSV source without cleaning."""
    result = load_google_sheet_raw_with_debug(sheet_url)
    if result.dataframe is None:
        raise ValueError(result.fetch_debug.exception_message)
    return result.dataframe


def load_pos_sheet(sheet_url: str) -> pd.DataFrame:
    """Backward-compatible alias for the POS loader."""
    return load_pos_data(sheet_url)


def load_delivery_sheet(sheet_url: str) -> pd.DataFrame:
    """Backward-compatible alias for the delivery financial loader."""
    return load_delivery_financials(sheet_url)
