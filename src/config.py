"""Project configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st


@dataclass
class SheetConfig:
    """Keeps all Google Sheets links in one simple object."""

    pos_url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRcO2-POwu4fFaS1AaDksNLRjzG_qyhm5Y2xbFFpL2xfzes8msF-xtmSS9BwXqrLAmJbNse_sgwGmmo/pub?gid=152151064&single=true&output=csv"
    delivery_financials_url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSKXQYJXWYVEKsgAMCrY3_IQAeNOEvO0pt6T-t-okk9VYisKTM3T-nPOtPEybV4SYDgIYl0h_sxxJAZ/pub?gid=1483094230&single=true&output=csv"
    delivery_orders_url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRcO2-POwu4fFaS1AaDksNLRjzG_qyhm5Y2xbFFpL2xfzes8msF-xtmSS9BwXqrLAmJbNse_sgwGmmo/pub?gid=1302691369&single=true&output=csv"
    delivery_partner_raw_url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSKXQYJXWYVEKsgAMCrY3_IQAeNOEvO0pt6T-t-okk9VYisKTM3T-nPOtPEybV4SYDgIYl0h_sxxJAZ/pub?gid=923737959&single=true&output=csv"


@dataclass
class ResolvedUrl:
    """A single resolved URL together with the source it came from."""

    value: str = ""
    source: str = "missing"


@dataclass
class ResolvedSheetConfig:
    """Resolved URLs for all datasets, including source metadata."""

    pos_url: ResolvedUrl
    delivery_financials_url: ResolvedUrl
    delivery_orders_url: ResolvedUrl
    delivery_partner_raw_url: ResolvedUrl


# Optional code-level fallback URLs.
# The app will use these if no manual input, secrets, or environment values exist.
CONFIG_DEFAULT_SHEET_URLS = SheetConfig(
    pos_url=(
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRcO2-POwu4fFaS1AaDksNLRjzG_qyhm5Y2xbFFpL2xfzes8msF-xtmSS9BwXqrLAmJbNse_sgwGmmo/"
        "pub?gid=152151064&single=true&output=csv"
    ),
    delivery_financials_url=(
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vSKXQYJXWYVEKsgAMCrY3_IQAeNOEvO0pt6T-t-okk9VYisKTM3T-nPOtPEybV4SYDgIYl0h_sxxJAZ/"
        "pub?gid=1483094230&single=true&output=csv"
    ),
    delivery_orders_url=(
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vRcO2-POwu4fFaS1AaDksNLRjzG_qyhm5Y2xbFFpL2xfzes8msF-xtmSS9BwXqrLAmJbNse_sgwGmmo/"
        "pub?gid=1302691369&single=true&output=csv"
    ),
    # Raw published Google Sheet source for delivery partner data from Jan to Mar.
    delivery_partner_raw_url=(
        "https://docs.google.com/spreadsheets/d/e/"
        "2PACX-1vSKXQYJXWYVEKsgAMCrY3_IQAeNOEvO0pt6T-t-okk9VYisKTM3T-nPOtPEybV4SYDgIYl0h_sxxJAZ/"
        "pub?gid=923737959&single=true&output=csv"
    ),
)


def _read_secret(*path: str) -> str:
    """
    Read a value from Streamlit secrets without breaking the app
    when the key does not exist yet.
    """
    try:
        value = st.secrets
        for part in path:
            value = value[part]
        return str(value).strip()
    except Exception:
        return ""


def _clean_url(value: str | None) -> str:
    """Normalize URL strings so empty values are handled consistently."""
    return str(value or "").strip()


def _resolve_from_candidates(candidates: list[tuple[str, str]]) -> ResolvedUrl:
    """Pick the first non-empty URL from a priority-ordered list."""
    for source, value in candidates:
        cleaned_value = _clean_url(value)
        if cleaned_value:
            return ResolvedUrl(value=cleaned_value, source=source)
    return ResolvedUrl()


def _apply_manual_override(
    manual_value: str,
    fallback_value: ResolvedUrl,
) -> ResolvedUrl:
    """
    Use a sidebar override only when the user typed a non-empty value.

    If the sidebar field is blank, we fall back to the resolved config value.
    """
    cleaned_manual_value = _clean_url(manual_value)
    if cleaned_manual_value and cleaned_manual_value != fallback_value.value:
        return ResolvedUrl(value=cleaned_manual_value, source="manual")
    return fallback_value


def load_sheet_config_resolution() -> ResolvedSheetConfig:
    """
    Resolve sheet URLs from non-manual sources.

    Priority:
    1. Streamlit secrets
    2. Environment variables
    3. Config defaults in this file
    """
    pos_url = _resolve_from_candidates(
        [
            ("secrets", _read_secret("google_sheets", "pos_url")),
            ("secrets", _read_secret("pos_sheet_url")),
            ("environment", os.getenv("POS_SHEET_URL", "")),
            ("config default", CONFIG_DEFAULT_SHEET_URLS.pos_url),
        ]
    )

    delivery_financials_url = _resolve_from_candidates(
        [
            ("secrets", _read_secret("google_sheets", "delivery_financials_url")),
            ("secrets", _read_secret("delivery_financial_summary_url")),
            ("secrets", _read_secret("google_sheets", "delivery_url")),
            ("secrets", _read_secret("delivery_sheet_url")),
            ("environment", os.getenv("DELIVERY_FINANCIALS_SHEET_URL", "")),
            ("environment", os.getenv("DELIVERY_FINANCIAL_SUMMARY_URL", "")),
            ("environment", os.getenv("DELIVERY_SHEET_URL", "")),
            ("config default", CONFIG_DEFAULT_SHEET_URLS.delivery_financials_url),
        ]
    )

    delivery_orders_url = _resolve_from_candidates(
        [
            ("secrets", _read_secret("google_sheets", "delivery_orders_url")),
            ("secrets", _read_secret("delivery_order_level_url")),
            ("environment", os.getenv("DELIVERY_ORDERS_SHEET_URL", "")),
            ("environment", os.getenv("DELIVERY_ORDER_LEVEL_URL", "")),
            ("config default", CONFIG_DEFAULT_SHEET_URLS.delivery_orders_url),
        ]
    )

    delivery_partner_raw_url = _resolve_from_candidates(
        [
            ("secrets", _read_secret("google_sheets", "delivery_partner_raw_url")),
            ("secrets", _read_secret("delivery_partner_data_raw_url")),
            ("environment", os.getenv("DELIVERY_PARTNER_RAW_URL", "")),
            ("environment", os.getenv("DELIVERY_PARTNER_DATA_RAW_URL", "")),
            ("config default", CONFIG_DEFAULT_SHEET_URLS.delivery_partner_raw_url),
        ]
    )

    return ResolvedSheetConfig(
        pos_url=pos_url,
        delivery_financials_url=delivery_financials_url,
        delivery_orders_url=delivery_orders_url,
        delivery_partner_raw_url=delivery_partner_raw_url,
    )


def resolve_sheet_config_with_manual_inputs(
    manual_pos_url: str = "",
    manual_delivery_financials_url: str = "",
    manual_delivery_orders_url: str = "",
) -> ResolvedSheetConfig:
    """Resolve the final URLs after applying optional sidebar overrides."""
    base_config = load_sheet_config_resolution()

    return ResolvedSheetConfig(
        pos_url=_apply_manual_override(manual_pos_url, base_config.pos_url),
        delivery_financials_url=_apply_manual_override(
            manual_delivery_financials_url,
            base_config.delivery_financials_url,
        ),
        delivery_orders_url=_apply_manual_override(
            manual_delivery_orders_url,
            base_config.delivery_orders_url,
        ),
        delivery_partner_raw_url=base_config.delivery_partner_raw_url,
    )


def load_sheet_config() -> SheetConfig:
    """Return resolved non-manual URLs as plain strings."""
    resolved_config = load_sheet_config_resolution()
    return SheetConfig(
        pos_url=resolved_config.pos_url.value,
        delivery_financials_url=resolved_config.delivery_financials_url.value,
        delivery_orders_url=resolved_config.delivery_orders_url.value,
        delivery_partner_raw_url=resolved_config.delivery_partner_raw_url.value,
    )
