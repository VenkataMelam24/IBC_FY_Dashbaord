"""Cleaning and preparation helpers for all source datasets."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


MONTH_LOOKUP = {
    "1": 1,
    "01": 1,
    "jan": 1,
    "january": 1,
    "januar": 1,
    "2": 2,
    "02": 2,
    "feb": 2,
    "february": 2,
    "februar": 2,
    "3": 3,
    "03": 3,
    "mar": 3,
    "march": 3,
    "marz": 3,
    "maerz": 3,
    "märz": 3,
    "4": 4,
    "04": 4,
    "apr": 4,
    "april": 4,
    "5": 5,
    "05": 5,
    "may": 5,
    "mai": 5,
    "6": 6,
    "06": 6,
    "jun": 6,
    "june": 6,
    "juni": 6,
    "7": 7,
    "07": 7,
    "jul": 7,
    "july": 7,
    "juli": 7,
    "8": 8,
    "08": 8,
    "aug": 8,
    "august": 8,
    "9": 9,
    "09": 9,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "10": 10,
    "oct": 10,
    "okt": 10,
    "october": 10,
    "oktober": 10,
    "11": 11,
    "nov": 11,
    "november": 11,
    "12": 12,
    "dec": 12,
    "dez": 12,
    "december": 12,
    "dezember": 12,
}

MONTH_NAME_LOOKUP = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def _strip_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove extra spaces from text columns only."""
    cleaned = dataframe.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return cleaned


def _normalize_text(value: str) -> str:
    """Normalize text to make month matching more reliable."""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    return text


def _normalize_single_separator_number(text: str, separator: str) -> str:
    """
    Decide whether one separator is acting like a decimal separator
    or a thousands separator.
    """
    parts = text.split(separator)

    if len(parts) > 1 and len(parts[-1]) in (1, 2):
        integer_part = "".join(parts[:-1]) or "0"
        decimal_part = parts[-1]
        return f"{integer_part}.{decimal_part}"

    return "".join(parts)


def _normalize_mixed_separator_number(
    text: str, decimal_separator: str, thousands_separator: str
) -> str:
    """Handle values like 1.234,56 or 1,234.56."""
    text = text.replace(thousands_separator, "")
    parts = text.split(decimal_separator)

    if len(parts) > 1:
        integer_part = "".join(parts[:-1]) or "0"
        decimal_part = parts[-1]
        return f"{integer_part}.{decimal_part}"

    return text


def parse_number(value: object) -> float | pd.NA:
    """
    Convert text like:
    - €1.234,56
    - -45,00
    - (12.50)
    - 1,234.56
    into a numeric value.

    Important rule for this project:
    - negative values must stay negative
    - values in brackets like (45.00) become -45.00
    - we never force absolute values
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
    if "-" in text or "−" in text:
        negative = True

    # Keep only characters that matter for numeric conversion.
    text = text.replace("−", "-")
    text = text.replace("€", "").replace("%", "")
    text = re.sub(r"[A-Za-z]", "", text)
    text = text.replace(" ", "")
    text = text.replace("(", "").replace(")", "").replace("-", "")
    text = re.sub(r"[^0-9,\.]", "", text)

    if not text:
        return pd.NA

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = _normalize_mixed_separator_number(text, ",", ".")
        else:
            text = _normalize_mixed_separator_number(text, ".", ",")
    elif "," in text:
        text = _normalize_single_separator_number(text, ",")
    elif "." in text:
        text = _normalize_single_separator_number(text, ".")

    try:
        number = float(text)
    except ValueError:
        return pd.NA

    return -number if negative else number


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Apply numeric parsing to a whole pandas column."""
    return pd.to_numeric(series.map(parse_number), errors="coerce")


def clean_integer_series(series: pd.Series) -> pd.Series:
    """Parse a numeric column and store it as pandas nullable integers."""
    return clean_numeric_series(series).round().astype("Int64")


def parse_month_value(value: object) -> int | pd.NA:
    """Convert many month formats into one month number from 1 to 12."""
    if pd.isna(value):
        return pd.NA

    text = _normalize_text(value)
    if text in MONTH_LOOKUP:
        return MONTH_LOOKUP[text]

    # Some month cells may contain extra text like "March 2025" or "03 - March".
    tokens = re.split(r"[^a-z0-9]+", text)
    for token in tokens:
        if token in MONTH_LOOKUP:
            return MONTH_LOOKUP[token]

    return pd.NA


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Parse timestamps carefully.

    We try `dayfirst=True` first because a Berlin-based business often exports
    dates in European order. Then we retry remaining failed values once.
    """
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing_mask = parsed.isna() & series.notna()

    if missing_mask.any():
        parsed_retry = pd.to_datetime(
            series[missing_mask], errors="coerce", dayfirst=False
        )
        parsed.loc[missing_mask] = parsed_retry

    return parsed


def _add_datetime_helper_columns(
    dataframe: pd.DataFrame, datetime_column: str
) -> pd.DataFrame:
    """Create standard calendar helper columns from one datetime column."""
    if datetime_column not in dataframe.columns:
        return dataframe

    cleaned = dataframe.copy()
    datetime_series = cleaned[datetime_column]

    cleaned["Date"] = datetime_series.dt.normalize()
    cleaned["Year"] = datetime_series.dt.year.astype("Int64")
    cleaned["Month Number"] = datetime_series.dt.month.astype("Int64")

    month_name = datetime_series.dt.strftime("%B")
    cleaned["Month Name"] = month_name.where(datetime_series.notna())

    cleaned["Hour"] = datetime_series.dt.hour.astype("Int64")

    weekday = datetime_series.dt.day_name()
    cleaned["Weekday"] = weekday.where(datetime_series.notna())

    return cleaned


def _build_order_datetime(
    order_date_series: pd.Series | None, order_time_series: pd.Series | None
) -> pd.Series:
    """Combine order date and order time into one parsed datetime series."""
    if order_date_series is None and order_time_series is None:
        return pd.Series(dtype="datetime64[ns]")

    if order_date_series is None:
        combined_text = order_time_series.fillna("").astype(str).str.strip()
    elif order_time_series is None:
        combined_text = order_date_series.fillna("").astype(str).str.strip()
    else:
        combined_text = (
            order_date_series.fillna("").astype(str).str.strip()
            + " "
            + order_time_series.fillna("").astype(str).str.strip()
        ).str.strip()

    combined_text = combined_text.replace("", pd.NA)
    return _parse_timestamp_series(combined_text)


def prepare_pos_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the POS / in-store sales data."""
    cleaned = _strip_text_columns(dataframe)

    if "Amount" in cleaned.columns:
        # Keep refunds / reverse transactions negative.
        cleaned["Amount"] = clean_numeric_series(cleaned["Amount"])

    if "Time-Stamp" in cleaned.columns:
        cleaned["Time-Stamp"] = _parse_timestamp_series(cleaned["Time-Stamp"])
        cleaned = _add_datetime_helper_columns(cleaned, "Time-Stamp")

    return cleaned


def prepare_delivery_financials(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the delivery financial summary data."""
    cleaned = _strip_text_columns(dataframe)

    currency_columns = ["Gross", "Deduction", "Net"]
    for column in currency_columns:
        if column in cleaned.columns:
            cleaned[column] = clean_numeric_series(cleaned[column])

    if "Deduction percentage" in cleaned.columns:
        cleaned["Deduction percentage"] = clean_numeric_series(
            cleaned["Deduction percentage"]
        )

    if "orders" in cleaned.columns:
        cleaned["orders"] = clean_integer_series(cleaned["orders"])

    if "year" in cleaned.columns:
        cleaned["year"] = clean_integer_series(cleaned["year"])

    if "Month" in cleaned.columns:
        cleaned["Month Number"] = cleaned["Month"].map(parse_month_value).astype("Int64")
        cleaned["Month Name"] = cleaned["Month Number"].map(MONTH_NAME_LOOKUP)

    if "year" in cleaned.columns and "Month Number" in cleaned.columns:
        cleaned["Month Start"] = cleaned.apply(build_month_start, axis=1)

    return cleaned


def prepare_delivery_orders(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean the delivery partner order-level data."""
    cleaned = _strip_text_columns(dataframe)

    order_date_series = cleaned["Order date"] if "Order date" in cleaned.columns else None
    order_time_series = cleaned["Order Time"] if "Order Time" in cleaned.columns else None

    if order_date_series is None and order_time_series is None:
        cleaned["Order datetime"] = pd.Series(
            pd.NaT,
            index=cleaned.index,
            dtype="datetime64[ns]",
        )
    else:
        cleaned["Order datetime"] = _build_order_datetime(
            order_date_series=order_date_series,
            order_time_series=order_time_series,
        )

    cleaned = _add_datetime_helper_columns(cleaned, "Order datetime")

    return cleaned


def prepare_delivery_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for delivery financial cleaning."""
    return prepare_delivery_financials(dataframe)


def build_month_start(row: pd.Series) -> pd.Timestamp | pd.NaT:
    """Create one clean first-of-month date from year + month."""
    year = row.get("year")
    month = row.get("Month Number")

    if pd.isna(year) or pd.isna(month):
        return pd.NaT

    return pd.Timestamp(year=int(year), month=int(month), day=1)
