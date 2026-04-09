"""Reusable star-schema-style modeling helpers for the dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.date_table import build_date_table, build_hour_table


def _empty_dataframe(columns: list[str]) -> pd.DataFrame:
    """Return an empty table with the expected columns."""
    return pd.DataFrame(columns=columns)


def _normalize_date_column(dataframe: pd.DataFrame, column_name: str = "Date") -> pd.DataFrame:
    """Normalize a date column so it is safe for joining to the date dimension."""
    modeled = dataframe.copy()
    if column_name in modeled.columns:
        modeled[column_name] = pd.to_datetime(modeled[column_name], errors="coerce").dt.normalize()
    return modeled


def _add_date_key(dataframe: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """Create a YYYYMMDD-style integer key from a date column."""
    modeled = dataframe.copy()
    if date_column in modeled.columns:
        date_series = pd.to_datetime(modeled[date_column], errors="coerce")
        modeled["Date Key"] = pd.to_numeric(
            date_series.dt.strftime("%Y%m%d"),
            errors="coerce",
        ).astype("Int64")
    else:
        modeled["Date Key"] = pd.Series(pd.NA, index=modeled.index, dtype="Int64")
    return modeled


def _add_hour_key(dataframe: pd.DataFrame, hour_column: str = "Hour") -> pd.DataFrame:
    """Create a nullable integer hour key from an hour column."""
    modeled = dataframe.copy()
    if hour_column in modeled.columns:
        modeled[hour_column] = pd.to_numeric(modeled[hour_column], errors="coerce").astype("Int64")
        modeled["Hour Key"] = modeled[hour_column]
    else:
        modeled["Hour Key"] = pd.Series(pd.NA, index=modeled.index, dtype="Int64")
    return modeled


def _reorder_columns(dataframe: pd.DataFrame, preferred_columns: list[str]) -> pd.DataFrame:
    """
    Reorder columns while keeping extra columns.

    This keeps the modeled tables readable without throwing away useful fields.
    """
    existing_preferred_columns = [
        column for column in preferred_columns if column in dataframe.columns
    ]
    extra_columns = [
        column for column in dataframe.columns if column not in existing_preferred_columns
    ]
    return dataframe[existing_preferred_columns + extra_columns]


def build_fact_pos_sales(df_pos_cleaned: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build the POS sales fact table.

    Grain:
    - one row = one POS bill / transaction
    """
    preferred_columns = [
        "Bill-nr.",
        "Amount",
        "Payment type",
        "Employee",
        "Terminal",
        "Time-Stamp",
        "Date",
        "Date Key",
        "Hour",
        "Hour Key",
        "Year",
        "Month Number",
        "Month Name",
        "Weekday",
        "Channel",
    ]

    if df_pos_cleaned is None:
        return _empty_dataframe(preferred_columns)

    fact = df_pos_cleaned.copy()
    fact = _normalize_date_column(fact, "Date")
    fact = _add_date_key(fact, "Date")
    fact = _add_hour_key(fact, "Hour")
    fact["Channel"] = "Offline"
    return _reorder_columns(fact, preferred_columns)


def build_fact_delivery_orders(
    df_delivery_orders_cleaned: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build the delivery order-level fact table.

    Grain:
    - one row = one delivery order event
    """
    preferred_columns = [
        "Partner",
        "Order date",
        "Order Time",
        "Order datetime",
        "Date",
        "Date Key",
        "Hour",
        "Hour Key",
        "Year",
        "Month Number",
        "Month Name",
        "Weekday",
        "Channel",
    ]

    if df_delivery_orders_cleaned is None:
        return _empty_dataframe(preferred_columns)

    fact = df_delivery_orders_cleaned.copy()
    fact = _normalize_date_column(fact, "Date")
    fact = _add_date_key(fact, "Date")
    fact = _add_hour_key(fact, "Hour")
    fact["Channel"] = "Online"
    return _reorder_columns(fact, preferred_columns)


def build_fact_delivery_financials(
    df_delivery_financials_cleaned: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build the delivery financial summary fact table.

    Grain:
    - one row = one delivery partner financial summary row
    """
    preferred_columns = [
        "Partner",
        "Month",
        "year",
        "orders",
        "Gross",
        "Deduction",
        "Deduction percentage",
        "Net",
        "Month Start",
        "Date",
        "Date Key",
        "Month Number",
        "Month Name",
        "Channel",
    ]

    if df_delivery_financials_cleaned is None:
        return _empty_dataframe(preferred_columns)

    fact = df_delivery_financials_cleaned.copy()

    if "Date" not in fact.columns and "Month Start" in fact.columns:
        fact["Date"] = pd.to_datetime(fact["Month Start"], errors="coerce").dt.normalize()

    fact = _normalize_date_column(fact, "Date")
    fact = _add_date_key(fact, "Date")
    fact["Channel"] = "Online"
    return _reorder_columns(fact, preferred_columns)


def build_dim_date(
    df_pos_cleaned: pd.DataFrame | None = None,
    df_delivery_orders_cleaned: pd.DataFrame | None = None,
    forward_months: int = 12,
) -> pd.DataFrame:
    """Build the daily date dimension from the cleaned dated fact tables."""
    return build_date_table(
        pos_dataframe=df_pos_cleaned,
        delivery_orders_dataframe=df_delivery_orders_cleaned,
        forward_months=forward_months,
    )


def build_dim_hour() -> pd.DataFrame:
    """Build the reusable hour dimension."""
    return build_hour_table()


def build_data_model(
    df_pos_cleaned: pd.DataFrame | None = None,
    df_delivery_orders_cleaned: pd.DataFrame | None = None,
    df_delivery_financials_cleaned: pd.DataFrame | None = None,
    forward_months: int = 12,
) -> dict[str, Any]:
    """
    Build and return all modeled fact and dimension tables.

    The model keeps each fact table separate and relationship-ready.
    """
    fact_pos_sales = build_fact_pos_sales(df_pos_cleaned)
    fact_delivery_orders = build_fact_delivery_orders(df_delivery_orders_cleaned)
    fact_delivery_financials = build_fact_delivery_financials(
        df_delivery_financials_cleaned
    )
    dim_date = build_dim_date(
        df_pos_cleaned=df_pos_cleaned,
        df_delivery_orders_cleaned=df_delivery_orders_cleaned,
        forward_months=forward_months,
    )
    dim_hour = build_dim_hour()

    return {
        "fact_pos_sales": fact_pos_sales,
        "fact_delivery_orders": fact_delivery_orders,
        "fact_delivery_financials": fact_delivery_financials,
        "dim_date": dim_date,
        "dim_hour": dim_hour,
    }
