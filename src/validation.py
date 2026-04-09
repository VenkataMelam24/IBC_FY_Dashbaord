"""Validation helpers for dataset readiness checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


@dataclass
class ValidationCheck:
    """Represents one validation result shown in the Streamlit app."""

    name: str
    passed: bool
    detail: str


def _count_non_null(series: pd.Series | None) -> int:
    """Safely count non-empty values in a column."""
    if series is None:
        return 0
    return int(series.notna().sum())


def build_required_columns_check(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    check_name: str,
) -> ValidationCheck:
    """Check whether all expected columns exist."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        return ValidationCheck(
            name=check_name,
            passed=False,
            detail=f"Missing columns: {', '.join(missing_columns)}",
        )

    return ValidationCheck(
        name=check_name,
        passed=True,
        detail=f"All expected columns are present: {', '.join(required_columns)}",
    )


def build_numeric_columns_check(
    raw_dataframe: pd.DataFrame,
    cleaned_dataframe: pd.DataFrame,
    columns: list[str],
    check_name: str,
) -> ValidationCheck:
    """Check that numeric fields exist and were converted to numeric dtypes."""
    messages: list[str] = []
    passed = True

    for column in columns:
        if column not in cleaned_dataframe.columns:
            passed = False
            messages.append(f"{column}: missing after cleaning")
            continue

        if not is_numeric_dtype(cleaned_dataframe[column]):
            passed = False
            messages.append(f"{column}: not numeric after cleaning")
            continue

        raw_non_null = _count_non_null(raw_dataframe[column]) if column in raw_dataframe.columns else 0
        cleaned_non_null = _count_non_null(cleaned_dataframe[column])

        if raw_non_null > 0 and cleaned_non_null == 0:
            passed = False
            messages.append(f"{column}: 0/{raw_non_null} values parsed")
        else:
            messages.append(f"{column}: {cleaned_non_null}/{raw_non_null} values parsed")

    return ValidationCheck(name=check_name, passed=passed, detail=" | ".join(messages))


def build_datetime_column_check(
    raw_dataframe: pd.DataFrame,
    cleaned_dataframe: pd.DataFrame,
    source_columns: list[str],
    cleaned_column: str,
    check_name: str,
) -> ValidationCheck:
    """Check that one datetime column was created and parsed successfully."""
    if cleaned_column not in cleaned_dataframe.columns:
        return ValidationCheck(
            name=check_name,
            passed=False,
            detail=f"{cleaned_column} is missing after cleaning.",
        )

    if not is_datetime64_any_dtype(cleaned_dataframe[cleaned_column]):
        return ValidationCheck(
            name=check_name,
            passed=False,
            detail=f"{cleaned_column} is not a datetime column after cleaning.",
        )

    source_non_null = 0
    available_source_columns = [
        column for column in source_columns if column in raw_dataframe.columns
    ]
    if available_source_columns:
        source_non_null = int(raw_dataframe[available_source_columns].notna().any(axis=1).sum())

    cleaned_non_null = _count_non_null(cleaned_dataframe[cleaned_column])

    if source_non_null > 0 and cleaned_non_null == 0:
        return ValidationCheck(
            name=check_name,
            passed=False,
            detail=f"{cleaned_column} parsed 0 values from {source_non_null} non-empty source cells.",
        )

    return ValidationCheck(
        name=check_name,
        passed=True,
        detail=(
            f"{cleaned_column} parsed successfully with "
            f"{cleaned_non_null} non-empty datetime values."
        ),
    )


def build_helper_columns_check(
    cleaned_dataframe: pd.DataFrame,
    helper_columns: list[str],
    check_name: str,
) -> ValidationCheck:
    """Check that helper columns were created during cleaning."""
    missing_columns = [column for column in helper_columns if column not in cleaned_dataframe.columns]
    if missing_columns:
        return ValidationCheck(
            name=check_name,
            passed=False,
            detail=f"Missing helper columns: {', '.join(missing_columns)}",
        )

    non_null_summary = ", ".join(
        f"{column}: {_count_non_null(cleaned_dataframe[column])}"
        for column in helper_columns
    )
    return ValidationCheck(
        name=check_name,
        passed=True,
        detail=f"Helper columns created successfully. Non-null counts -> {non_null_summary}",
    )


def build_pos_validation_checks(
    raw_dataframe: pd.DataFrame,
    cleaned_dataframe: pd.DataFrame,
) -> list[ValidationCheck]:
    """Validation checks for the POS / offline sales dataset."""
    return [
        build_required_columns_check(
            raw_dataframe,
            ["Bill-nr.", "Amount", "Payment type", "Employee", "Terminal", "Time-Stamp"],
            "Expected raw columns",
        ),
        build_numeric_columns_check(
            raw_dataframe,
            cleaned_dataframe,
            ["Amount"],
            "Amount numeric check",
        ),
        build_datetime_column_check(
            raw_dataframe,
            cleaned_dataframe,
            ["Time-Stamp"],
            "Time-Stamp",
            "Time-Stamp parsing check",
        ),
        build_helper_columns_check(
            cleaned_dataframe,
            ["Date", "Year", "Month Number", "Month Name", "Hour", "Weekday"],
            "POS helper columns check",
        ),
    ]


def build_delivery_financials_validation_checks(
    raw_dataframe: pd.DataFrame,
    cleaned_dataframe: pd.DataFrame,
) -> list[ValidationCheck]:
    """Validation checks for the delivery financial summary dataset."""
    return [
        build_required_columns_check(
            raw_dataframe,
            [
                "Month",
                "Partner",
                "year",
                "orders",
                "Gross",
                "Deduction",
                "Deduction percentage",
                "Net",
            ],
            "Expected raw columns",
        ),
        build_numeric_columns_check(
            raw_dataframe,
            cleaned_dataframe,
            ["Gross", "Deduction", "Net", "Deduction percentage", "orders", "year"],
            "Financial numeric columns check",
        ),
        build_helper_columns_check(
            cleaned_dataframe,
            ["Month Number", "Month Name", "Month Start"],
            "Financial helper columns check",
        ),
    ]


def build_delivery_orders_validation_checks(
    raw_dataframe: pd.DataFrame,
    cleaned_dataframe: pd.DataFrame,
) -> list[ValidationCheck]:
    """Validation checks for the delivery partner order-level dataset."""
    return [
        build_required_columns_check(
            raw_dataframe,
            ["Partner", "Order date", "Order Time"],
            "Expected raw columns",
        ),
        build_datetime_column_check(
            raw_dataframe,
            cleaned_dataframe,
            ["Order date", "Order Time"],
            "Order datetime",
            "Order datetime build check",
        ),
        build_helper_columns_check(
            cleaned_dataframe,
            ["Date", "Year", "Month Number", "Month Name", "Hour", "Weekday"],
            "Order helper columns check",
        ),
    ]
