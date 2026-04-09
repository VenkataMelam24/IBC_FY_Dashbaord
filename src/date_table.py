"""Reusable date and hour dimensions for time-based analysis."""

from __future__ import annotations

import pandas as pd

DATE_TABLE_COLUMNS = [
    "Date",
    "Date Key",
    "Year",
    "Month Number",
    "Month Name",
    "Month Short",
    "Quarter",
    "Quarter Label",
    "Week Number",
    "Day",
    "Day Name",
    "Day Short",
    "Day of Week Number",
    "Day of Year",
    "Year Month",
    "Year Month Label",
    "Year Month Sort",
    "Month Start",
    "Month End",
    "Week Start",
    "Week End",
    "Is Weekend",
    "Is Weekday",
    "Financial Year Start",
    "Financial Year End",
    "Financial Year Label",
    "Financial Quarter",
    "Financial Month Number",
]

HOUR_TABLE_COLUMNS = [
    "Hour",
    "Hour Label",
    "Time Bucket",
]


def _empty_date_table() -> pd.DataFrame:
    """Return an empty date table with the expected columns."""
    date_table = pd.DataFrame(columns=DATE_TABLE_COLUMNS)
    date_table.attrs["source_min_date"] = None
    date_table.attrs["source_max_date"] = None
    date_table.attrs["calendar_start_date"] = None
    date_table.attrs["calendar_end_date"] = None
    date_table.attrs["forward_months"] = None
    date_table.attrs["source_datasets_used"] = []
    date_table.attrs["build_status"] = "missing_dates"
    return date_table


def _extract_date_series(
    dataframe: pd.DataFrame | None,
    date_column: str = "Date",
) -> pd.Series:
    """
    Pull one clean daily date series out of a cleaned fact table.

    We use the already-cleaned `Date` column from fact tables because:
    - POS data and delivery-order data already create it
    - this keeps the date dimension tied to real business data
    """
    if dataframe is None or dataframe.empty or date_column not in dataframe.columns:
        return pd.Series(dtype="datetime64[ns]")

    date_series = pd.to_datetime(dataframe[date_column], errors="coerce")
    date_series = date_series.dt.normalize().dropna()
    return date_series


def _format_financial_year_label(start_year: int, end_year: int) -> str:
    """Create labels like FY 2025-26."""
    return f"FY {start_year}-{str(end_year)[-2:]}"


def _format_hour_label(hour: int) -> str:
    """Convert a 24-hour number into a readable business label."""
    if hour == 0:
        return "12 AM"
    if hour < 12:
        return f"{hour} AM"
    if hour == 12:
        return "12 PM"
    return f"{hour - 12} PM"


def _assign_time_bucket(hour: int) -> str:
    """Group hours into simple restaurant-friendly time buckets."""
    if 6 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 21:
        return "Evening"
    return "Late Night"


def build_date_table(
    pos_dataframe: pd.DataFrame | None = None,
    delivery_orders_dataframe: pd.DataFrame | None = None,
    forward_months: int = 12,
) -> pd.DataFrame:
    """
    Build a daily-grain calendar table from the available fact datasets.

    Range logic:
    - start date = earliest available date from POS or delivery orders
    - end date = latest available date from POS or delivery orders + forward buffer

    The date table stays at daily grain on purpose.
    Hour-level analysis should stay in fact tables or an hour dimension.
    """
    source_dates: list[pd.Series] = []
    source_names_used: list[str] = []

    pos_dates = _extract_date_series(pos_dataframe)
    if not pos_dates.empty:
        source_dates.append(pos_dates)
        source_names_used.append("POS / Offline Sales")

    delivery_order_dates = _extract_date_series(delivery_orders_dataframe)
    if not delivery_order_dates.empty:
        source_dates.append(delivery_order_dates)
        source_names_used.append("Delivery Order-Level Data")

    if not source_dates:
        return _empty_date_table()

    combined_dates = pd.concat(source_dates, ignore_index=True)
    source_min_date = combined_dates.min().normalize()
    source_max_date = combined_dates.max().normalize()
    calendar_start_date = source_min_date
    calendar_end_date = (source_max_date + pd.DateOffset(months=forward_months)).normalize()

    date_range = pd.date_range(
        start=calendar_start_date,
        end=calendar_end_date,
        freq="D",
    )

    date_table = pd.DataFrame({"Date": date_range})

    date_table["Date Key"] = pd.to_numeric(
        date_table["Date"].dt.strftime("%Y%m%d"),
        errors="coerce",
    ).astype("Int64")
    date_table["Year"] = date_table["Date"].dt.year.astype("Int64")
    date_table["Month Number"] = date_table["Date"].dt.month.astype("Int64")
    date_table["Month Name"] = date_table["Date"].dt.strftime("%B")
    date_table["Month Short"] = date_table["Date"].dt.strftime("%b")
    date_table["Quarter"] = date_table["Date"].dt.quarter.astype("Int64")
    date_table["Quarter Label"] = "Q" + date_table["Quarter"].astype(str)
    date_table["Week Number"] = date_table["Date"].dt.isocalendar().week.astype("Int64")
    date_table["Day"] = date_table["Date"].dt.day.astype("Int64")
    date_table["Day Name"] = date_table["Date"].dt.day_name()
    date_table["Day Short"] = date_table["Date"].dt.strftime("%a")
    date_table["Day of Week Number"] = (date_table["Date"].dt.dayofweek + 1).astype("Int64")
    date_table["Day of Year"] = date_table["Date"].dt.dayofyear.astype("Int64")

    date_table["Year Month"] = date_table["Date"].dt.strftime("%Y-%m")
    date_table["Year Month Label"] = date_table["Date"].dt.strftime("%b %Y")
    date_table["Year Month Sort"] = pd.to_numeric(
        date_table["Date"].dt.strftime("%Y%m"),
        errors="coerce",
    ).astype("Int64")

    date_table["Month Start"] = date_table["Date"].dt.to_period("M").dt.to_timestamp()
    date_table["Month End"] = (
        date_table["Date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    )

    date_table["Week Start"] = (
        date_table["Date"]
        - pd.to_timedelta(date_table["Date"].dt.dayofweek, unit="D")
    ).dt.normalize()
    date_table["Week End"] = date_table["Week Start"] + pd.Timedelta(days=6)

    date_table["Is Weekend"] = date_table["Day of Week Number"] >= 6
    date_table["Is Weekday"] = ~date_table["Is Weekend"]

    date_table["Financial Year Start"] = (
        date_table["Year"]
        - (date_table["Month Number"] < 4).astype("Int64")
    ).astype("Int64")
    date_table["Financial Year End"] = (
        date_table["Financial Year Start"] + 1
    ).astype("Int64")
    date_table["Financial Month Number"] = (
        ((date_table["Month Number"] - 4) % 12) + 1
    ).astype("Int64")

    financial_quarter_number = (
        ((date_table["Financial Month Number"] - 1) // 3) + 1
    ).astype("Int64")
    date_table["Financial Quarter"] = "FQ" + financial_quarter_number.astype(str)

    date_table["Financial Year Label"] = [
        _format_financial_year_label(int(start_year), int(end_year))
        for start_year, end_year in zip(
            date_table["Financial Year Start"],
            date_table["Financial Year End"],
        )
    ]

    date_table = date_table[DATE_TABLE_COLUMNS]
    date_table.attrs["source_min_date"] = source_min_date
    date_table.attrs["source_max_date"] = source_max_date
    date_table.attrs["calendar_start_date"] = calendar_start_date
    date_table.attrs["calendar_end_date"] = calendar_end_date
    date_table.attrs["forward_months"] = forward_months
    date_table.attrs["source_datasets_used"] = source_names_used
    date_table.attrs["build_status"] = "success"
    return date_table


def build_hour_table() -> pd.DataFrame:
    """
    Build a small reusable hour dimension for future operational analysis.

    This stays separate from the date table because the main calendar
    dimension should remain daily grain.
    """
    hour_table = pd.DataFrame({"Hour": range(24)})
    hour_table["Hour Label"] = hour_table["Hour"].map(_format_hour_label)
    hour_table["Time Bucket"] = hour_table["Hour"].map(_assign_time_bucket)
    return hour_table[HOUR_TABLE_COLUMNS]
