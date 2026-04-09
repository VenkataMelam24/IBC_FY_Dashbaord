"""Dashboard page renderers for the clean shell layout."""

from __future__ import annotations

import textwrap
import unicodedata
from datetime import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cleaning import parse_number

FY_2025_2026_START = pd.Timestamp("2025-04-01")
FY_2025_2026_END = pd.Timestamp("2026-03-31")
OVERALL_SALES_QUARTER_OPTIONS = ["All Quarters", "Q1", "Q2", "Q3", "Q4"]
OVERALL_SALES_CHANNEL_OPTIONS = ["Overall", "Offline", "Online"]
GROWTH_CHART_CHANNEL_OPTIONS = ["Combined", "In-house", "Online"]
SCENARIO_IMPACT_OPTIONS = [-30, -20, -10, 0, 10, 20, 30]
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
SAGE_MIST_PAGE_BG = "#0F172A"
SAGE_MIST_SURFACE = "#111827"
SAGE_MIST_SURFACE_SOFT = "#1F2937"
SAGE_MIST_PRIMARY = "#14B8A6"
SAGE_MIST_ACCENT = "#34D399"
SAGE_MIST_SECONDARY = "#A78BFA"
SAGE_MIST_ONLINE = "#60A5FA"
SAGE_MIST_ALERT = "#FB7185"
SAGE_MIST_TEXT_STRONG = "#F8FAFC"
SAGE_MIST_TEXT_MUTED = "#94A3B8"
SAGE_MIST_BORDER = "rgba(148, 163, 184, 0.18)"
SAGE_MIST_GRID = "rgba(148, 163, 184, 0.14)"


def _apply_sage_mist_theme() -> None:
    """Apply the shared Sage Mist dashboard theme across the app."""
    st.markdown(
        f"""
        <style>
        :root {{
            --sage-bg: {SAGE_MIST_PAGE_BG};
            --sage-surface: {SAGE_MIST_SURFACE};
            --sage-soft: {SAGE_MIST_SURFACE_SOFT};
            --sage-primary: {SAGE_MIST_PRIMARY};
            --sage-accent: {SAGE_MIST_ACCENT};
            --sage-secondary: {SAGE_MIST_SECONDARY};
            --sage-online: {SAGE_MIST_ONLINE};
            --sage-alert: {SAGE_MIST_ALERT};
            --sage-text: {SAGE_MIST_TEXT_STRONG};
            --sage-muted: {SAGE_MIST_TEXT_MUTED};
            --sage-border: {SAGE_MIST_BORDER};
            --sage-grid: {SAGE_MIST_GRID};
            --sage-shadow: 0 18px 34px rgba(0, 0, 0, 0.32);
        }}
        .stApp {{
            background: var(--sage-bg);
            color: var(--sage-text);
        }}
        [data-testid="stAppViewContainer"] {{
            background: var(--sage-bg);
        }}
        [data-testid="stHeader"] {{
            background: rgba(17, 24, 39, 0.82);
        }}
        section[data-testid="stSidebar"] {{
            background: var(--sage-surface);
            border-right: 1px solid var(--sage-border);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--sage-text);
        }}
        section[data-testid="stSidebar"] [data-baseweb="radio"] > div {{
            color: var(--sage-text);
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: var(--sage-text);
        }}
        .stApp p, .stApp label, .stApp span, .stApp small {{
            color: var(--sage-muted);
        }}
        div[data-baseweb="select"] > div {{
            background: var(--sage-surface) !important;
            border-color: var(--sage-border) !important;
            color: var(--sage-text) !important;
            box-shadow: none !important;
        }}
        div[data-baseweb="select"] input {{
            color: var(--sage-text) !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--sage-soft);
            border: 1px solid var(--sage-border);
            border-radius: 24px;
            box-shadow: var(--sage-shadow);
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            background: transparent;
        }}
        [data-testid="stAlert"] {{
            background: rgba(251, 113, 133, 0.14);
            border: 1px solid rgba(251, 113, 133, 0.26);
            border-radius: 18px;
            color: var(--sage-text);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_table(cleaned_tables: dict[str, pd.DataFrame], table_name: str) -> pd.DataFrame:
    """Return a cleaned dataset safely."""
    return cleaned_tables.get(table_name, pd.DataFrame()).copy()


def _safe_sum(dataframe: pd.DataFrame, column_name: str) -> float:
    """Safely sum a numeric column."""
    if dataframe.empty or column_name not in dataframe.columns:
        return 0.0
    return float(pd.to_numeric(dataframe[column_name], errors="coerce").fillna(0).sum())


def _safe_row_count(dataframe: pd.DataFrame) -> int:
    """Return the number of rows in a table."""
    if dataframe.empty:
        return 0
    return int(len(dataframe.index))


def _filter_by_date_range(
    dataframe: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Keep only rows inside one inclusive date range."""
    if dataframe.empty or date_column not in dataframe.columns:
        return dataframe.iloc[0:0].copy()

    date_series = pd.to_datetime(dataframe[date_column], errors="coerce")
    mask = date_series.between(start_date, end_date, inclusive="both")
    return dataframe.loc[mask].copy()


def _filter_by_date_columns(
    dataframe: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    date_columns: list[str],
) -> pd.DataFrame:
    """Filter a dataset by the first usable date column from a list."""
    for column_name in date_columns:
        if column_name in dataframe.columns:
            return _filter_by_date_range(
                dataframe=dataframe,
                start_date=start_date,
                end_date=end_date,
                date_column=column_name,
            )
    return dataframe.iloc[0:0].copy()


def _build_valid_bill_series(dataframe: pd.DataFrame) -> pd.Series:
    """Return clean bill ids for distinct POS order counting."""
    if dataframe.empty or "Bill-nr." not in dataframe.columns:
        return pd.Series(dtype="object")

    bill_series = dataframe["Bill-nr."].dropna().astype(str).str.strip()
    valid_mask = (
        bill_series.ne("")
        & bill_series.str.lower().ne("nan")
        & bill_series.str.lower().ne("none")
    )
    return bill_series.loc[valid_mask]


def _count_valid_offline_orders(dataframe: pd.DataFrame) -> int:
    """
    Count distinct positive POS bills.

    Refund-only negative rows reduce sales but should not inflate order counts.
    """
    if dataframe.empty:
        return 0

    positive_sales = dataframe.copy()
    if "Amount" in positive_sales.columns:
        amount_series = pd.to_numeric(positive_sales["Amount"], errors="coerce").fillna(0)
        positive_sales = positive_sales.loc[amount_series > 0].copy()

    valid_bills = _build_valid_bill_series(positive_sales)
    if not valid_bills.empty:
        return int(valid_bills.nunique())

    return _safe_row_count(positive_sales)


def _count_partner_period_duplicates(dataframe: pd.DataFrame) -> int:
    """Count repeated partner-period rows in the delivery financial data."""
    if dataframe.empty or "Partner" not in dataframe.columns:
        return 0

    duplicate_keys = [
        column for column in ["Partner", "Date", "Month Start", "Month Number", "year"]
        if column in dataframe.columns
    ]
    if len(duplicate_keys) < 2:
        return 0

    return int(dataframe.duplicated(subset=duplicate_keys, keep=False).sum())


def _format_currency(value: float) -> str:
    """Format values as euro amounts."""
    return f"€{value:,.2f}"


def _format_whole_number(value: int) -> str:
    """Format whole numbers cleanly for KPI display."""
    return f"{int(value):,}"


def _format_percent(value: float) -> str:
    """Format one share value as a clean whole-number percentage."""
    return f"{round(value * 100):.0f}%"


def _format_month_support_text(month_count: int) -> str:
    """Format the active-month support text for the KPI card."""
    month_label = "month" if int(month_count) == 1 else "months"
    return f"across {int(month_count)} {month_label}"


def _format_percentage_value(value: float) -> str:
    """Format KPI percentage values cleanly."""
    return f"{value:.1f}%"


def _get_quarter_date_range(selected_quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the FY 2025-2026 date range for the selected quarter."""
    quarter_ranges = {
        "All Quarters": (FY_2025_2026_START, FY_2025_2026_END),
        "Q1": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30")),
        "Q2": (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30")),
        "Q3": (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")),
        "Q4": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")),
    }
    return quarter_ranges.get(selected_quarter, quarter_ranges["All Quarters"])


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Safely divide two numbers for share calculations."""
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _clamp_ratio(value: float) -> float:
    """Clamp a ratio into the 0 to 1 range for progress-bar widths."""
    return max(0.0, min(1.0, float(value)))


def _build_split_block_html(
    block_title: str,
    offline_value_text: str,
    online_value_text: str,
    offline_share_text: str,
    online_share_text: str,
    offline_bar_ratio: float,
    online_bar_ratio: float,
) -> str:
    """Build one dark-theme split block styled like the reference card."""
    offline_width = _clamp_ratio(offline_bar_ratio) * 100
    online_width = _clamp_ratio(online_bar_ratio) * 100
    offline_share_class = "split-share offline"
    online_share_class = "split-share online"
    offline_share_display = offline_share_text or "&nbsp;"
    online_share_display = online_share_text or "&nbsp;"

    if not offline_share_text:
        offline_share_class += " is-empty"
    if not online_share_text:
        online_share_class += " is-empty"

    return f"""
    <div class="split-card">
        <div class="split-card-title">{block_title}</div>
        <div class="split-card-layout">
            <div class="split-card-left">
                <div class="split-row">
                    <div class="split-row-head">
                        <span class="split-label">Offline</span>
                        <span class="{offline_share_class}">{offline_share_display}</span>
                    </div>
                    <div class="split-track">
                        <div class="split-fill offline" style="width: {offline_width:.1f}%;"></div>
                    </div>
                </div>
                <div class="split-row">
                    <div class="split-row-head">
                        <span class="split-label">Online</span>
                        <span class="{online_share_class}">{online_share_display}</span>
                    </div>
                    <div class="split-track">
                        <div class="split-fill online" style="width: {online_width:.1f}%;"></div>
                    </div>
                </div>
            </div>
            <div class="split-card-right">
                <div class="split-value-card offline">
                    <div class="split-value-label">Offline</div>
                    <div class="split-value-number offline">{offline_value_text}</div>
                </div>
                <div class="split-value-card online">
                    <div class="split-value-label">Online</div>
                    <div class="split-value-number online">{online_value_text}</div>
                </div>
            </div>
        </div>
    </div>
    """


def _build_kpi_card_html(label: str, value_text: str, support_text: str = "") -> str:
    """Build one KPI card with optional supporting text."""
    support_class = "kpi-card-support"
    support_display = support_text or "&nbsp;"
    if not support_text:
        support_class += " is-empty"

    return f"""
    <div class="kpi-card">
        <div class="kpi-card-label">{label}</div>
        <div class="kpi-card-value">{value_text}</div>
        <div class="{support_class}">{support_display}</div>
    </div>
    """


def _build_kpi_placeholder_html() -> str:
    """Build an invisible reserved KPI slot for future use."""
    return '<div class="kpi-card-placeholder" aria-hidden="true"></div>'


def _build_combined_monthly_gross_sales(
    pos_data_fy: pd.DataFrame,
    delivery_financials_data_fy: pd.DataFrame,
) -> pd.DataFrame:
    """Build one monthly combined gross-sales table for the FY period."""
    monthly_offline = pd.DataFrame(columns=["Month Start", "Offline Gross Sale"])
    if not pos_data_fy.empty and {"Date", "Amount"}.issubset(pos_data_fy.columns):
        monthly_offline = (
            pos_data_fy.assign(
                **{
                    "Month Start": pd.to_datetime(
                        pos_data_fy["Date"], errors="coerce"
                    ).dt.to_period("M").dt.to_timestamp()
                }
            )
            .dropna(subset=["Month Start"])
            .groupby("Month Start", as_index=False)["Amount"]
            .sum()
            .rename(columns={"Amount": "Offline Gross Sale"})
        )

    monthly_online = pd.DataFrame(columns=["Month Start", "Online Gross Sale"])
    online_month_column = None
    if "Month Start" in delivery_financials_data_fy.columns:
        online_month_column = "Month Start"
    elif "Date" in delivery_financials_data_fy.columns:
        online_month_column = "Date"

    if (
        not delivery_financials_data_fy.empty
        and online_month_column is not None
        and "Gross" in delivery_financials_data_fy.columns
    ):
        monthly_online = (
            delivery_financials_data_fy.assign(
                **{
                    "Month Start": pd.to_datetime(
                        delivery_financials_data_fy[online_month_column],
                        errors="coerce",
                    ).dt.to_period("M").dt.to_timestamp()
                }
            )
            .dropna(subset=["Month Start"])
            .groupby("Month Start", as_index=False)["Gross"]
            .sum()
            .rename(columns={"Gross": "Online Gross Sale"})
        )

    combined_monthly = monthly_offline.merge(
        monthly_online,
        on="Month Start",
        how="outer",
    ).sort_values("Month Start")

    if combined_monthly.empty:
        return combined_monthly

    combined_monthly["Offline Gross Sale"] = pd.to_numeric(
        combined_monthly["Offline Gross Sale"], errors="coerce"
    ).fillna(0)
    combined_monthly["Online Gross Sale"] = pd.to_numeric(
        combined_monthly["Online Gross Sale"], errors="coerce"
    ).fillna(0)
    combined_monthly["Gross Sale"] = (
        combined_monthly["Offline Gross Sale"] + combined_monthly["Online Gross Sale"]
    )
    return combined_monthly.reset_index(drop=True)


def _count_active_months(monthly_sales: pd.DataFrame, sales_column: str) -> int:
    """Count months where a given monthly sales series actually has non-zero activity."""
    if monthly_sales.empty or sales_column not in monthly_sales.columns:
        return 0

    sales_series = pd.to_numeric(monthly_sales[sales_column], errors="coerce").fillna(0)
    return int((sales_series != 0).sum())


def _calculate_average_month_growth(monthly_sales: pd.DataFrame, sales_column: str) -> float:
    """Calculate the average month-over-month growth percentage for one series."""
    if monthly_sales.empty or sales_column not in monthly_sales.columns:
        return 0.0

    sales_series = pd.to_numeric(monthly_sales[sales_column], errors="coerce")
    previous_series = sales_series.shift(1)
    valid_mask = previous_series.notna() & (previous_series != 0) & sales_series.notna()

    if not valid_mask.any():
        return 0.0

    growth_series = ((sales_series - previous_series) / previous_series) * 100
    return float(growth_series.loc[valid_mask].mean())


def _build_channel_sensitive_split(
    offline_value: float,
    online_value: float,
    selected_channel: str,
) -> dict[str, float]:
    """Build split-card values that react to the current channel filter."""
    if selected_channel == "Offline":
        return {
            "offline_value": float(offline_value),
            "online_value": 0.0,
            "offline_share": 1.0 if float(offline_value) else 0.0,
            "online_share": 0.0,
        }

    if selected_channel == "Online":
        return {
            "offline_value": 0.0,
            "online_value": float(online_value),
            "offline_share": 0.0,
            "online_share": 1.0 if float(online_value) else 0.0,
        }

    total_value = float(offline_value) + float(online_value)
    return {
        "offline_value": float(offline_value),
        "online_value": float(online_value),
        "offline_share": _safe_ratio(offline_value, total_value),
        "online_share": _safe_ratio(online_value, total_value),
    }


def _prepare_sales_breakdown_data(
    cleaned_tables: dict[str, pd.DataFrame],
    selected_quarter: str,
    selected_channel: str = "Overall",
) -> pd.DataFrame:
    """Prepare monthly offline, online, and total gross sales for the charts."""
    pos_data = _get_table(cleaned_tables, "pos")
    delivery_financials_data = _get_table(cleaned_tables, "delivery_financials")
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)

    pos_data_filtered = _filter_by_date_range(
        pos_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
    )
    delivery_financials_filtered = _filter_by_date_columns(
        delivery_financials_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Date", "Month Start"],
    )

    if selected_channel == "Offline":
        delivery_financials_filtered = delivery_financials_filtered.iloc[0:0].copy()
    elif selected_channel == "Online":
        pos_data_filtered = pos_data_filtered.iloc[0:0].copy()

    monthly_sales = _build_combined_monthly_gross_sales(
        pos_data_fy=pos_data_filtered,
        delivery_financials_data_fy=delivery_financials_filtered,
    )
    if monthly_sales.empty:
        return monthly_sales

    monthly_sales["Month Start"] = pd.to_datetime(
        monthly_sales["Month Start"], errors="coerce"
    )
    monthly_sales = monthly_sales.dropna(subset=["Month Start"]).sort_values(
        "Month Start"
    )
    monthly_sales["Month Short"] = monthly_sales["Month Start"].dt.strftime("%b")
    monthly_sales["Month Label"] = monthly_sales["Month Start"].dt.strftime("%b %Y")
    return monthly_sales.reset_index(drop=True)


def _count_scope_weeks(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """Return the exact inclusive date-span expressed in weeks."""
    total_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    if total_days <= 0:
        return 0.0
    return float(total_days) / 7.0


def _format_compact_count(value: float) -> str:
    """Format order counts in a compact readable style."""
    numeric_value = float(value)
    if numeric_value >= 1000:
        return f"{numeric_value / 1000:.1f}k"
    return f"{int(round(numeric_value)):,}"


def _prepare_orders_breakdown_data(
    cleaned_tables: dict[str, pd.DataFrame],
    selected_quarter: str,
    selected_channel: str,
) -> pd.DataFrame:
    """Prepare monthly offline, online, and total order counts for the charts."""
    pos_data = _get_table(cleaned_tables, "pos")
    delivery_financials_data = _get_table(cleaned_tables, "delivery_financials")
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)

    pos_data_filtered = _filter_by_date_range(
        pos_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
    )
    delivery_financials_filtered = _filter_by_date_columns(
        delivery_financials_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Date", "Month Start"],
    )

    if selected_channel == "Offline":
        delivery_financials_filtered = delivery_financials_filtered.iloc[0:0].copy()
    elif selected_channel == "Online":
        pos_data_filtered = pos_data_filtered.iloc[0:0].copy()

    monthly_offline = pd.DataFrame(columns=["Month Start", "Offline Orders"])
    if (
        not pos_data_filtered.empty
        and {"Date", "Bill-nr."}.issubset(pos_data_filtered.columns)
    ):
        positive_pos = pos_data_filtered.copy()
        if "Amount" in positive_pos.columns:
            amount_series = pd.to_numeric(
                positive_pos["Amount"], errors="coerce"
            ).fillna(0)
            positive_pos = positive_pos.loc[amount_series > 0].copy()

        if not positive_pos.empty:
            positive_pos["Month Start"] = pd.to_datetime(
                positive_pos["Date"], errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            bill_series = _build_valid_bill_series(positive_pos)
            if not bill_series.empty:
                positive_pos = positive_pos.loc[bill_series.index].copy()
                positive_pos["Bill Clean"] = bill_series
                monthly_offline = (
                    positive_pos.dropna(subset=["Month Start"])
                    .groupby("Month Start", as_index=False)["Bill Clean"]
                    .nunique()
                    .rename(columns={"Bill Clean": "Offline Orders"})
                )

    monthly_online = pd.DataFrame(columns=["Month Start", "Online Orders"])
    online_month_column = None
    if "Month Start" in delivery_financials_filtered.columns:
        online_month_column = "Month Start"
    elif "Date" in delivery_financials_filtered.columns:
        online_month_column = "Date"

    if (
        not delivery_financials_filtered.empty
        and online_month_column is not None
        and "orders" in delivery_financials_filtered.columns
    ):
        monthly_online = (
            delivery_financials_filtered.assign(
                **{
                    "Month Start": pd.to_datetime(
                        delivery_financials_filtered[online_month_column],
                        errors="coerce",
                    ).dt.to_period("M").dt.to_timestamp()
                }
            )
            .dropna(subset=["Month Start"])
            .groupby("Month Start", as_index=False)["orders"]
            .sum()
            .rename(columns={"orders": "Online Orders"})
        )

    combined_monthly = monthly_offline.merge(
        monthly_online,
        on="Month Start",
        how="outer",
    ).sort_values("Month Start")

    if combined_monthly.empty:
        return combined_monthly

    combined_monthly["Offline Orders"] = pd.to_numeric(
        combined_monthly["Offline Orders"], errors="coerce"
    ).fillna(0)
    combined_monthly["Online Orders"] = pd.to_numeric(
        combined_monthly["Online Orders"], errors="coerce"
    ).fillna(0)
    combined_monthly["Total Orders"] = (
        combined_monthly["Offline Orders"] + combined_monthly["Online Orders"]
    )
    combined_monthly["Month Start"] = pd.to_datetime(
        combined_monthly["Month Start"], errors="coerce"
    )
    combined_monthly = combined_monthly.dropna(subset=["Month Start"]).sort_values(
        "Month Start"
    )
    combined_monthly["Month Short"] = combined_monthly["Month Start"].dt.strftime("%b")
    combined_monthly["Month Label"] = combined_monthly["Month Start"].dt.strftime("%b %Y")
    return combined_monthly.reset_index(drop=True)


def _prepare_weekday_orders_analysis_data(
    cleaned_tables: dict[str, pd.DataFrame],
    selected_quarter: str,
    selected_channel: str,
) -> pd.DataFrame:
    """
    Prepare weekday order averages from the two operational source tables only.

    Sources used on purpose:
    - POS / offline transactions -> weekday derived from `Time-Stamp`
    - Delivery order-level table -> weekday derived from `Order datetime`
      which is built from `Order date` + `Order Time`

    This function does NOT use the monthly delivery financial summary.
    """
    pos_data = _get_table(cleaned_tables, "pos")
    delivery_orders_data = _get_table(cleaned_tables, "delivery_orders")
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)

    # Use the transaction timestamp for offline weekday behavior.
    pos_data_filtered = _filter_by_date_columns(
        pos_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Time-Stamp", "Date"],
    )
    # Use the order-level datetime built from Order date + Order Time.
    delivery_orders_filtered = _filter_by_date_columns(
        delivery_orders_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Order datetime", "Date"],
    )

    if selected_channel == "Offline":
        delivery_orders_filtered = delivery_orders_filtered.iloc[0:0].copy()
    elif selected_channel == "Online":
        pos_data_filtered = pos_data_filtered.iloc[0:0].copy()

    weekday_occurrence_series = (
        pd.Series(
            pd.date_range(selected_start_date, selected_end_date, freq="D").day_name()
        )
        .value_counts()
        .reindex(WEEKDAY_ORDER)
        .fillna(0)
    )

    offline_totals = pd.Series(0.0, index=WEEKDAY_ORDER, dtype="float64")
    if (
        not pos_data_filtered.empty
        and {"Weekday", "Bill-nr."}.issubset(pos_data_filtered.columns)
    ):
        positive_pos = pos_data_filtered.copy()
        if "Amount" in positive_pos.columns:
            amount_series = pd.to_numeric(positive_pos["Amount"], errors="coerce").fillna(0)
            positive_pos = positive_pos.loc[amount_series > 0].copy()

        if not positive_pos.empty:
            bill_series = _build_valid_bill_series(positive_pos)
            if not bill_series.empty:
                positive_pos = positive_pos.loc[bill_series.index].copy()
                positive_pos["Bill Clean"] = bill_series
                offline_counts = (
                    positive_pos.dropna(subset=["Weekday"])
                    .groupby("Weekday")["Bill Clean"]
                    .nunique()
                    .reindex(WEEKDAY_ORDER)
                    .fillna(0)
                )
                offline_totals = pd.to_numeric(offline_counts, errors="coerce").fillna(0)

    online_totals = pd.Series(0.0, index=WEEKDAY_ORDER, dtype="float64")
    if not delivery_orders_filtered.empty and "Weekday" in delivery_orders_filtered.columns:
        online_counts = (
            delivery_orders_filtered.dropna(subset=["Weekday"])
            .groupby("Weekday")
            .size()
            .reindex(WEEKDAY_ORDER)
            .fillna(0)
        )
        online_totals = pd.to_numeric(online_counts, errors="coerce").fillna(0)

    weekday_summary = pd.DataFrame(
        {
            "Weekday": WEEKDAY_ORDER,
            "Occurrence Count": weekday_occurrence_series.astype("float64").values,
            "Offline Orders": offline_totals.values,
            "Online Orders": online_totals.values,
        }
    )

    valid_occurrence_mask = weekday_summary["Occurrence Count"] > 0
    weekday_summary = weekday_summary.loc[valid_occurrence_mask].copy()
    if weekday_summary.empty:
        return weekday_summary

    weekday_summary["Offline Average Orders"] = (
        weekday_summary["Offline Orders"] / weekday_summary["Occurrence Count"]
    )
    weekday_summary["Online Average Orders"] = (
        weekday_summary["Online Orders"] / weekday_summary["Occurrence Count"]
    )
    weekday_summary["Total Average Orders"] = (
        weekday_summary["Offline Average Orders"]
        + weekday_summary["Online Average Orders"]
    )

    weekday_summary = weekday_summary.loc[
        weekday_summary["Total Average Orders"] > 0
    ].copy()
    if weekday_summary.empty:
        return weekday_summary

    weekday_summary = weekday_summary.sort_values(
        "Total Average Orders",
        ascending=False,
    ).reset_index(drop=True)
    _log_weekday_analysis_debug(
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
        pos_data_filtered=pos_data_filtered,
        delivery_orders_filtered=delivery_orders_filtered,
        weekday_summary=weekday_summary,
    )
    return weekday_summary


def _log_weekday_analysis_debug(
    selected_quarter: str,
    selected_channel: str,
    pos_data_filtered: pd.DataFrame,
    delivery_orders_filtered: pd.DataFrame,
    weekday_summary: pd.DataFrame,
) -> None:
    """Print one terminal-only validation block for weekday analysis sources."""
    print(
        "WEEKDAY_ANALYSIS_DEBUG",
        {
            "selected_quarter": selected_quarter,
            "selected_channel": selected_channel,
            "offline_source": "POS Time-Stamp -> distinct positive Bill-nr.",
            "online_source": "Delivery order-level Order date + Order Time -> order rows",
            "uses_delivery_financial_summary": False,
            "offline_rows_after_filter": int(len(pos_data_filtered.index)),
            "online_rows_after_filter": int(len(delivery_orders_filtered.index)),
        },
    )
    if not weekday_summary.empty:
        debug_columns = [
            column_name
            for column_name in [
                "Weekday",
                "Occurrence Count",
                "Offline Orders",
                "Online Orders",
                "Offline Average Orders",
                "Online Average Orders",
                "Total Average Orders",
            ]
            if column_name in weekday_summary.columns
        ]
        print(weekday_summary[debug_columns].to_string(index=False))


def _format_hour_label(hour_value: int | float) -> str:
    """Format 24-hour integers into clean 12-hour business labels."""
    hour_int = int(hour_value)
    suffix = "AM" if hour_int < 12 else "PM"
    hour_12 = hour_int % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12} {suffix}"


def _format_hour_bucket_label(hour_value: int | float) -> str:
    """Format one hour into an explicit one-hour bucket such as 12pm–1pm."""
    start_hour = _format_hour_label(int(hour_value)).replace(" ", "").lower()
    end_hour = _format_hour_label((int(hour_value) + 1) % 24).replace(" ", "").lower()
    return f"{start_hour}–{end_hour}"


def _prepare_hourly_orders_analysis_data(
    cleaned_tables: dict[str, pd.DataFrame],
    selected_quarter: str,
    selected_channel: str,
) -> pd.DataFrame:
    """
    Prepare hourly average orders from the two operational source tables only.

    Sources used on purpose:
    - POS / offline transactions -> filtered by `Time-Stamp`
    - Delivery order-level table -> filtered by `Order datetime`

    Excluded on purpose:
    - Delivery financial summary table
    """
    pos_data = _get_table(cleaned_tables, "pos")
    delivery_orders_data = _get_table(cleaned_tables, "delivery_orders")
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)

    pos_data_filtered = _filter_by_date_columns(
        pos_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Time-Stamp", "Date"],
    )
    delivery_orders_filtered = _filter_by_date_columns(
        delivery_orders_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Order datetime", "Date"],
    )

    if selected_channel == "Offline":
        delivery_orders_filtered = delivery_orders_filtered.iloc[0:0].copy()
    elif selected_channel == "Online":
        pos_data_filtered = pos_data_filtered.iloc[0:0].copy()

    total_day_count = float(len(pd.date_range(selected_start_date, selected_end_date, freq="D")))

    hourly_index = pd.Index(range(24), name="Hour")
    offline_totals = pd.Series(0.0, index=hourly_index, dtype="float64")
    if not pos_data_filtered.empty and {"Hour", "Bill-nr."}.issubset(pos_data_filtered.columns):
        positive_pos = pos_data_filtered.copy()
        if "Amount" in positive_pos.columns:
            amount_series = pd.to_numeric(positive_pos["Amount"], errors="coerce").fillna(0)
            positive_pos = positive_pos.loc[amount_series > 0].copy()

        if not positive_pos.empty:
            bill_series = _build_valid_bill_series(positive_pos)
            if not bill_series.empty:
                positive_pos = positive_pos.loc[bill_series.index].copy()
                positive_pos["Bill Clean"] = bill_series
                offline_counts = (
                    positive_pos.dropna(subset=["Hour"])
                    .groupby("Hour")["Bill Clean"]
                    .nunique()
                    .reindex(hourly_index)
                    .fillna(0)
                )
                offline_totals = pd.to_numeric(offline_counts, errors="coerce").fillna(0)

    online_totals = pd.Series(0.0, index=hourly_index, dtype="float64")
    if not delivery_orders_filtered.empty and "Hour" in delivery_orders_filtered.columns:
        online_counts = (
            delivery_orders_filtered.dropna(subset=["Hour"])
            .groupby("Hour")
            .size()
            .reindex(hourly_index)
            .fillna(0)
        )
        online_totals = pd.to_numeric(online_counts, errors="coerce").fillna(0)

    hourly_summary = pd.DataFrame(
        {
            "Hour": hourly_index.astype(int),
            "Occurrence Count": total_day_count,
            "Offline Orders": offline_totals.values,
            "Online Orders": online_totals.values,
        }
    )
    if hourly_summary.empty:
        return hourly_summary

    hourly_summary["Offline Average Orders"] = (
        hourly_summary["Offline Orders"] / hourly_summary["Occurrence Count"]
    )
    hourly_summary["Online Average Orders"] = (
        hourly_summary["Online Orders"] / hourly_summary["Occurrence Count"]
    )
    hourly_summary["Total Average Orders"] = (
        hourly_summary["Offline Average Orders"]
        + hourly_summary["Online Average Orders"]
    )
    hourly_summary["Hour Label"] = hourly_summary["Hour"].map(_format_hour_label)

    hourly_summary = hourly_summary.loc[
        hourly_summary["Total Average Orders"] > 0
    ].copy()
    if hourly_summary.empty:
        return hourly_summary

    hourly_summary = hourly_summary.sort_values("Hour").reset_index(drop=True)
    _log_hourly_analysis_debug(
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
        pos_data_filtered=pos_data_filtered,
        delivery_orders_filtered=delivery_orders_filtered,
        hourly_summary=hourly_summary,
    )
    return hourly_summary


def _find_column_name(dataframe: pd.DataFrame, target_names: list[str]) -> str | None:
    """Find one column by normalized label without changing the raw table itself."""
    normalized_map = {
        str(column_name).strip().lower(): str(column_name)
        for column_name in dataframe.columns
    }
    for target_name in target_names:
        resolved_name = normalized_map.get(str(target_name).strip().lower())
        if resolved_name is not None:
            return resolved_name
    return None


def _prepare_hourly_exact_q4_data(
    cleaned_tables: dict[str, pd.DataFrame],
    raw_tables: dict[str, pd.DataFrame],
    selected_quarter: str,
    selected_channel: str,
) -> pd.DataFrame:
    """
    Build exact Q4 hourly orders and sales totals for Hourly Analysis.

    Sources used on purpose:
    - POS table for offline orders and offline sales
    - Raw delivery partner published CSV source for online orders and online sales

    Excluded on purpose:
    - Delivery financial summary table
    - Approximate AOV-based hourly sales logic
    """
    pos_data = _get_table(cleaned_tables, "pos")
    delivery_partner_raw = raw_tables.get("delivery_partner_raw", pd.DataFrame()).copy()

    q4_start = pd.Timestamp("2026-01-01")
    q4_end = pd.Timestamp("2026-03-31")
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)
    scope_start = max(selected_start_date, q4_start)
    scope_end = min(selected_end_date, q4_end)
    day_count_in_scope = float(len(pd.date_range(scope_start, scope_end, freq="D")))

    if scope_start > scope_end:
        return pd.DataFrame(
            columns=[
                "Hour",
                "Hour Label",
                "Offline Orders",
                "Online Orders",
                "Total Orders",
                "Offline Sales",
                "Online Sales",
                "Total Sales",
            ]
        )

    if not pos_data.empty and "Time-Stamp" in pos_data.columns:
        pos_timestamp_series = pd.to_datetime(pos_data["Time-Stamp"], errors="coerce")
        pos_data_filtered = pos_data.loc[
            pos_timestamp_series.between(
                scope_start,
                scope_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            )
        ].copy()
    else:
        pos_data_filtered = _filter_by_date_columns(
            pos_data,
            start_date=scope_start,
            end_date=scope_end,
            date_columns=["Date"],
        )

    if selected_channel == "Offline":
        delivery_partner_raw = delivery_partner_raw.iloc[0:0].copy()
    elif selected_channel == "Online":
        pos_data_filtered = pos_data_filtered.iloc[0:0].copy()

    hourly_index = pd.Index(range(24), name="Hour")

    offline_order_totals = pd.Series(0.0, index=hourly_index, dtype="float64")
    offline_sales_totals = pd.Series(0.0, index=hourly_index, dtype="float64")
    if not pos_data_filtered.empty and "Hour" in pos_data_filtered.columns:
        if "Amount" in pos_data_filtered.columns:
            amount_series = pd.to_numeric(
                pos_data_filtered["Amount"], errors="coerce"
            ).fillna(0)
            offline_sales_counts = (
                pos_data_filtered.assign(HourValue=pd.to_numeric(pos_data_filtered["Hour"], errors="coerce"))
                .dropna(subset=["HourValue"])
                .groupby("HourValue")["Amount"]
                .sum()
                .reindex(hourly_index)
                .fillna(0)
            )
            offline_sales_totals = pd.to_numeric(
                offline_sales_counts, errors="coerce"
            ).fillna(0)

        if {"Bill-nr.", "Amount"}.issubset(pos_data_filtered.columns):
            positive_pos = pos_data_filtered.copy()
            amount_series = pd.to_numeric(
                positive_pos["Amount"], errors="coerce"
            ).fillna(0)
            positive_pos = positive_pos.loc[amount_series > 0].copy()
            if not positive_pos.empty:
                bill_series = _build_valid_bill_series(positive_pos)
                if not bill_series.empty:
                    positive_pos = positive_pos.loc[bill_series.index].copy()
                    positive_pos["Bill Clean"] = bill_series
                    positive_pos["HourValue"] = pd.to_numeric(
                        positive_pos["Hour"], errors="coerce"
                    )
                    offline_counts = (
                        positive_pos.dropna(subset=["HourValue"])
                        .groupby("HourValue")["Bill Clean"]
                        .nunique()
                        .reindex(hourly_index)
                        .fillna(0)
                    )
                    offline_order_totals = pd.to_numeric(
                        offline_counts, errors="coerce"
                    ).fillna(0)

    online_order_totals = pd.Series(0.0, index=hourly_index, dtype="float64")
    online_sales_totals = pd.Series(0.0, index=hourly_index, dtype="float64")
    online_rows_after_filter = 0
    if not delivery_partner_raw.empty:
        # This raw published Google Sheet source covers Jan to Mar delivery partner data.
        date_column = _find_column_name(delivery_partner_raw, ["Date", "Date "])
        time_column = _find_column_name(delivery_partner_raw, ["Time"])
        sale_column = _find_column_name(delivery_partner_raw, ["Sale"])

        if date_column is not None and time_column is not None:
            online_data = delivery_partner_raw.copy()
            online_datetime = pd.to_datetime(
                online_data[date_column].astype(str).str.strip()
                + " "
                + online_data[time_column].astype(str).str.strip(),
                errors="coerce",
            )
            online_data["Order Datetime Raw"] = online_datetime
            online_data["HourValue"] = online_datetime.dt.hour
            online_data = online_data.loc[
                online_data["Order Datetime Raw"].between(scope_start, scope_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            ].copy()
            online_rows_after_filter = int(len(online_data.index))

            if not online_data.empty:
                online_counts = (
                    online_data.dropna(subset=["HourValue"])
                    .groupby("HourValue")
                    .size()
                    .reindex(hourly_index)
                    .fillna(0)
                )
                online_order_totals = pd.to_numeric(
                    online_counts, errors="coerce"
                ).fillna(0)

                if sale_column is not None:
                    online_data["Sale Numeric"] = pd.to_numeric(
                        online_data[sale_column].map(parse_number), errors="coerce"
                    ).fillna(0)
                    online_sales = (
                        online_data.dropna(subset=["HourValue"])
                        .groupby("HourValue")["Sale Numeric"]
                        .sum()
                        .reindex(hourly_index)
                        .fillna(0)
                    )
                    online_sales_totals = pd.to_numeric(
                        online_sales, errors="coerce"
                    ).fillna(0)

    hourly_summary = pd.DataFrame(
        {
            "Hour": hourly_index.astype(int),
            "Offline Orders": offline_order_totals.values,
            "Online Orders": online_order_totals.values,
            "Offline Sales": offline_sales_totals.values,
            "Online Sales": online_sales_totals.values,
        }
    )
    hourly_summary["Total Orders"] = (
        hourly_summary["Offline Orders"] + hourly_summary["Online Orders"]
    )
    hourly_summary["Offline Average Orders"] = (
        hourly_summary["Offline Orders"] / day_count_in_scope
        if day_count_in_scope
        else 0.0
    )
    hourly_summary["Online Average Orders"] = (
        hourly_summary["Online Orders"] / day_count_in_scope
        if day_count_in_scope
        else 0.0
    )
    hourly_summary["Total Average Orders"] = (
        hourly_summary["Total Orders"] / day_count_in_scope
        if day_count_in_scope
        else 0.0
    )
    hourly_summary["Total Sales"] = (
        hourly_summary["Offline Sales"] + hourly_summary["Online Sales"]
    )
    hourly_summary["Offline Average Sales"] = (
        hourly_summary["Offline Sales"] / day_count_in_scope
        if day_count_in_scope
        else 0.0
    )
    hourly_summary["Online Average Sales"] = (
        hourly_summary["Online Sales"] / day_count_in_scope
        if day_count_in_scope
        else 0.0
    )
    hourly_summary["Total Average Sales"] = (
        hourly_summary["Total Sales"] / day_count_in_scope
        if day_count_in_scope
        else 0.0
    )
    hourly_summary["Hour Label"] = hourly_summary["Hour"].map(_format_hour_label)
    hourly_summary = hourly_summary.loc[
        (hourly_summary["Hour"] >= 12)
        & ((hourly_summary["Total Orders"] > 0) | (hourly_summary["Total Sales"] != 0))
    ].copy()
    if hourly_summary.empty:
        return hourly_summary

    hourly_summary = hourly_summary.sort_values("Hour").reset_index(drop=True)
    _log_hourly_exact_q4_debug(
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
        scope_start=scope_start,
        scope_end=scope_end,
        day_count_in_scope=day_count_in_scope,
        pos_data_filtered=pos_data_filtered,
        online_rows_after_filter=online_rows_after_filter,
        hourly_summary=hourly_summary,
    )
    return hourly_summary


def _log_hourly_exact_q4_debug(
    selected_quarter: str,
    selected_channel: str,
    scope_start: pd.Timestamp,
    scope_end: pd.Timestamp,
    day_count_in_scope: float,
    pos_data_filtered: pd.DataFrame,
    online_rows_after_filter: int,
    hourly_summary: pd.DataFrame,
) -> None:
    """Print one terminal-only validation block for exact Q4 hourly analysis."""
    print(
        "HOURLY_ANALYSIS_Q4_DEBUG",
        {
            "selected_quarter": selected_quarter,
            "selected_channel": selected_channel,
            "scope_start": str(pd.Timestamp(scope_start).date()),
            "scope_end": str(pd.Timestamp(scope_end).date()),
            "day_count_in_scope": float(day_count_in_scope),
            "offline_source": "POS Time-Stamp -> distinct positive Bill-nr. + Amount",
            "online_source": "delivery_partner_raw_url raw table -> Date + Time + Sale",
            "uses_delivery_financial_summary": False,
            "offline_rows_after_filter": int(len(pos_data_filtered.index)),
            "online_rows_after_filter": int(online_rows_after_filter),
        },
    )
    if not hourly_summary.empty:
        debug_columns = [
            column_name
            for column_name in [
                "Hour",
                "Hour Label",
                "Offline Orders",
                "Online Orders",
                "Total Orders",
                "Offline Average Orders",
                "Online Average Orders",
                "Total Average Orders",
                "Offline Sales",
                "Online Sales",
                "Total Sales",
                "Offline Average Sales",
                "Online Average Sales",
                "Total Average Sales",
            ]
            if column_name in hourly_summary.columns
        ]
        print(hourly_summary[debug_columns].to_string(index=False))


def _log_hourly_analysis_debug(
    selected_quarter: str,
    selected_channel: str,
    pos_data_filtered: pd.DataFrame,
    delivery_orders_filtered: pd.DataFrame,
    hourly_summary: pd.DataFrame,
) -> None:
    """Print one terminal-only validation block for hourly analysis sources."""
    print(
        "HOURLY_ANALYSIS_DEBUG",
        {
            "selected_quarter": selected_quarter,
            "selected_channel": selected_channel,
            "offline_source": "POS Time-Stamp -> distinct positive Bill-nr.",
            "online_source": "Delivery order-level Order date + Order Time -> order rows",
            "uses_delivery_financial_summary": False,
            "offline_rows_after_filter": int(len(pos_data_filtered.index)),
            "online_rows_after_filter": int(len(delivery_orders_filtered.index)),
        },
    )
    if not hourly_summary.empty:
        debug_columns = [
            column_name
            for column_name in [
                "Hour",
                "Hour Label",
                "Offline Orders",
                "Online Orders",
                "Offline Average Orders",
                "Online Average Orders",
                "Total Average Orders",
            ]
            if column_name in hourly_summary.columns
        ]
        print(hourly_summary[debug_columns].to_string(index=False))


def _build_sales_breakdown_legend_html() -> str:
    """Render the compact legend displayed at the top-right of the section."""
    return """
    <div class="sales-breakdown-legend">
        <span class="sales-breakdown-legend-item">
            <span class="sales-breakdown-dot offline"></span>Offline
        </span>
        <span class="sales-breakdown-legend-item">
            <span class="sales-breakdown-dot online"></span>Online
        </span>
        <span class="sales-breakdown-legend-item">
            <span class="sales-breakdown-dot total"></span>Total
        </span>
    </div>
    """


def _build_chart_card_header_html(title: str, subtitle: str) -> str:
    """Render a consistent title and subtitle block above each chart."""
    return f"""
    <div class="sales-chart-header">
        <div class="sales-chart-title">{title}</div>
        <div class="sales-chart-subtitle">{subtitle}</div>
    </div>
    """


def _build_month_analysis_legend_html() -> str:
    """Render the compact legend for the month analysis section."""
    return textwrap.dedent(
        """
        <div class="month-analysis-legend">
            <span class="month-analysis-legend-item offline">
                <span class="month-analysis-legend-dot offline"></span>Offline
            </span>
            <span class="month-analysis-legend-item online">
                <span class="month-analysis-legend-dot online"></span>Online
            </span>
        </div>
        """
    ).strip()


def _build_month_analysis_rank_card_html(
    title: str,
    month_rows: pd.DataFrame,
    tone_class: str,
    max_total_orders: float,
) -> str:
    """Render one right-side month insight card as a clean HTML block."""
    if month_rows.empty:
        rows_html = (
            '<div class="month-analysis-card-empty">'
            "No monthly order pattern is available for the current filters."
            "</div>"
        )
    else:
        row_html_parts: list[str] = []
        for _, row in month_rows.iterrows():
            total_orders = float(row.get("Total Orders", 0))
            fill_ratio = (total_orders / max_total_orders) if max_total_orders else 0.0
            month_label = str(row.get("Month Short", ""))
            row_html_parts.append(
                (
                    '<div class="month-analysis-rank-row">'
                    f'<div class="month-analysis-rank-month">{month_label}</div>'
                    '<div class="month-analysis-rank-track">'
                    f'<div class="month-analysis-rank-fill {tone_class}" style="width: {fill_ratio * 100:.1f}%"></div>'
                    "</div>"
                    f'<div class="month-analysis-rank-value {tone_class}">{_format_whole_number(int(round(total_orders)))}</div>'
                    "</div>"
                )
            )
        rows_html = (
            '<div class="month-analysis-rank-stack">'
            + "".join(row_html_parts)
            + "</div>"
        )

    return (
        '<div class="month-analysis-side-card">'
        f'<div class="month-analysis-card-title">{title}</div>'
        + rows_html
        + "</div>"
    )


def _build_week_analysis_legend_html() -> str:
    """Render the compact legend for the weekday analysis chart."""
    return textwrap.dedent(
        """
        <div class="week-analysis-legend">
            <span class="week-analysis-legend-item offline">
                <span class="week-analysis-legend-dot offline"></span>Offline
            </span>
            <span class="week-analysis-legend-item online">
                <span class="week-analysis-legend-dot online"></span>Online
            </span>
        </div>
        """
    ).strip()


def _format_average_orders(value: float) -> str:
    """Format average-order values cleanly for weekday insights."""
    numeric_value = float(value)
    if abs(numeric_value - round(numeric_value)) < 0.05:
        return f"{int(round(numeric_value))}"
    return f"{numeric_value:.1f}"


def _format_compact_currency_label(value: float) -> str:
    """Format one currency label compactly for in-bar display."""
    numeric_value = float(value)
    if abs(numeric_value) >= 1000:
        return f"€{numeric_value / 1000:.1f}k"
    if abs(numeric_value) >= 100:
        return f"€{numeric_value:,.0f}"
    return f"€{numeric_value:,.1f}"


def _classify_hourly_peak_bands(values: pd.Series) -> pd.Series:
    """Classify one plotted hourly series into high, medium, and dead/low bands."""
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0)
    classifications = pd.Series("Dead / low", index=numeric_values.index, dtype="object")

    positive_values = numeric_values.loc[numeric_values > 0].copy()
    if positive_values.empty:
        return classifications

    if len(positive_values.index) == 1:
        classifications.loc[positive_values.index] = "Medium peak"
        return classifications

    positive_ranks = positive_values.rank(method="average", pct=True)
    classifications.loc[positive_values.index] = "Medium peak"
    classifications.loc[positive_ranks.index[positive_ranks <= 0.34]] = "Dead / low"
    classifications.loc[positive_ranks.index[positive_ranks >= 0.67]] = "High peak"
    return classifications


def _get_hourly_peak_palette() -> dict[str, str]:
    """Return the display colors for the hourly peak classification legend and markers."""
    return {
        "High peak": SAGE_MIST_ACCENT,
        "Medium peak": SAGE_MIST_SECONDARY,
        "Dead / low": SAGE_MIST_ALERT,
    }


def _build_hourly_analysis_legend_html() -> str:
    """Render the compact peak-band legend for Hourly Analysis."""
    return textwrap.dedent(
        """
        <div class="hourly-analysis-legend">
            <span class="hourly-analysis-legend-item high">
                <span class="hourly-analysis-legend-dot high"></span>High peak
            </span>
            <span class="hourly-analysis-legend-item medium">
                <span class="hourly-analysis-legend-dot medium"></span>Medium peak
            </span>
            <span class="hourly-analysis-legend-item low">
                <span class="hourly-analysis-legend-dot low"></span>Dead / low
            </span>
        </div>
        """
    ).strip()


def _build_weekday_analysis_rank_card_html(
    title: str,
    weekday_rows: pd.DataFrame,
    tone_class: str,
    max_total_orders: float,
) -> str:
    """Render one weekday insight card as a clean HTML block."""
    if weekday_rows.empty:
        rows_html = (
            '<div class="month-analysis-card-empty">'
            "No weekday order pattern is available for the current filters."
            "</div>"
        )
    else:
        row_html_parts: list[str] = []
        for _, row in weekday_rows.iterrows():
            total_orders = float(row.get("Total Average Orders", 0))
            fill_ratio = (total_orders / max_total_orders) if max_total_orders else 0.0
            weekday_label = str(row.get("Weekday", ""))
            row_html_parts.append(
                (
                    '<div class="weekday-analysis-rank-row">'
                    f'<div class="weekday-analysis-rank-day">{weekday_label}</div>'
                    '<div class="month-analysis-rank-track">'
                    f'<div class="weekday-analysis-rank-fill {tone_class}" style="width: {fill_ratio * 100:.1f}%"></div>'
                    "</div>"
                    f'<div class="weekday-analysis-rank-value {tone_class}">{_format_average_orders(total_orders)}</div>'
                    "</div>"
                )
            )
        rows_html = (
            '<div class="weekday-analysis-rank-stack">'
            + "".join(row_html_parts)
            + "</div>"
        )

    return (
        '<div class="month-analysis-side-card">'
        f'<div class="month-analysis-card-title">{title}</div>'
        + rows_html
        + "</div>"
    )


def _create_hourly_total_orders_figure(
    hourly_orders: pd.DataFrame,
    selected_channel: str = "Overall",
) -> go.Figure:
    """Create the exact Q4 hourly average-orders line chart."""
    customdata = hourly_orders[
        [
            "Offline Orders",
            "Online Orders",
            "Total Orders",
            "Offline Average Orders",
            "Online Average Orders",
            "Total Average Orders",
        ]
    ].to_numpy()
    if selected_channel == "Offline":
        y_series = pd.to_numeric(
            hourly_orders["Offline Average Orders"], errors="coerce"
        ).fillna(0)
        hover_template = (
            "%{x}<br>"
            "Offline orders: %{customdata[0]:,.0f}<br>"
            "Average orders per day at that hour: %{customdata[3]:.1f}<extra></extra>"
        )
    elif selected_channel == "Online":
        y_series = pd.to_numeric(
            hourly_orders["Online Average Orders"], errors="coerce"
        ).fillna(0)
        hover_template = (
            "%{x}<br>"
            "Online orders: %{customdata[1]:,.0f}<br>"
            "Average orders per day at that hour: %{customdata[4]:.1f}<extra></extra>"
        )
    else:
        y_series = pd.to_numeric(
            hourly_orders["Total Average Orders"], errors="coerce"
        ).fillna(0)
        hover_template = (
            "%{x}<br>"
            "Offline orders: %{customdata[0]:,.0f}<br>"
            "Online orders: %{customdata[1]:,.0f}<br>"
            "Total orders: %{customdata[2]:,.0f}<br>"
            "Average orders per day at that hour: %{customdata[5]:.1f}<extra></extra>"
        )

    band_labels = _classify_hourly_peak_bands(y_series)
    band_palette = _get_hourly_peak_palette()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=hourly_orders["Hour Label"],
            y=y_series,
            mode="lines",
            line=dict(color=SAGE_MIST_SECONDARY, width=3, shape="spline", smoothing=0.45),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for band_name in ["High peak", "Medium peak", "Dead / low"]:
        band_mask = band_labels.eq(band_name)
        if not band_mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=hourly_orders.loc[band_mask, "Hour Label"],
                y=y_series.loc[band_mask],
                mode="markers",
                marker=dict(
                    size=10,
                    color=band_palette[band_name],
                    line=dict(color=SAGE_MIST_SURFACE_SOFT, width=1.6),
                ),
                customdata=customdata[band_mask.to_numpy()],
                hovertemplate=hover_template,
                showlegend=False,
            )
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=380,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(
            type="category",
            tickangle=0,
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickformat=".1f",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
        hoverlabel=dict(
            bgcolor=SAGE_MIST_SURFACE_SOFT,
            bordercolor=SAGE_MIST_BORDER,
            font=dict(color=SAGE_MIST_TEXT_STRONG),
        ),
    )
    return figure


def _create_hourly_estimated_sales_figure(
    hourly_orders: pd.DataFrame,
    selected_channel: str = "Overall",
) -> go.Figure:
    """
    Create the exact Q4 hourly sales chart.

    This chart uses exact Q4 hourly sales derived from:
    - POS table for offline sales
    - delivery_partner_raw_url raw table for online sales
    It does not use approximate AOV-based logic.
    """
    customdata = pd.DataFrame(
        {
            "offline_sales": pd.to_numeric(
                hourly_orders["Offline Sales"], errors="coerce"
            ).fillna(0),
            "online_sales": pd.to_numeric(
                hourly_orders["Online Sales"], errors="coerce"
            ).fillna(0),
            "total_sales": pd.to_numeric(
                hourly_orders["Total Sales"], errors="coerce"
            ).fillna(0),
            "average_sales": pd.to_numeric(
                hourly_orders["Total Average Sales"], errors="coerce"
            ).fillna(0),
            "offline_average_sales": pd.to_numeric(
                hourly_orders["Offline Average Sales"], errors="coerce"
            ).fillna(0),
            "online_average_sales": pd.to_numeric(
                hourly_orders["Online Average Sales"], errors="coerce"
            ).fillna(0),
        }
    ).to_numpy()
    if selected_channel == "Offline":
        y_series = pd.to_numeric(
            hourly_orders["Offline Average Sales"], errors="coerce"
        ).fillna(0)
        hover_template = (
            "%{x}<br>"
            "Offline sales: €%{customdata[0]:,.2f}<br>"
            "Average sales per day at that hour: €%{customdata[4]:,.2f}<extra></extra>"
        )
    elif selected_channel == "Online":
        y_series = pd.to_numeric(
            hourly_orders["Online Average Sales"], errors="coerce"
        ).fillna(0)
        hover_template = (
            "%{x}<br>"
            "Online sales: €%{customdata[1]:,.2f}<br>"
            "Average sales per day at that hour: €%{customdata[5]:,.2f}<extra></extra>"
        )
    else:
        y_series = pd.to_numeric(
            hourly_orders["Total Average Sales"], errors="coerce"
        ).fillna(0)
        hover_template = (
            "%{x}<br>"
            "Offline sales: €%{customdata[0]:,.2f}<br>"
            "Online sales: €%{customdata[1]:,.2f}<br>"
            "Total sales: €%{customdata[2]:,.2f}<br>"
            "Average sales per day at that hour: €%{customdata[3]:,.2f}<extra></extra>"
        )

    band_labels = _classify_hourly_peak_bands(y_series)
    band_palette = _get_hourly_peak_palette()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=hourly_orders["Hour Label"],
            y=y_series,
            mode="lines",
            line=dict(color=SAGE_MIST_PRIMARY, width=3, shape="spline", smoothing=0.45),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for band_name in ["High peak", "Medium peak", "Dead / low"]:
        band_mask = band_labels.eq(band_name)
        if not band_mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=hourly_orders.loc[band_mask, "Hour Label"],
                y=y_series.loc[band_mask],
                mode="markers",
                marker=dict(
                    size=10,
                    color=band_palette[band_name],
                    line=dict(color=SAGE_MIST_SURFACE_SOFT, width=1.6),
                ),
                customdata=customdata[band_mask.to_numpy()],
                hovertemplate=hover_template,
                showlegend=False,
            )
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=380,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(
            type="category",
            tickangle=0,
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickprefix="€",
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
        hoverlabel=dict(
            bgcolor=SAGE_MIST_SURFACE_SOFT,
            bordercolor=SAGE_MIST_BORDER,
            font=dict(color=SAGE_MIST_TEXT_STRONG),
        ),
    )
    return figure


def _format_shift_slot_label(hour_value: int) -> str:
    """Format one shift hour bucket label such as 12–1pm."""
    return _format_hour_bucket_label(hour_value)


def _prepare_shift_analysis_q4_data(
    cleaned_tables: dict[str, pd.DataFrame],
    raw_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """
    Build exact Jan-to-Mar 12 PM to 4 PM shift average net revenue data.

    Sources used on purpose:
    - POS table for offline in-house sales
    - delivery_partner_raw_url raw table for online sales

    Excluded on purpose:
    - Delivery financial summary tables
    - Any AOV-based or estimated hourly sales logic
    """
    q4_start = pd.Timestamp("2026-01-01")
    q4_end = pd.Timestamp("2026-03-31")
    included_hours = [12, 13, 14, 15]
    day_count_in_scope = float(len(pd.date_range(q4_start, q4_end, freq="D")))
    offline_net_multiplier = 0.87
    online_net_multiplier = 0.62

    pos_data = _get_table(cleaned_tables, "pos")
    if not pos_data.empty and "Time-Stamp" in pos_data.columns:
        pos_timestamp_series = pd.to_datetime(pos_data["Time-Stamp"], errors="coerce")
        pos_filtered = pos_data.loc[
            pos_timestamp_series.between(
                q4_start,
                q4_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            )
        ].copy()
    else:
        pos_filtered = _filter_by_date_columns(
            pos_data,
            start_date=q4_start,
            end_date=q4_end,
            date_columns=["Date"],
        )

    hourly_index = pd.Index(included_hours, name="Hour")
    offline_sales = pd.Series(0.0, index=hourly_index, dtype="float64")
    if not pos_filtered.empty and {"Hour", "Amount"}.issubset(pos_filtered.columns):
        offline_sales = (
            pos_filtered.assign(
                HourValue=pd.to_numeric(pos_filtered["Hour"], errors="coerce"),
                AmountValue=pd.to_numeric(pos_filtered["Amount"], errors="coerce").fillna(0),
            )
            .dropna(subset=["HourValue"])
            .loc[lambda dataframe: dataframe["HourValue"].isin(included_hours)]
            .groupby("HourValue")["AmountValue"]
            .sum()
            .reindex(hourly_index)
            .fillna(0)
        )

    delivery_partner_raw = raw_tables.get("delivery_partner_raw", pd.DataFrame()).copy()
    online_sales = pd.Series(0.0, index=hourly_index, dtype="float64")
    raw_rows_after_filter = 0
    if not delivery_partner_raw.empty:
        date_column = _find_column_name(delivery_partner_raw, ["Date", "Date "])
        time_column = _find_column_name(delivery_partner_raw, ["Time"])
        sale_column = _find_column_name(delivery_partner_raw, ["Sale"])

        if date_column is not None and time_column is not None and sale_column is not None:
            online_data = delivery_partner_raw.copy()
            online_datetime = pd.to_datetime(
                online_data[date_column].astype(str).str.strip()
                + " "
                + online_data[time_column].astype(str).str.strip(),
                errors="coerce",
            )
            online_data["Order Datetime Raw"] = online_datetime
            online_data["HourValue"] = online_datetime.dt.hour
            online_data["Sale Numeric"] = pd.to_numeric(
                online_data[sale_column].map(parse_number), errors="coerce"
            ).fillna(0)
            online_data = online_data.loc[
                online_data["Order Datetime Raw"].between(
                    q4_start,
                    q4_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                )
            ].copy()
            raw_rows_after_filter = int(len(online_data.index))

            if not online_data.empty:
                online_sales = (
                    online_data.dropna(subset=["HourValue"])
                    .loc[lambda dataframe: dataframe["HourValue"].isin(included_hours)]
                    .groupby("HourValue")["Sale Numeric"]
                    .sum()
                    .reindex(hourly_index)
                    .fillna(0)
                )

    shift_hourly = pd.DataFrame(
        {
            "Hour": hourly_index.astype(int),
            "Slot Label": [_format_shift_slot_label(hour_value) for hour_value in included_hours],
            "Offline Gross Revenue": pd.to_numeric(offline_sales, errors="coerce").fillna(0).values,
            "Online Gross Revenue": pd.to_numeric(online_sales, errors="coerce").fillna(0).values,
        }
    )
    shift_hourly["Total Gross Revenue"] = (
        shift_hourly["Offline Gross Revenue"] + shift_hourly["Online Gross Revenue"]
    )
    shift_hourly["Offline Average Gross Revenue"] = (
        shift_hourly["Offline Gross Revenue"] / day_count_in_scope if day_count_in_scope else 0.0
    )
    shift_hourly["Online Average Gross Revenue"] = (
        shift_hourly["Online Gross Revenue"] / day_count_in_scope if day_count_in_scope else 0.0
    )
    shift_hourly["Total Average Gross Revenue"] = (
        shift_hourly["Total Gross Revenue"] / day_count_in_scope if day_count_in_scope else 0.0
    )
    shift_hourly["Offline Net Revenue"] = (
        shift_hourly["Offline Gross Revenue"] * offline_net_multiplier
    )
    shift_hourly["Online Net Revenue"] = (
        shift_hourly["Online Gross Revenue"] * online_net_multiplier
    )
    shift_hourly["Total Net Revenue"] = (
        shift_hourly["Offline Net Revenue"] + shift_hourly["Online Net Revenue"]
    )
    shift_hourly["Offline Average Net Revenue"] = (
        shift_hourly["Offline Net Revenue"] / day_count_in_scope if day_count_in_scope else 0.0
    )
    shift_hourly["Online Average Net Revenue"] = (
        shift_hourly["Online Net Revenue"] / day_count_in_scope if day_count_in_scope else 0.0
    )
    shift_hourly["Total Average Net Revenue"] = (
        shift_hourly["Total Net Revenue"] / day_count_in_scope if day_count_in_scope else 0.0
    )

    result = {
        "scope_start": q4_start,
        "scope_end": q4_end,
        "included_hours": included_hours,
        "day_count_in_scope": day_count_in_scope,
        "hourly_revenue": shift_hourly,
        "offline_gross_revenue": float(shift_hourly["Offline Average Gross Revenue"].sum()),
        "online_gross_revenue": float(shift_hourly["Online Average Gross Revenue"].sum()),
        "offline_net_revenue": float(shift_hourly["Offline Average Net Revenue"].sum()),
        "online_net_revenue": float(shift_hourly["Online Average Net Revenue"].sum()),
        "total_net_revenue": float(shift_hourly["Total Average Net Revenue"].sum()),
    }

    print(
        "SHIFT_ANALYSIS_Q4_DEBUG",
        {
            "scope_start": str(q4_start.date()),
            "scope_end": str(q4_end.date()),
            "included_hours": included_hours,
            "day_count_in_scope": day_count_in_scope,
            "offline_source": "POS Time-Stamp + Amount",
            "online_source": "delivery_partner_raw_url raw table Date + Time + Sale",
            "uses_delivery_financial_summary": False,
            "offline_deduction_pct": 13.0,
            "online_deduction_pct": 38.0,
            "offline_rows_after_filter": int(len(pos_filtered.index)),
            "online_rows_after_filter": raw_rows_after_filter,
            "offline_average_shift_net_revenue": result["offline_net_revenue"],
            "online_average_shift_net_revenue": result["online_net_revenue"],
            "average_shift_net_revenue": result["total_net_revenue"],
        },
    )
    print(shift_hourly.to_string(index=False))
    return result


def _format_time_display(time_value: time) -> str:
    """Format a Python time object into a readable 12-hour label."""
    hour_value = int(time_value.hour)
    minute_value = int(time_value.minute)
    suffix = "AM" if hour_value < 12 else "PM"
    hour_12 = hour_value % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12}:{minute_value:02d} {suffix}"


def _calculate_shift_duration_hours(start_time: time, end_time: time) -> float:
    """Calculate one employee shift duration in decimal hours, clamped at zero."""
    start_minutes = (int(start_time.hour) * 60) + int(start_time.minute)
    end_minutes = (int(end_time.hour) * 60) + int(end_time.minute)
    if end_minutes <= start_minutes:
        return 0.0
    return float(end_minutes - start_minutes) / 60.0


def _format_duration_label(duration_hours: float) -> str:
    """Format one duration value cleanly for the staffing summary."""
    whole_hours = int(duration_hours)
    minute_value = int(round((float(duration_hours) - whole_hours) * 60))
    if minute_value == 60:
        whole_hours += 1
        minute_value = 0
    if minute_value == 0:
        return f"{whole_hours} hrs"
    return f"{whole_hours} hrs {minute_value} min"


def _build_shift_setup_summary_html(
    employee_rows: list[dict[str, object]],
    total_labour_hours: float,
    total_labour_cost: float,
    other_expenses: float,
    total_cost: float,
) -> str:
    """Render the employee-wise shift setup summary block."""
    rows = [
        (
            "Total employees",
            f"{len(employee_rows)} employees" if len(employee_rows) != 1 else "1 employee",
        ),
        ("Total labour hours", _format_duration_label(total_labour_hours)),
        ("Total labour cost", _format_currency(total_labour_cost)),
        ("Other expenses", _format_currency(other_expenses)),
        ("Total cost", _format_currency(total_cost)),
    ]
    rows_html = "".join(
        (
            '<div class="shift-summary-row">'
            f'<span class="shift-summary-label">{label}</span>'
            f'<span class="shift-summary-value">{value}</span>'
            "</div>"
        )
        for label, value in rows
    )
    employee_breakdown_html = "".join(
        (
            '<div class="shift-employee-breakdown-row">'
            f'<span class="shift-employee-breakdown-name">{str(employee_row["label"])}</span>'
            f'<span class="shift-employee-breakdown-time">{str(employee_row["start_label"])} – {str(employee_row["end_label"])}</span>'
            f'<span class="shift-employee-breakdown-hours">{str(employee_row["hours_label"])}</span>'
            f'<span class="shift-employee-breakdown-wage">{_format_currency(float(employee_row["hourly_wage"]))}/hr</span>'
            f'<span class="shift-employee-breakdown-cost">{_format_currency(float(employee_row["employee_cost"]))}</span>'
            "</div>"
        )
        for employee_row in employee_rows
    )
    return (
        '<div class="shift-summary-card">'
        '<div class="shift-breakdown-title">Per-employee breakdown</div>'
        '<div class="shift-employee-breakdown-head">'
        '<span>Name</span><span>Shift</span><span>Hours</span><span>Wage</span><span>Cost</span>'
        '</div>'
        f'{employee_breakdown_html}'
        '<div class="shift-summary-divider"></div>'
        f'{rows_html}'
        '</div>'
    )


def _build_shift_data_summary_html(
    shift_hourly: pd.DataFrame,
    offline_net_revenue: float,
    online_net_revenue: float,
    total_net_revenue: float,
) -> str:
    """Render the average Jan-to-Mar bucket net revenue breakdown."""
    hour_rows_html = "".join(
        (
            '<div class="shift-data-hour-row">'
            f'<span class="shift-data-hour-label">{row["Slot Label"]}</span>'
            f'<span class="shift-data-hour-value">{_format_currency(float(row["Total Average Net Revenue"]))}</span>'
            "</div>"
        )
        for _, row in shift_hourly.iterrows()
    )
    totals_html = "".join(
        [
            (
                '<div class="shift-summary-row">'
                '<span class="shift-summary-label">Offline avg net revenue</span>'
                f'<span class="shift-summary-value">{_format_currency(offline_net_revenue)}</span>'
                "</div>"
            ),
            (
                '<div class="shift-summary-row">'
                '<span class="shift-summary-label">Online avg net revenue</span>'
                f'<span class="shift-summary-value">{_format_currency(online_net_revenue)}</span>'
                "</div>"
            ),
            (
                '<div class="shift-summary-row total">'
                '<span class="shift-summary-label">Average shift net revenue</span>'
                f'<span class="shift-summary-value">{_format_currency(total_net_revenue)}</span>'
                "</div>"
            ),
        ]
    )
    return (
        '<div class="shift-data-card">'
        '<div class="shift-data-block-label">Average net sales per included hour · Jan to Mar</div>'
        f'<div class="shift-data-hour-stack">{hour_rows_html}</div>'
        '<div class="shift-summary-divider"></div>'
        f'{totals_html}'
        "</div>"
    )


def _build_shift_result_card_html(
    net_result: float,
    margin_pct: float,
) -> str:
    """Render the profit/loss result card for the shift."""
    if net_result > 0:
        tone_class = "profit"
        result_label = "Profit"
        helper_text = "This shift is generating positive returns."
    elif net_result < 0:
        tone_class = "loss"
        result_label = "Loss"
        helper_text = "This shift is operating at a loss."
    else:
        tone_class = "neutral"
        result_label = "Break-even"
        helper_text = "This shift is close to break-even."

    sign_prefix = "+" if net_result > 0 else ""
    return (
        f'<div class="shift-result-card {tone_class}">'
        '<div class="shift-result-title">Net Result</div>'
        '<div class="shift-result-layout">'
        '<div class="shift-result-main">'
        f'<div class="shift-result-value">{sign_prefix}{_format_currency(net_result)}</div>'
        f'<div class="shift-result-label">{result_label}</div>'
        f'<div class="shift-result-helper">{helper_text}</div>'
        "</div>"
        '<div class="shift-result-margin-block">'
        '<div class="shift-result-margin-label">Margin</div>'
        f'<div class="shift-result-margin-value">{margin_pct:.1f}%</div>'
        "</div>"
        "</div>"
        "</div>"
    )


def _create_shift_revenue_vs_cost_figure(
    offline_gross_revenue: float,
    offline_net_revenue: float,
    online_gross_revenue: float,
    online_net_revenue: float,
    total_net_revenue: float,
    total_labour_cost: float,
    other_expenses: float,
    total_cost: float,
) -> go.Figure:
    """Create the direct net-revenue vs cost comparison chart for the shift."""
    revenue_value = float(total_net_revenue)
    cost_value = float(total_cost)
    label_text = [
        _format_compact_currency_label(revenue_value) if abs(revenue_value) >= 1 else "",
        _format_compact_currency_label(cost_value) if abs(cost_value) >= 1 else "",
    ]
    label_y = [revenue_value * 0.5, cost_value * 0.5]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=["12–4pm Avg Net Revenue"],
            y=[revenue_value],
            marker=dict(color=SAGE_MIST_ACCENT),
            width=[0.36],
            customdata=[
                [
                    float(offline_gross_revenue),
                    float(offline_net_revenue),
                    float(online_gross_revenue),
                    float(online_net_revenue),
                    float(total_net_revenue),
                ]
            ],
            hovertemplate=(
                "%{x}<br>"
                "Offline gross revenue: €%{customdata[0]:,.2f}<br>"
                "Offline net revenue: €%{customdata[1]:,.2f}<br>"
                "Online gross revenue: €%{customdata[2]:,.2f}<br>"
                "Online net revenue: €%{customdata[3]:,.2f}<br>"
                "Total net revenue: €%{customdata[4]:,.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Bar(
            x=["12–4pm Cost"],
            y=[cost_value],
            marker=dict(color=SAGE_MIST_ALERT),
            width=[0.36],
            customdata=[[float(total_labour_cost), float(other_expenses), float(total_cost)]],
            hovertemplate=(
                "%{x}<br>"
                "Total labour cost: €%{customdata[0]:,.2f}<br>"
                "Other expenses: €%{customdata[1]:,.2f}<br>"
                "Total cost: €%{customdata[2]:,.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=["12–4pm Avg Net Revenue", "12–4pm Cost"],
            y=label_y,
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(
                color=SAGE_MIST_TEXT_STRONG,
                size=12,
                family="Arial Black, Arial, sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        xaxis=dict(
            tickangle=0,
            showgrid=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickprefix="€",
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
        hoverlabel=dict(
            bgcolor=SAGE_MIST_SURFACE_SOFT,
            bordercolor=SAGE_MIST_BORDER,
            font=dict(color=SAGE_MIST_TEXT_STRONG),
        ),
    )
    return figure


def _create_shift_hourly_revenue_figure(shift_hourly: pd.DataFrame) -> go.Figure:
    """Create the exact average hourly net-revenue chart for the 12 PM to 4 PM bucket."""
    customdata = pd.DataFrame(
        {
            "offline_gross_revenue": pd.to_numeric(
                shift_hourly["Offline Average Gross Revenue"], errors="coerce"
            ).fillna(0),
            "offline_net_revenue": pd.to_numeric(
                shift_hourly["Offline Average Net Revenue"], errors="coerce"
            ).fillna(0),
            "online_gross_revenue": pd.to_numeric(
                shift_hourly["Online Average Gross Revenue"], errors="coerce"
            ).fillna(0),
            "online_net_revenue": pd.to_numeric(
                shift_hourly["Online Average Net Revenue"], errors="coerce"
            ).fillna(0),
            "total_net_revenue": pd.to_numeric(
                shift_hourly["Total Average Net Revenue"], errors="coerce"
            ).fillna(0),
        }
    ).to_numpy()
    y_values = pd.to_numeric(
        shift_hourly["Total Average Net Revenue"], errors="coerce"
    ).fillna(0)
    label_text = [
        _format_compact_currency_label(value) if float(value) >= 1 else ""
        for value in y_values
    ]
    label_y = [float(value) * 0.5 for value in y_values]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=shift_hourly["Slot Label"],
            y=y_values,
            marker=dict(color=SAGE_MIST_SECONDARY),
            customdata=customdata,
            hovertemplate=(
                "%{x}<br>"
                "Offline gross revenue: €%{customdata[0]:,.2f}<br>"
                "Offline net revenue: €%{customdata[1]:,.2f}<br>"
                "Online gross revenue: €%{customdata[2]:,.2f}<br>"
                "Online net revenue: €%{customdata[3]:,.2f}<br>"
                "Total net revenue: €%{customdata[4]:,.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=shift_hourly["Slot Label"],
            y=label_y,
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(
                color=SAGE_MIST_TEXT_STRONG,
                size=11,
                family="Arial Black, Arial, sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        xaxis=dict(
            tickangle=0,
            showgrid=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickprefix="€",
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
        hoverlabel=dict(
            bgcolor=SAGE_MIST_SURFACE_SOFT,
            bordercolor=SAGE_MIST_BORDER,
            font=dict(color=SAGE_MIST_TEXT_STRONG),
        ),
    )
    return figure


def _prepare_scenario_impact_q4_data(
    cleaned_tables: dict[str, pd.DataFrame],
    raw_tables: dict[str, pd.DataFrame],
) -> dict[str, float | str]:
    """
    Build the Jan-to-Mar baseline used by the scenario simulator.

    This section is a scenario simulator based on the Jan-to-Mar baseline.
    The selected percentage is applied uniformly to baseline sales and orders.
    Offline and online splits come from POS + delivery raw data only.
    """
    q4_start = pd.Timestamp("2026-01-01")
    q4_end = pd.Timestamp("2026-03-31")

    pos_data = _get_table(cleaned_tables, "pos")
    if not pos_data.empty and "Time-Stamp" in pos_data.columns:
        pos_timestamp_series = pd.to_datetime(pos_data["Time-Stamp"], errors="coerce")
        pos_filtered = pos_data.loc[
            pos_timestamp_series.between(
                q4_start,
                q4_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            )
        ].copy()
    else:
        pos_filtered = _filter_by_date_columns(
            pos_data,
            start_date=q4_start,
            end_date=q4_end,
            date_columns=["Date"],
        )

    offline_gross_sale = _safe_sum(pos_filtered, "Amount")
    offline_orders = float(_count_valid_offline_orders(pos_filtered))

    delivery_partner_raw = raw_tables.get("delivery_partner_raw", pd.DataFrame()).copy()
    online_gross_sale = 0.0
    online_orders = 0.0
    online_rows_after_filter = 0
    if not delivery_partner_raw.empty:
        date_column = _find_column_name(delivery_partner_raw, ["Date", "Date "])
        time_column = _find_column_name(delivery_partner_raw, ["Time"])
        sale_column = _find_column_name(delivery_partner_raw, ["Sale"])
        if date_column is not None and time_column is not None:
            online_data = delivery_partner_raw.copy()
            online_datetime = pd.to_datetime(
                online_data[date_column].astype(str).str.strip()
                + " "
                + online_data[time_column].astype(str).str.strip(),
                errors="coerce",
            )
            online_data["Order Datetime Raw"] = online_datetime
            online_data = online_data.loc[
                online_data["Order Datetime Raw"].between(
                    q4_start,
                    q4_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                )
            ].copy()
            online_rows_after_filter = int(len(online_data.index))
            online_orders = float(online_rows_after_filter)
            if sale_column is not None and not online_data.empty:
                online_gross_sale = float(
                    pd.to_numeric(
                        online_data[sale_column].map(parse_number),
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )

    total_gross_sale = offline_gross_sale + online_gross_sale
    total_orders = offline_orders + online_orders

    result = {
        "scope_start": str(q4_start.date()),
        "scope_end": str(q4_end.date()),
        "offline_gross_sale": float(offline_gross_sale),
        "online_gross_sale": float(online_gross_sale),
        "total_gross_sale": float(total_gross_sale),
        "offline_orders": float(offline_orders),
        "online_orders": float(online_orders),
        "total_orders": float(total_orders),
    }
    print(
        "SCENARIO_IMPACT_BASELINE_DEBUG",
        {
            **result,
            "offline_source": "POS Time-Stamp + Amount + distinct positive Bill-nr.",
            "online_source": "delivery_partner_raw_url raw table Date + Time + Sale",
            "uses_delivery_financial_summary": False,
            "offline_rows_after_filter": int(len(pos_filtered.index)),
            "online_rows_after_filter": online_rows_after_filter,
        },
    )
    return result


def _build_scenario_kpi_card_html(
    title: str,
    value_text: str,
    support_text: str,
    tone_class: str = "neutral",
) -> str:
    """Render one scenario KPI card."""
    return f"""
    <div class="scenario-kpi-card {tone_class}">
        <div class="scenario-kpi-title">{title}</div>
        <div class="scenario-kpi-value">{value_text}</div>
        <div class="scenario-kpi-support">{support_text}</div>
    </div>
    """


def _build_scenario_split_card_html(
    title: str,
    baseline_text: str,
    projected_text: str,
    change_text: str,
    tone_class: str,
) -> str:
    """Render one compact scenario split-impact card."""
    return f"""
    <div class="scenario-split-card">
        <div class="scenario-split-title">{title}</div>
        <div class="scenario-split-row">
            <span class="scenario-split-label">Baseline</span>
            <span class="scenario-split-value">{baseline_text}</span>
        </div>
        <div class="scenario-split-row">
            <span class="scenario-split-label projected">Projected</span>
            <span class="scenario-split-value">{projected_text}</span>
        </div>
        <div class="scenario-split-footer {tone_class}">{change_text}</div>
    </div>
    """


def _build_scenario_summary_html(
    scenario_pct: int,
    gross_sale_change: float,
    orders_change: float,
    offline_sale_change: float,
    online_sale_change: float,
    offline_order_change: float,
    online_order_change: float,
) -> str:
    """Render the short interpretation box for the scenario analysis."""
    direction_label = "growth" if int(scenario_pct) > 0 else "decline" if int(scenario_pct) < 0 else "flat scenario"
    pct_text = f"{abs(int(scenario_pct))}%"
    if int(scenario_pct) == 0:
        first_line = (
            "A 0% scenario leaves Jan–Mar baseline gross sale and orders unchanged."
        )
    else:
        first_line = (
            f"A {pct_text} {direction_label} from the Jan–Mar baseline would change total gross sale by "
            f"{_format_signed_currency(gross_sale_change)} and total orders by "
            f"{_format_signed_whole_number(orders_change)}."
        )

    second_line = (
        "Offline sales would change by "
        f"{_format_signed_currency(offline_sale_change)} and online sales by "
        f"{_format_signed_currency(online_sale_change)}."
    )
    third_line = (
        "Offline orders would change by "
        f"{_format_signed_whole_number(offline_order_change)} and online orders by "
        f"{_format_signed_whole_number(online_order_change)}."
    )
    return f"""
    <div class="scenario-summary-box">
        <div class="scenario-summary-line">{first_line}</div>
        <div class="scenario-summary-line">{second_line}</div>
        <div class="scenario-summary-line">{third_line}</div>
    </div>
    """


def _create_scenario_comparison_figure(
    baseline_value: float,
    projected_value: float,
    scenario_pct: int,
    is_currency: bool,
) -> go.Figure:
    """Create one baseline-vs-projected comparison bar chart."""
    projected_label = f"Projected ({int(scenario_pct):+d}%)" if int(scenario_pct) else "Projected (0%)"
    bar_labels = ["Baseline", projected_label]
    values = [float(baseline_value), float(projected_value)]
    projected_color = SAGE_MIST_ALERT if int(scenario_pct) < 0 else SAGE_MIST_ACCENT
    marker_colors = [SAGE_MIST_ONLINE, projected_color]
    if is_currency:
        label_text = [_format_compact_currency_label(value) for value in values]
        hover_template = "%{x}<br>Amount: €%{y:,.2f}<extra></extra>"
    else:
        label_text = [_format_whole_number(int(round(value))) for value in values]
        hover_template = "%{x}<br>Orders: %{y:,.0f}<extra></extra>"

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=bar_labels,
            y=values,
            marker=dict(color=marker_colors),
            width=[0.42, 0.42],
            hovertemplate=hover_template,
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bar_labels,
            y=[value * 0.5 for value in values],
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(
                color=SAGE_MIST_TEXT_STRONG,
                size=12,
                family="Arial Black, Arial, sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_annotation(
        x=0.5,
        y=max(values) * 0.78 if max(values) else 0.0,
        text=_format_signed_percentage(float(scenario_pct)),
        showarrow=False,
        font=dict(
            color=projected_color,
            size=16,
            family="Arial Black, Arial, sans-serif",
        ),
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=320,
        showlegend=False,
        xaxis=dict(
            tickangle=0,
            showgrid=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickprefix="€" if is_currency else "",
            tickformat="~s" if is_currency else ",.0f",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
        hoverlabel=dict(
            bgcolor=SAGE_MIST_SURFACE_SOFT,
            bordercolor=SAGE_MIST_BORDER,
            font=dict(color=SAGE_MIST_TEXT_STRONG),
        ),
    )
    return figure


def _calculate_hourly_kpis(hourly_orders: pd.DataFrame) -> dict[str, object]:
    """Calculate the KPI row for Hourly Analysis."""
    if hourly_orders.empty:
        return {
            "peak_hour": "N/A",
            "peak_orders": 0.0,
            "offline_peak": "N/A",
            "offline_peak_orders": 0.0,
            "online_peak": "N/A",
            "online_peak_orders": 0.0,
            "quietest_hour": "N/A",
            "quietest_orders": 0.0,
        }

    peak_total_row = hourly_orders.sort_values("Total Average Orders", ascending=False).iloc[0]
    quietest_row = hourly_orders.sort_values("Total Average Orders", ascending=True).iloc[0]

    offline_active = hourly_orders.loc[hourly_orders["Offline Average Orders"] > 0].copy()
    online_active = hourly_orders.loc[hourly_orders["Online Average Orders"] > 0].copy()

    if offline_active.empty:
        offline_peak_label = "N/A"
        offline_peak_orders = 0.0
    else:
        offline_peak_row = offline_active.sort_values("Offline Average Orders", ascending=False).iloc[0]
        offline_peak_label = str(offline_peak_row["Hour Label"])
        offline_peak_orders = float(offline_peak_row["Offline Average Orders"])

    if online_active.empty:
        online_peak_label = "N/A"
        online_peak_orders = 0.0
    else:
        online_peak_row = online_active.sort_values("Online Average Orders", ascending=False).iloc[0]
        online_peak_label = str(online_peak_row["Hour Label"])
        online_peak_orders = float(online_peak_row["Online Average Orders"])

    return {
        "peak_hour": str(peak_total_row["Hour Label"]),
        "peak_orders": float(peak_total_row["Total Average Orders"]),
        "offline_peak": offline_peak_label,
        "offline_peak_orders": offline_peak_orders,
        "online_peak": online_peak_label,
        "online_peak_orders": online_peak_orders,
        "quietest_hour": str(quietest_row["Hour Label"]),
        "quietest_orders": float(quietest_row["Total Average Orders"]),
    }


def _build_operational_notes_html(notes: list[str]) -> str:
    """Render short operational notes as a compact card block."""
    if not notes:
        notes = ["Hourly operational notes will appear when hourly order patterns are available."]

    lines = "".join(
        f'<div class="operational-note-line">{note}</div>'
        for note in notes
    )
    return (
        '<div class="operational-notes-card">'
        '<div class="operational-notes-title">Operational Notes</div>'
        f'<div class="operational-notes-body">{lines}</div>'
        '</div>'
    )


def _build_hourly_operational_notes(hourly_orders: pd.DataFrame) -> list[str]:
    """Generate short business-friendly notes from the hourly pattern."""
    if hourly_orders.empty:
        return ["Hourly operational notes will appear when hourly order patterns are available."]

    peak_total_row = hourly_orders.sort_values("Total Average Orders", ascending=False).iloc[0]
    quietest_row = hourly_orders.sort_values("Total Average Orders", ascending=True).iloc[0]

    offline_active = hourly_orders.loc[hourly_orders["Offline Average Orders"] > 0].copy()
    online_active = hourly_orders.loc[hourly_orders["Online Average Orders"] > 0].copy()

    offline_peak_row = (
        offline_active.sort_values("Offline Average Orders", ascending=False).iloc[0]
        if not offline_active.empty
        else None
    )
    online_peak_row = (
        online_active.sort_values("Online Average Orders", ascending=False).iloc[0]
        if not online_active.empty
        else None
    )

    peak_hour_label = str(peak_total_row["Hour Label"])
    quietest_hour_label = str(quietest_row["Hour Label"])
    peak_orders = float(peak_total_row["Total Average Orders"])
    quietest_orders = float(quietest_row["Total Average Orders"])

    notes = [
        f"{peak_hour_label} carries the highest average load at {_format_average_orders(peak_orders)} orders.",
    ]

    if offline_peak_row is not None and online_peak_row is not None:
        offline_peak_label = str(offline_peak_row["Hour Label"])
        online_peak_label = str(online_peak_row["Hour Label"])
        if offline_peak_label == online_peak_label:
            notes.append(
                f"Offline and online demand peak together at {offline_peak_label}, so channel peaks align."
            )
        else:
            notes.append(
                f"Offline peaks at {offline_peak_label} while online peaks at {online_peak_label}, so demand is staggered."
            )
    elif offline_peak_row is not None:
        notes.append(
            f"Offline demand is concentrated around {str(offline_peak_row['Hour Label'])}."
        )
    elif online_peak_row is not None:
        notes.append(
            f"Online demand is concentrated around {str(online_peak_row['Hour Label'])}."
        )

    peak_hour_numeric = int(peak_total_row["Hour"])
    lead_in_label = _format_hour_label(max(0, peak_hour_numeric - 1))
    notes.append(
        f"Staffing and prep should tighten from {lead_in_label} into {peak_hour_label} to protect service flow."
    )
    notes.append(
        f"{quietest_hour_label} is the weakest active window at {_format_average_orders(quietest_orders)} orders, making it a better slot for offers or lighter staffing."
    )
    return notes[:4]


def _normalize_payment_type(value: object) -> str:
    """Normalize payment labels to make cash/digital matching safer."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(character for character in text if not unicodedata.combining(character))


def _is_cash_payment_type(value: object) -> bool:
    """Return True when a POS payment label clearly indicates cash."""
    normalized = _normalize_payment_type(value)
    if not normalized:
        return False

    cash_exact_values = {
        "cash",
        "bar",
        "barzahlung",
        "bargeld",
    }
    cash_keywords = (
        "cash",
        "barzahlung",
        "bargeld",
    )

    return normalized in cash_exact_values or any(
        keyword in normalized for keyword in cash_keywords
    )


def _is_card_payment_type(value: object) -> bool:
    """Return True when a POS payment label indicates a card or digital POS payment."""
    normalized = _normalize_payment_type(value)
    if not normalized or _is_cash_payment_type(normalized):
        return False

    card_keywords = (
        "card",
        "ec",
        "kredit",
        "credit",
        "debit",
        "visa",
        "master",
        "amex",
        "maestro",
        "electronic",
        "digital",
        "kontaktlos",
        "contactless",
        "apple pay",
        "google pay",
        "tap",
    )

    return any(keyword in normalized for keyword in card_keywords)


def _prepare_payment_method_breakdown_data(
    cleaned_tables: dict[str, pd.DataFrame],
    selected_quarter: str,
    selected_channel: str,
) -> dict[str, float]:
    """
    Build the combined gross-sale payment split.

    Business rule:
    - Cash = offline cash only
    - Card Payment = offline POS card only
    - Bank Deposit = all online gross sales
    """
    pos_data = _get_table(cleaned_tables, "pos")
    delivery_financials_data = _get_table(cleaned_tables, "delivery_financials")
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)

    pos_data_filtered = _filter_by_date_range(
        pos_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
    )
    delivery_financials_filtered = _filter_by_date_columns(
        delivery_financials_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Date", "Month Start"],
    )

    if selected_channel == "Offline":
        delivery_financials_filtered = delivery_financials_filtered.iloc[0:0].copy()
    elif selected_channel == "Online":
        pos_data_filtered = pos_data_filtered.iloc[0:0].copy()

    online_gross = _safe_sum(delivery_financials_filtered, "Gross")

    offline_cash = 0.0
    offline_card_payment = 0.0
    if (
        not pos_data_filtered.empty
        and "Payment type" in pos_data_filtered.columns
        and "Amount" in pos_data_filtered.columns
    ):
        normalized_payment_types = pos_data_filtered["Payment type"].map(
            _normalize_payment_type
        )
        cash_mask = normalized_payment_types.map(_is_cash_payment_type)
        card_mask = normalized_payment_types.map(_is_card_payment_type)
        remaining_non_cash_mask = (
            normalized_payment_types.ne("")
            & ~cash_mask
            & ~card_mask
        )

        offline_cash = float(
            pd.to_numeric(
                pos_data_filtered.loc[cash_mask, "Amount"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        offline_card_payment = float(
            pd.to_numeric(
                pos_data_filtered.loc[card_mask | remaining_non_cash_mask, "Amount"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    cash_value = offline_cash
    card_payment_value = offline_card_payment
    bank_deposit_value = online_gross
    total_value = cash_value + card_payment_value + bank_deposit_value

    if total_value > 0:
        cash_share = cash_value / total_value
        card_payment_share = card_payment_value / total_value
        bank_deposit_share = bank_deposit_value / total_value
    else:
        cash_share = 0.0
        card_payment_share = 0.0
        bank_deposit_share = 0.0

    return {
        "cash_value": float(cash_value),
        "card_payment_value": float(card_payment_value),
        "bank_deposit_value": float(bank_deposit_value),
        "total_value": float(total_value),
        "cash_share": float(cash_share),
        "card_payment_share": float(card_payment_share),
        "bank_deposit_share": float(bank_deposit_share),
    }


def _build_payment_method_summary_card_html(
    label: str,
    value_text: str,
    share_text: str,
    fill_ratio: float,
    tone_class: str,
) -> str:
    """Render one payment-method summary card."""
    fill_width = _clamp_ratio(fill_ratio) * 100
    return f"""
    <div class="payment-summary-card {tone_class}">
        <div class="payment-summary-head">
            <span class="payment-summary-dot {tone_class}"></span>
            <span class="payment-summary-label">{label}</span>
        </div>
        <div class="payment-summary-value">{value_text}</div>
        <div class="payment-summary-share {tone_class}">{share_text}</div>
        <div class="payment-summary-track">
            <div class="payment-summary-fill {tone_class}" style="width: {fill_width:.1f}%;"></div>
        </div>
    </div>
    """


def _create_payment_method_breakdown_figure(
    payment_breakdown: dict[str, float],
) -> go.Figure:
    """Create the cash vs card payment vs delivery partner donut chart."""
    cash_value = max(float(payment_breakdown["cash_value"]), 0.0)
    card_payment_value = max(float(payment_breakdown["card_payment_value"]), 0.0)
    bank_deposit_value = max(float(payment_breakdown["bank_deposit_value"]), 0.0)
    total_value = float(payment_breakdown["total_value"])

    figure = go.Figure()

    if cash_value > 0 or card_payment_value > 0 or bank_deposit_value > 0:
        figure.add_trace(
            go.Pie(
                labels=["Cash", "Card Payment", "Delivery Partner"],
                values=[cash_value, card_payment_value, bank_deposit_value],
                hole=0.72,
                domain=dict(x=[0.08, 0.92], y=[0.08, 0.92]),
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[
                        SAGE_MIST_ACCENT,
                        SAGE_MIST_ONLINE,
                        SAGE_MIST_SECONDARY,
                    ],
                    line=dict(color=SAGE_MIST_SURFACE, width=6),
                ),
                textinfo="none",
                hovertemplate="%{label}<br>Gross sale: €%{value:,.2f}<br>Share: %{percent}<extra></extra>",
                showlegend=False,
            )
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        annotations=[
            dict(
                text=f"<span style='color:{SAGE_MIST_TEXT_STRONG};font-size:28px;font-weight:700'>{_format_currency(total_value)}</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
                xanchor="center",
                yanchor="middle",
            )
        ],
        font=dict(color=SAGE_MIST_TEXT_STRONG),
    )
    return figure


def _create_monthly_total_gross_sale_figure(monthly_sales: pd.DataFrame) -> go.Figure:
    """Create the monthly total gross sale bar chart."""
    label_text = [
        f"{(value / 1000):.1f}k" if pd.notna(value) else ""
        for value in monthly_sales["Gross Sale"]
    ]
    label_y = [
        float(value) * 0.5 if pd.notna(value) else 0
        for value in monthly_sales["Gross Sale"]
    ]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=monthly_sales["Month Short"],
            y=monthly_sales["Gross Sale"],
            marker=dict(color=SAGE_MIST_PRIMARY),
            hovertemplate="%{x}<br>Total gross sale: €%{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=monthly_sales["Month Short"],
            y=label_y,
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(
                color=SAGE_MIST_TEXT_STRONG,
                size=11,
                family="Arial Black, Arial, sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        bargap=0.28,
        xaxis=dict(
            type="category",
            tickangle=0,
            showgrid=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickprefix="€",
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
    )
    return figure


def _create_online_vs_offline_trend_figure(monthly_sales: pd.DataFrame) -> go.Figure:
    """Create the monthly online versus offline gross sales line chart."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=monthly_sales["Month Short"],
            y=monthly_sales["Offline Gross Sale"],
            mode="lines+markers",
            name="Offline",
            line=dict(color=SAGE_MIST_ACCENT, width=3, shape="spline", smoothing=0.55),
            marker=dict(size=8, color=SAGE_MIST_ACCENT),
            hovertemplate="%{x}<br>Offline gross sale: €%{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=monthly_sales["Month Short"],
            y=monthly_sales["Online Gross Sale"],
            mode="lines+markers",
            name="Online",
            line=dict(color=SAGE_MIST_ONLINE, width=3, shape="spline", smoothing=0.55),
            marker=dict(size=8, color=SAGE_MIST_ONLINE),
            hovertemplate="%{x}<br>Online gross sale: €%{y:,.2f}<extra></extra>",
        )
    )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        xaxis=dict(
            type="category",
            tickangle=0,
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickprefix="€",
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
    )
    return figure


def _create_monthly_total_orders_figure(monthly_orders: pd.DataFrame) -> go.Figure:
    """Create the monthly total orders bar chart."""
    label_text = [
        _format_compact_count(value) if pd.notna(value) else ""
        for value in monthly_orders["Total Orders"]
    ]
    label_y = [
        float(value) * 0.5 if pd.notna(value) else 0
        for value in monthly_orders["Total Orders"]
    ]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=monthly_orders["Month Short"],
            y=monthly_orders["Total Orders"],
            marker=dict(color=SAGE_MIST_PRIMARY),
            hovertemplate="%{x}<br>Total orders: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=monthly_orders["Month Short"],
            y=label_y,
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(
                color=SAGE_MIST_TEXT_STRONG,
                size=11,
                family="Arial Black, Arial, sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        bargap=0.28,
        xaxis=dict(
            type="category",
            tickangle=0,
            showgrid=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
    )
    return figure


def _create_online_vs_offline_orders_figure(monthly_orders: pd.DataFrame) -> go.Figure:
    """Create the monthly online versus offline orders line chart."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=monthly_orders["Month Short"],
            y=monthly_orders["Offline Orders"],
            mode="lines+markers",
            name="Offline",
            line=dict(color=SAGE_MIST_ACCENT, width=3, shape="spline", smoothing=0.55),
            marker=dict(size=8, color=SAGE_MIST_ACCENT),
            hovertemplate="%{x}<br>Offline orders: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=monthly_orders["Month Short"],
            y=monthly_orders["Online Orders"],
            mode="lines+markers",
            name="Online",
            line=dict(color=SAGE_MIST_ONLINE, width=3, shape="spline", smoothing=0.55),
            marker=dict(size=8, color=SAGE_MIST_ONLINE),
            hovertemplate="%{x}<br>Online orders: %{y:,.0f}<extra></extra>",
        )
    )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        xaxis=dict(
            type="category",
            tickangle=0,
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
    )
    return figure


def _create_monthly_order_volume_stacked_figure(monthly_orders: pd.DataFrame) -> go.Figure:
    """Create the stacked monthly order volume chart for Deep Insights."""
    label_text = [
        _format_compact_count(value) if pd.notna(value) else ""
        for value in monthly_orders["Total Orders"]
    ]
    label_y = [
        float(value) * 0.5 if pd.notna(value) else 0
        for value in monthly_orders["Total Orders"]
    ]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=monthly_orders["Month Short"],
            y=monthly_orders["Offline Orders"],
            name="Offline",
            marker=dict(color=SAGE_MIST_ACCENT),
            hovertemplate="%{x}<br>Offline orders: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=monthly_orders["Month Short"],
            y=monthly_orders["Online Orders"],
            name="Online",
            marker=dict(color=SAGE_MIST_ONLINE),
            hovertemplate="%{x}<br>Online orders: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=monthly_orders["Month Short"],
            y=label_y,
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(
                color=SAGE_MIST_TEXT_STRONG,
                size=11,
                family="Arial Black, Arial, sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=12, b=10),
        height=420,
        showlegend=False,
        barmode="stack",
        bargap=0.24,
        hovermode="closest",
        xaxis=dict(
            type="category",
            tickangle=0,
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickformat="~s",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
        hoverlabel=dict(
            bgcolor=SAGE_MIST_SURFACE_SOFT,
            bordercolor=SAGE_MIST_BORDER,
            font=dict(color=SAGE_MIST_TEXT_STRONG),
        ),
    )
    return figure


def _create_weekday_order_volume_figure(weekday_orders: pd.DataFrame) -> go.Figure:
    """Create the stacked weekday average-order chart for Deep Insights."""
    offline_values = pd.to_numeric(
        weekday_orders["Offline Average Orders"], errors="coerce"
    ).fillna(0)
    online_values = pd.to_numeric(
        weekday_orders["Online Average Orders"], errors="coerce"
    ).fillna(0)
    total_values = offline_values + online_values
    total_labels = [
        _format_average_orders(value) if float(value) >= 6 else ""
        for value in total_values
    ]
    total_centers = [float(value) * 0.5 for value in total_values]
    customdata = pd.DataFrame(
        {
            "offline": offline_values,
            "online": online_values,
            "total": total_values,
        }
    ).to_numpy()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=weekday_orders["Weekday"],
            x=offline_values,
            name="Offline",
            orientation="h",
            marker=dict(color=SAGE_MIST_ACCENT),
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>"
                "Offline avg orders: %{customdata[0]:.1f}<br>"
                "Online avg orders: %{customdata[1]:.1f}<br>"
                "Total avg orders: %{customdata[2]:.1f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            y=weekday_orders["Weekday"],
            x=online_values,
            name="Online",
            orientation="h",
            marker=dict(color=SAGE_MIST_ONLINE),
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>"
                "Offline avg orders: %{customdata[0]:.1f}<br>"
                "Online avg orders: %{customdata[1]:.1f}<br>"
                "Total avg orders: %{customdata[2]:.1f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=total_centers,
            y=weekday_orders["Weekday"],
            mode="text",
            text=total_labels,
            textposition="middle center",
            textfont=dict(
                color=SAGE_MIST_TEXT_STRONG,
                size=11,
                family="Arial Black, Arial, sans-serif",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=12, b=10),
        height=420,
        showlegend=False,
        barmode="stack",
        bargap=0.26,
        hovermode="closest",
        xaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=False,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
            tickformat=".0f",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(weekday_orders["Weekday"]),
            autorange="reversed",
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            showspikes=False,
            tickfont=dict(color=SAGE_MIST_TEXT_STRONG),
            linecolor=SAGE_MIST_BORDER,
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
        hoverlabel=dict(
            bgcolor=SAGE_MIST_SURFACE_SOFT,
            bordercolor=SAGE_MIST_BORDER,
            font=dict(color=SAGE_MIST_TEXT_STRONG),
        ),
    )
    return figure


def _prepare_total_sales_growth_data(
    monthly_sales: pd.DataFrame,
    sales_column: str,
) -> pd.DataFrame:
    """Prepare valid month-on-month growth points from the selected monthly sales series."""
    if monthly_sales.empty:
        return pd.DataFrame(
            columns=[
                "Month Start",
                "Month Short",
                "Month Label",
                "Sales Value",
                "Previous Sales Value",
                "MoM Growth %",
                "Month Summary",
            ]
        )

    required_columns = {"Month Start", "Month Short", "Month Label", sales_column}
    if not required_columns.issubset(monthly_sales.columns):
        return pd.DataFrame(
            columns=[
                "Month Start",
                "Month Short",
                "Month Label",
                "Sales Value",
                "Previous Sales Value",
                "MoM Growth %",
                "Month Summary",
            ]
        )

    growth_data = monthly_sales[
        ["Month Start", "Month Short", "Month Label", sales_column]
    ].copy()
    growth_data = growth_data.rename(columns={sales_column: "Sales Value"})
    growth_data["Month Start"] = pd.to_datetime(growth_data["Month Start"], errors="coerce")
    growth_data = growth_data.dropna(subset=["Month Start"]).sort_values("Month Start")
    growth_data["Sales Value"] = pd.to_numeric(
        growth_data["Sales Value"], errors="coerce"
    ).fillna(0)
    growth_data["Previous Sales Value"] = growth_data["Sales Value"].shift(1)
    valid_mask = (
        growth_data["Previous Sales Value"].notna()
        & (growth_data["Previous Sales Value"] != 0)
    )
    growth_data["MoM Growth %"] = (
        (growth_data["Sales Value"] - growth_data["Previous Sales Value"])
        / growth_data["Previous Sales Value"]
    ) * 100
    growth_data = growth_data.loc[valid_mask].copy()

    if growth_data.empty:
        return pd.DataFrame(
            columns=[
                "Month Start",
                "Month Short",
                "Month Label",
                "Sales Value",
                "Previous Sales Value",
                "MoM Growth %",
                "Month Summary",
            ]
        )

    growth_data["Month Summary"] = pd.to_datetime(
        growth_data["Month Start"], errors="coerce"
    ).dt.strftime("%b %y")
    return growth_data.reset_index(drop=True)


def _log_total_sales_growth_debug(monthly_sales: pd.DataFrame) -> None:
    """Print the combined monthly gross series and MoM growth to the terminal log."""
    if monthly_sales.empty:
        print("[Total sales growth debug] No monthly combined gross sales available.")
        return

    debug_table = monthly_sales[
        [
            "Month Label",
            "Offline Gross Sale",
            "Online Gross Sale",
            "Gross Sale",
        ]
    ].copy()
    debug_table["Previous Gross Sale"] = debug_table["Gross Sale"].shift(1)
    valid_mask = (
        debug_table["Previous Gross Sale"].notna()
        & (debug_table["Previous Gross Sale"] != 0)
    )
    debug_table["MoM Growth %"] = pd.NA
    debug_table.loc[valid_mask, "MoM Growth %"] = (
        (
            debug_table.loc[valid_mask, "Gross Sale"]
            - debug_table.loc[valid_mask, "Previous Gross Sale"]
        )
        / debug_table.loc[valid_mask, "Previous Gross Sale"]
    ) * 100
    debug_table = debug_table.rename(columns={"Month Label": "Month"})
    print("[Total sales growth debug] Combined gross monthly series")
    print(debug_table.to_string(index=False))


def _format_signed_percentage(value: float) -> str:
    """Format a signed percentage with one decimal place."""
    if value > 0:
        return f"+{value:.1f}%"
    return f"{value:.1f}%"


def _format_signed_currency(value: float) -> str:
    """Format one signed euro amount cleanly."""
    numeric_value = float(value)
    if numeric_value > 0:
        return f"+€{numeric_value:,.2f}"
    if numeric_value < 0:
        return f"-€{abs(numeric_value):,.2f}"
    return "€0.00"


def _format_signed_whole_number(value: float) -> str:
    """Format one signed whole-number value cleanly."""
    rounded_value = int(round(float(value)))
    if rounded_value > 0:
        return f"+{rounded_value:,}"
    return f"{rounded_value:,}"


def _build_growth_summary_card_html(
    title: str,
    value_text: str,
    subtext: str,
    value_class: str,
) -> str:
    """Render one compact summary card for the growth section."""
    return f"""
    <div class="growth-summary-card">
        <div class="growth-summary-title">{title}</div>
        <div class="growth-summary-value {value_class}">{value_text}</div>
        <div class="growth-summary-subtext">{subtext}</div>
    </div>
    """


def _build_total_sales_growth_summary(
    growth_data: pd.DataFrame,
) -> dict[str, str]:
    """Build the summary-card values for the total sales growth section."""
    if growth_data.empty:
        return {
            "best_value": "0.0%",
            "best_month": "No data",
            "worst_value": "0.0%",
            "worst_month": "No data",
            "positive_value": "0 of 0",
            "positive_subtext": "0% hit rate",
        }

    best_row = growth_data.loc[growth_data["MoM Growth %"].idxmax()]
    worst_row = growth_data.loc[growth_data["MoM Growth %"].idxmin()]
    positive_months = int((growth_data["MoM Growth %"] > 0).sum())
    total_months = int(len(growth_data.index))
    hit_rate = (positive_months / total_months * 100) if total_months else 0.0

    return {
        "best_value": _format_signed_percentage(float(best_row["MoM Growth %"])),
        "best_month": str(best_row["Month Summary"]),
        "worst_value": _format_signed_percentage(float(worst_row["MoM Growth %"])),
        "worst_month": str(worst_row["Month Summary"]),
        "positive_value": f"{positive_months} of {total_months}",
        "positive_subtext": f"{hit_rate:.0f}% hit rate",
    }


def _build_total_sales_growth_legend_html() -> str:
    """Render the chart legend for the total sales growth section."""
    return """
    <div class="growth-chart-legend">
        <span class="growth-chart-legend-item">
            <span class="growth-chart-dot positive"></span>Positive
        </span>
        <span class="growth-chart-legend-item">
            <span class="growth-chart-dot negative"></span>Negative
        </span>
        <span class="growth-chart-legend-item">
            <span class="growth-chart-zero-line"></span>Zero line
        </span>
    </div>
    """


def _create_total_sales_growth_figure(growth_data: pd.DataFrame) -> go.Figure:
    """Create the month-on-month growth chart for the selected sales series."""
    figure = go.Figure()
    if growth_data.empty:
        figure.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=8, r=8, t=10, b=10),
            height=360,
            showlegend=False,
        )
        return figure

    month_categories = growth_data["Month Short"].astype(str).tolist()
    growth_values = pd.to_numeric(growth_data["MoM Growth %"], errors="coerce").fillna(0)
    min_growth = float(growth_values.min()) if not growth_values.empty else 0.0
    max_growth = float(growth_values.max()) if not growth_values.empty else 0.0
    lower_bound = min(min_growth, 0.0)
    upper_bound = max(max_growth, 0.0)
    value_span = upper_bound - lower_bound
    padding = max(value_span * 0.12, 4.0)
    hover_custom_data = growth_data[
        ["Month Label", "Sales Value", "Previous Sales Value"]
    ].to_numpy()

    figure.add_trace(
        go.Scatter(
            x=month_categories,
            y=growth_data["MoM Growth %"],
            customdata=hover_custom_data,
            mode="lines",
            line=dict(color=SAGE_MIST_SECONDARY, width=4, shape="spline", smoothing=0.55),
            fill="tozeroy",
            fillcolor="rgba(167, 139, 250, 0.12)",
            hovertemplate=(
                "%{customdata[0]}"
                "<br>Current Month Sale: €%{customdata[1]:,.2f}"
                "<br>Previous Month Sale: €%{customdata[2]:,.2f}"
                "<br>MoM Change: %{y:.2f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    positive_points = growth_data[growth_data["MoM Growth %"] >= 0]
    negative_points = growth_data[growth_data["MoM Growth %"] < 0]

    if not positive_points.empty:
        figure.add_trace(
            go.Scatter(
                x=positive_points["Month Short"],
                y=positive_points["MoM Growth %"],
                customdata=positive_points[
                    ["Month Label", "Sales Value", "Previous Sales Value"]
                ].to_numpy(),
                mode="markers",
                marker=dict(color=SAGE_MIST_ACCENT, size=14),
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br>Current Month Sale: €%{customdata[1]:,.2f}"
                    "<br>Previous Month Sale: €%{customdata[2]:,.2f}"
                    "<br>MoM Change: %{y:.2f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    if not negative_points.empty:
        figure.add_trace(
            go.Scatter(
                x=negative_points["Month Short"],
                y=negative_points["MoM Growth %"],
                customdata=negative_points[
                    ["Month Label", "Sales Value", "Previous Sales Value"]
                ].to_numpy(),
                mode="markers",
                marker=dict(color=SAGE_MIST_ALERT, size=14),
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br>Current Month Sale: €%{customdata[1]:,.2f}"
                    "<br>Previous Month Sale: €%{customdata[2]:,.2f}"
                    "<br>MoM Change: %{y:.2f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=10, b=10),
        height=360,
        showlegend=False,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=month_categories,
            tickangle=0,
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            linecolor=SAGE_MIST_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=SAGE_MIST_GRID,
            zeroline=True,
            zerolinecolor="rgba(13, 148, 136, 0.18)",
            zerolinewidth=1.5,
            range=[lower_bound - padding, upper_bound + padding],
            tickfont=dict(color=SAGE_MIST_TEXT_MUTED),
            tickformat="+.0f",
            ticksuffix="%",
        ),
        font=dict(color=SAGE_MIST_TEXT_STRONG),
    )
    return figure


def _calculate_overall_sales_kpis(
    cleaned_tables: dict[str, pd.DataFrame],
    selected_quarter: str = "All Quarters",
    selected_channel: str = "Overall",
) -> dict[str, float | int | str]:
    """Calculate the FY 2025-2026 Overall Sales KPIs from cleaned datasets."""
    pos_data = _get_table(cleaned_tables, "pos")
    delivery_financials_data = _get_table(cleaned_tables, "delivery_financials")
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)

    pos_data_fy = _filter_by_date_range(
        pos_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
    )
    delivery_financials_data_fy = _filter_by_date_columns(
        delivery_financials_data,
        start_date=selected_start_date,
        end_date=selected_end_date,
        date_columns=["Date", "Month Start"],
    )

    offline_sales = _safe_sum(pos_data_fy, "Amount")
    online_gross_sales = _safe_sum(delivery_financials_data_fy, "Gross")
    gross_sale = offline_sales + online_gross_sales

    offline_orders = _count_valid_offline_orders(pos_data_fy)
    online_orders = int(round(_safe_sum(delivery_financials_data_fy, "orders")))
    total_orders = offline_orders + online_orders

    offline_sales_share = _safe_ratio(offline_sales, gross_sale)
    online_sales_share = _safe_ratio(online_gross_sales, gross_sale)
    offline_orders_share = _safe_ratio(offline_orders, total_orders)
    online_orders_share = _safe_ratio(online_orders, total_orders)

    offline_aov = offline_sales / offline_orders if offline_orders else 0.0
    online_aov = online_gross_sales / online_orders if online_orders else 0.0
    aov_max = max(offline_aov, online_aov, 0.0)
    monthly_gross_sales = _build_combined_monthly_gross_sales(
        pos_data_fy=pos_data_fy,
        delivery_financials_data_fy=delivery_financials_data_fy,
    )
    overall_active_month_count = _count_active_months(monthly_gross_sales, "Gross Sale")
    offline_active_month_count = _count_active_months(
        monthly_gross_sales, "Offline Gross Sale"
    )
    online_active_month_count = _count_active_months(
        monthly_gross_sales, "Online Gross Sale"
    )

    average_monthly_sale = gross_sale / overall_active_month_count if overall_active_month_count else 0.0
    offline_average_monthly_sale = (
        offline_sales / offline_active_month_count if offline_active_month_count else 0.0
    )
    online_average_monthly_sale = (
        online_gross_sales / online_active_month_count if online_active_month_count else 0.0
    )
    overall_average_month_growth = _calculate_average_month_growth(
        monthly_gross_sales,
        "Gross Sale",
    )
    offline_average_month_growth = _calculate_average_month_growth(
        monthly_gross_sales,
        "Offline Gross Sale",
    )
    online_average_month_growth = _calculate_average_month_growth(
        monthly_gross_sales,
        "Online Gross Sale",
    )

    average_order_value = gross_sale / total_orders if total_orders else 0.0
    if selected_channel == "Offline":
        displayed_gross_sale = offline_sales
        displayed_total_orders = offline_orders
        displayed_average_order_value = offline_aov
        displayed_average_monthly_sale = offline_average_monthly_sale
        displayed_active_month_count = offline_active_month_count
        displayed_average_month_growth = offline_average_month_growth
    elif selected_channel == "Online":
        displayed_gross_sale = online_gross_sales
        displayed_total_orders = online_orders
        displayed_average_order_value = online_aov
        displayed_average_monthly_sale = online_average_monthly_sale
        displayed_active_month_count = online_active_month_count
        displayed_average_month_growth = online_average_month_growth
    else:
        displayed_gross_sale = gross_sale
        displayed_total_orders = total_orders
        displayed_average_order_value = average_order_value
        displayed_average_monthly_sale = average_monthly_sale
        displayed_active_month_count = overall_active_month_count
        displayed_average_month_growth = overall_average_month_growth

    sales_split = _build_channel_sensitive_split(
        offline_value=offline_sales,
        online_value=online_gross_sales,
        selected_channel=selected_channel,
    )
    orders_split = _build_channel_sensitive_split(
        offline_value=offline_orders,
        online_value=online_orders,
        selected_channel=selected_channel,
    )
    aov_split = _build_channel_sensitive_split(
        offline_value=offline_aov,
        online_value=online_aov,
        selected_channel=selected_channel,
    )
    aov_split_max = max(aov_split["offline_value"], aov_split["online_value"], 0.0)

    duplicate_partner_period_rows = _count_partner_period_duplicates(
        delivery_financials_data_fy
    )
    negative_pos_rows = 0
    if not pos_data_fy.empty and "Amount" in pos_data_fy.columns:
        negative_pos_rows = int(
            (pd.to_numeric(pos_data_fy["Amount"], errors="coerce").fillna(0) < 0).sum()
        )

    return {
        "fy_start": str(selected_start_date.date()),
        "fy_end": str(selected_end_date.date()),
        "selected_quarter": selected_quarter,
        "selected_channel": selected_channel,
        "offline_sales": offline_sales,
        "online_gross_sales": online_gross_sales,
        "gross_sale": gross_sale,
        "offline_orders": offline_orders,
        "online_orders": online_orders,
        "total_orders": total_orders,
        "offline_sales_share": offline_sales_share,
        "online_sales_share": online_sales_share,
        "offline_orders_share": offline_orders_share,
        "online_orders_share": online_orders_share,
        "active_month_count": displayed_active_month_count,
        "average_monthly_sale": displayed_average_monthly_sale,
        "offline_aov": offline_aov,
        "online_aov": online_aov,
        "offline_aov_ratio": _safe_ratio(offline_aov, aov_max) if aov_max else 0.0,
        "online_aov_ratio": _safe_ratio(online_aov, aov_max) if aov_max else 0.0,
        "split_offline_sales": sales_split["offline_value"],
        "split_online_sales": sales_split["online_value"],
        "split_offline_sales_share": sales_split["offline_share"],
        "split_online_sales_share": sales_split["online_share"],
        "split_offline_orders": orders_split["offline_value"],
        "split_online_orders": orders_split["online_value"],
        "split_offline_orders_share": orders_split["offline_share"],
        "split_online_orders_share": orders_split["online_share"],
        "split_offline_aov": aov_split["offline_value"],
        "split_online_aov": aov_split["online_value"],
        "split_offline_aov_ratio": _safe_ratio(
            aov_split["offline_value"], aov_split_max
        )
        if aov_split_max
        else 0.0,
        "split_online_aov_ratio": _safe_ratio(
            aov_split["online_value"], aov_split_max
        )
        if aov_split_max
        else 0.0,
        "average_order_value": average_order_value,
        "displayed_gross_sale": displayed_gross_sale,
        "displayed_total_orders": displayed_total_orders,
        "displayed_average_order_value": displayed_average_order_value,
        "displayed_average_month_growth": displayed_average_month_growth,
        "negative_pos_rows": negative_pos_rows,
        "duplicate_partner_period_rows": duplicate_partner_period_rows,
    }


def _show_placeholder_section(title: str, description: str) -> None:
    """Render an empty dashboard section for future build-out."""
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(description)


def render_overall_sales_page(cleaned_tables: dict[str, pd.DataFrame]) -> None:
    """Render the executive overall sales shell page."""
    _apply_sage_mist_theme()
    st.markdown(
        """
        <style>
        div.block-container {
            padding-top: 2.75rem;
        }
        .overall-header-title {
            margin: 0.25rem 0 0 0;
            padding: 0;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1.2;
            color: #FFFFFF;
        }
        .overall-filter-label {
            color: var(--sage-muted);
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }
        .kpi-card {
            background: linear-gradient(180deg, rgba(31,41,55,0.98) 0%, rgba(17,24,39,0.98) 100%);
            border: 1px solid var(--sage-border);
            border-radius: 22px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            min-height: 9.9rem;
            height: 100%;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: var(--sage-shadow);
            position: relative;
            overflow: hidden;
        }
        .kpi-card-label {
            color: var(--sage-muted);
            font-size: 0.98rem;
            font-weight: 600;
            line-height: 1.3;
            min-height: 1.45rem;
        }
        .kpi-card-value {
            color: var(--sage-text);
            font-size: clamp(1.55rem, 0.75vw + 1.1rem, 2.2rem);
            font-weight: 700;
            line-height: 1.12;
            margin-top: 0.4rem;
            margin-bottom: 0.45rem;
        }
        .kpi-card-support {
            color: var(--sage-muted);
            font-size: 0.94rem;
            line-height: 1.35;
            min-height: 1.35rem;
        }
        .kpi-card-support.is-empty {
            visibility: hidden;
        }
        .kpi-card-placeholder {
            min-height: 9.9rem;
            height: 100%;
            box-sizing: border-box;
            border-radius: 22px;
            background: transparent;
            border: 1px solid transparent;
        }
        .split-card {
            background: linear-gradient(180deg, rgba(31,41,55,0.98) 0%, rgba(17,24,39,0.98) 100%);
            border: 1px solid var(--sage-border);
            border-radius: 24px;
            padding: 1.2rem;
            min-height: 17rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            box-shadow: var(--sage-shadow);
        }
        .split-card-title {
            color: var(--sage-muted);
            font-size: 0.98rem;
            font-weight: 600;
            margin-bottom: 1.1rem;
            min-height: 1.45rem;
            display: flex;
            align-items: center;
        }
        .split-card-layout {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(10.75rem, 11.25rem);
            gap: 1rem;
            align-items: stretch;
            flex: 1 1 auto;
        }
        .split-card-left {
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 1rem;
            height: 100%;
        }
        .split-row {
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 4.55rem;
        }
        .split-row-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.55rem;
        }
        .split-label {
            color: var(--sage-text);
            font-size: 1.15rem;
            font-weight: 600;
        }
        .split-share {
            font-size: 1.1rem;
            font-weight: 700;
            min-width: 2.6rem;
            text-align: right;
        }
        .split-share.offline {
            color: var(--sage-accent);
        }
        .split-share.online {
            color: var(--sage-online);
        }
        .split-share.is-empty {
            visibility: hidden;
        }
        .split-track {
            width: 100%;
            height: 0.72rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.16);
            overflow: hidden;
        }
        .split-fill {
            height: 100%;
            border-radius: 999px;
        }
        .split-fill.offline {
            background: linear-gradient(90deg, var(--sage-accent) 0%, #6ee7b7 100%);
        }
        .split-fill.online {
            background: linear-gradient(90deg, var(--sage-online) 0%, #93c5fd 100%);
        }
        .split-card-right {
            display: flex;
            flex-direction: column;
            justify-content: stretch;
            gap: 0.9rem;
            height: 100%;
            min-width: 0;
        }
        .split-value-card {
            border-radius: 18px;
            padding: 0.85rem 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            flex: 1 1 0;
            min-height: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
            overflow: hidden;
        }
        .split-value-card.offline {
            background: rgba(52, 211, 153, 0.10);
        }
        .split-value-card.online {
            background: rgba(96, 165, 250, 0.12);
        }
        .split-value-label {
            color: var(--sage-muted);
            font-size: 0.92rem;
            margin-bottom: 0.35rem;
        }
        .split-value-number {
            width: 100%;
            display: block;
            font-size: clamp(0.92rem, 0.35vw + 0.78rem, 1rem);
            font-weight: 700;
            line-height: 1.2;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .split-value-number.offline {
            color: var(--sage-accent);
        }
        .split-value-number.online {
            color: var(--sage-online);
        }
        .sales-breakdown-title {
            color: var(--sage-text);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }
        .sales-breakdown-legend {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 1.35rem;
            color: var(--sage-muted);
            font-size: 0.95rem;
            font-weight: 600;
            white-space: nowrap;
            padding-top: 0.45rem;
        }
        .sales-breakdown-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .sales-breakdown-dot {
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 0.28rem;
            display: inline-block;
        }
        .sales-breakdown-dot.offline {
            background: var(--sage-accent);
        }
        .sales-breakdown-dot.online {
            background: var(--sage-online);
        }
        .sales-breakdown-dot.total {
            background: var(--sage-secondary);
        }
        .sales-chart-header {
            min-height: 3.8rem;
            margin-bottom: 0.35rem;
        }
        .sales-chart-title {
            color: var(--sage-text);
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.18rem;
        }
        .sales-chart-subtitle {
            color: var(--sage-muted);
            font-size: 0.96rem;
            line-height: 1.35;
        }
        .growth-section-title {
            color: var(--sage-text);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }
        .growth-section-subtitle {
            color: var(--sage-muted);
            font-size: 1rem;
            line-height: 1.35;
            margin-top: 0.28rem;
        }
        .growth-summary-stack {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .growth-summary-card {
            background: linear-gradient(180deg, rgba(31,41,55,0.98) 0%, rgba(17,24,39,0.98) 100%);
            border: 1px solid var(--sage-border);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            min-height: 7.6rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }
        .growth-summary-title {
            color: var(--sage-muted);
            font-size: 0.96rem;
            font-weight: 600;
            line-height: 1.3;
            margin-bottom: 0.5rem;
        }
        .growth-summary-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.08;
            margin-bottom: 0.38rem;
        }
        .growth-summary-value.positive {
            color: #53d78b;
        }
        .growth-summary-value.negative {
            color: #ff7272;
        }
        .growth-summary-value.neutral {
            color: var(--sage-text);
        }
        .growth-summary-subtext {
            color: var(--sage-muted);
            font-size: 0.96rem;
            line-height: 1.3;
        }
        .growth-chart-legend {
            display: flex;
            justify-content: flex-start;
            align-items: center;
            gap: 1.35rem;
            color: var(--sage-muted);
            font-size: 0.95rem;
            font-weight: 600;
            white-space: nowrap;
            padding-top: 0.15rem;
            padding-bottom: 0.55rem;
        }
        .growth-chart-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .growth-chart-dot {
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 999px;
            display: inline-block;
        }
        .growth-chart-dot.positive {
            background: var(--sage-accent);
        }
        .growth-chart-dot.negative {
            background: var(--sage-alert);
        }
        .growth-chart-zero-line {
            width: 2rem;
            height: 0.14rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.26);
            display: inline-block;
        }
        .payment-summary-card {
            background: var(--sage-surface);
            border: 1px solid var(--sage-border);
            border-radius: 20px;
            padding: 0.7rem 0.9rem;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 6.35rem;
            box-shadow: var(--sage-shadow);
        }
        .payment-summary-card.cash {
            background: rgba(45, 212, 191, 0.12);
        }
        .payment-summary-card.card {
            background: rgba(96, 165, 250, 0.12);
        }
        .payment-summary-card.bank {
            background: rgba(167, 139, 250, 0.14);
        }
        .payment-summary-head {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.5rem;
        }
        .payment-summary-dot {
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 0.28rem;
            display: inline-block;
        }
        .payment-summary-dot.cash {
            background: var(--sage-accent);
        }
        .payment-summary-dot.card {
            background: var(--sage-online);
        }
        .payment-summary-dot.bank {
            background: var(--sage-secondary);
        }
        .payment-summary-label {
            color: var(--sage-muted);
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.3;
        }
        .payment-summary-value {
            color: var(--sage-text);
            font-size: clamp(1.28rem, 0.45vw + 0.98rem, 1.72rem);
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.24rem;
            overflow-wrap: anywhere;
        }
        .payment-summary-share {
            font-size: 0.94rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.28rem;
        }
        .payment-summary-share.cash {
            color: var(--sage-accent);
        }
        .payment-summary-share.card {
            color: var(--sage-online);
        }
        .payment-summary-share.bank {
            color: var(--sage-secondary);
        }
        .payment-summary-track {
            width: 78%;
            height: 0.42rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.16);
            overflow: hidden;
        }
        .payment-summary-fill {
            height: 100%;
            border-radius: 999px;
        }
        .payment-summary-fill.cash {
            background: linear-gradient(90deg, var(--sage-accent) 0%, #6ee7b7 100%);
        }
        .payment-summary-fill.card {
            background: linear-gradient(90deg, var(--sage-online) 0%, #93c5fd 100%);
        }
        .payment-summary-fill.bank {
            background: linear-gradient(90deg, var(--sage-secondary) 0%, #c4b5fd 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_column, quarter_column, channel_column = st.columns(
        [6.4, 1.6, 1.6],
        gap="small",
    )

    with title_column:
        st.markdown(
            '<h1 class="overall-header-title">IBC FY 2025-2026 Overall Performance</h1>',
            unsafe_allow_html=True,
        )

    with quarter_column:
        st.markdown('<div class="overall-filter-label">Quarter</div>', unsafe_allow_html=True)
        selected_quarter = st.selectbox(
            "Quarter",
            OVERALL_SALES_QUARTER_OPTIONS,
            index=0,
            key="overall_sales_quarter_filter",
            label_visibility="collapsed",
        )

    with channel_column:
        st.markdown('<div class="overall-filter-label">Channel</div>', unsafe_allow_html=True)
        selected_channel = st.selectbox(
            "Channel",
            OVERALL_SALES_CHANNEL_OPTIONS,
            index=0,
            key="overall_sales_channel_filter",
            label_visibility="collapsed",
        )

    kpi_values = _calculate_overall_sales_kpis(
        cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )

    st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

    first_row_columns = st.columns(3, gap="large")

    with first_row_columns[0]:
        st.markdown(
            _build_kpi_card_html(
                label="Gross Sale",
                value_text=_format_currency(float(kpi_values["displayed_gross_sale"])),
            ),
            unsafe_allow_html=True,
        )

    with first_row_columns[1]:
        st.markdown(
            _build_kpi_card_html(
                label="Average Monthly Sale",
                value_text=_format_currency(float(kpi_values["average_monthly_sale"])),
                support_text=_format_month_support_text(
                    int(kpi_values["active_month_count"])
                ),
            ),
            unsafe_allow_html=True,
        )

    with first_row_columns[2]:
        st.markdown(
            _build_kpi_card_html(
                label="Average Month Growth %",
                value_text=_format_percentage_value(
                    float(kpi_values["displayed_average_month_growth"])
                ),
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    second_row_columns = st.columns(3, gap="large")

    with second_row_columns[0]:
        st.markdown(
            _build_kpi_card_html(
                label="Average Order Value",
                value_text=_format_currency(
                    float(kpi_values["displayed_average_order_value"])
                ),
            ),
            unsafe_allow_html=True,
        )

    with second_row_columns[1]:
        st.markdown(
            _build_kpi_card_html(
                label="Total Orders",
                value_text=_format_whole_number(int(kpi_values["displayed_total_orders"])),
            ),
            unsafe_allow_html=True,
        )

    with second_row_columns[2]:
        st.markdown(_build_kpi_placeholder_html(), unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    split_columns = st.columns(3, gap="large")

    with split_columns[0]:
        st.markdown(
            _build_split_block_html(
                block_title="Gross Sale Split",
                offline_value_text=_format_currency(
                    float(kpi_values["split_offline_sales"])
                ),
                online_value_text=_format_currency(
                    float(kpi_values["split_online_sales"])
                ),
                offline_share_text=_format_percent(
                    float(kpi_values["split_offline_sales_share"])
                ),
                online_share_text=_format_percent(
                    float(kpi_values["split_online_sales_share"])
                ),
                offline_bar_ratio=float(kpi_values["split_offline_sales_share"]),
                online_bar_ratio=float(kpi_values["split_online_sales_share"]),
            ),
            unsafe_allow_html=True,
        )

    with split_columns[1]:
        st.markdown(
            _build_split_block_html(
                block_title="Total Orders Split",
                offline_value_text=_format_whole_number(
                    int(kpi_values["split_offline_orders"])
                ),
                online_value_text=_format_whole_number(
                    int(kpi_values["split_online_orders"])
                ),
                offline_share_text=_format_percent(
                    float(kpi_values["split_offline_orders_share"])
                ),
                online_share_text=_format_percent(
                    float(kpi_values["split_online_orders_share"])
                ),
                offline_bar_ratio=float(kpi_values["split_offline_orders_share"]),
                online_bar_ratio=float(kpi_values["split_online_orders_share"]),
            ),
            unsafe_allow_html=True,
        )

    with split_columns[2]:
        st.markdown(
            _build_split_block_html(
                block_title="Average Order Value Split",
                offline_value_text=_format_currency(float(kpi_values["split_offline_aov"])),
                online_value_text=_format_currency(float(kpi_values["split_online_aov"])),
                offline_share_text="",
                online_share_text="",
                offline_bar_ratio=float(kpi_values["split_offline_aov_ratio"]),
                online_bar_ratio=float(kpi_values["split_online_aov_ratio"]),
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)

    monthly_sales_breakdown = _prepare_sales_breakdown_data(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
    )

    section_title_column, section_legend_column = st.columns([3.4, 2.2], gap="small")
    with section_title_column:
        st.markdown(
            '<div class="sales-breakdown-title">Sales breakdown</div>',
            unsafe_allow_html=True,
        )
    with section_legend_column:
        st.markdown(_build_sales_breakdown_legend_html(), unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    chart_columns = st.columns(2, gap="large")

    with chart_columns[0]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Monthly total gross sale",
                    subtitle="Offline + online combined per month",
                ),
                unsafe_allow_html=True,
            )
            if monthly_sales_breakdown.empty:
                st.info("Monthly total gross sale will appear here when monthly FY data is available.")
            else:
                st.plotly_chart(
                    _create_monthly_total_gross_sale_figure(monthly_sales_breakdown),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    with chart_columns[1]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Online vs offline trend",
                    subtitle="Month-by-month channel comparison",
                ),
                unsafe_allow_html=True,
            )
            if monthly_sales_breakdown.empty:
                st.info("Online and offline monthly trend will appear here when monthly FY data is available.")
            else:
                st.plotly_chart(
                    _create_online_vs_offline_trend_figure(monthly_sales_breakdown),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    payment_method_breakdown = _prepare_payment_method_breakdown_data(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )
    growth_chart_row = st.columns(2, gap="large")
    with growth_chart_row[0]:
        with st.container(border=True):
            growth_header_column, growth_filter_column = st.columns(
                [3.2, 1.2], gap="small"
            )
            with growth_filter_column:
                st.markdown(
                    '<div class="overall-filter-label">Channel</div>',
                    unsafe_allow_html=True,
                )
                growth_channel_option = st.selectbox(
                    "Growth Chart Channel",
                    GROWTH_CHART_CHANNEL_OPTIONS,
                    index=0,
                    key="overall_sales_growth_channel_filter",
                    label_visibility="collapsed",
                )
            with growth_header_column:
                growth_subtitle = {
                    "Combined": "Sales change vs previous month · combined gross sales",
                    "In-house": "Sales change vs previous month · in-house gross sales",
                    "Online": "Sales change vs previous month · online gross sales",
                }.get(
                    growth_channel_option,
                    "Sales change vs previous month · combined gross sales",
                )
                st.markdown(
                    _build_chart_card_header_html(
                        title="Month-on-Month Change Percentage",
                        subtitle=growth_subtitle,
                    ),
                    unsafe_allow_html=True,
                )

            growth_channel_map = {
                "Combined": ("Overall", "Gross Sale"),
                "In-house": ("Offline", "Offline Gross Sale"),
                "Online": ("Online", "Online Gross Sale"),
            }
            growth_dashboard_channel, growth_sales_column = growth_channel_map.get(
                growth_channel_option,
                ("Overall", "Gross Sale"),
            )
            growth_monthly_sales = _prepare_sales_breakdown_data(
                cleaned_tables=cleaned_tables,
                selected_quarter=selected_quarter,
                selected_channel=growth_dashboard_channel,
            )
            _log_total_sales_growth_debug(growth_monthly_sales)
            total_sales_growth_data = _prepare_total_sales_growth_data(
                growth_monthly_sales,
                sales_column=growth_sales_column,
            )
            if total_sales_growth_data.empty:
                st.info(
                    "Month-on-month total sales growth will appear here when at least two valid active months are available."
                )
            else:
                st.plotly_chart(
                    _create_total_sales_growth_figure(total_sales_growth_data),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    with growth_chart_row[1]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Payment method breakdown",
                    subtitle="Cash vs Card Payment vs Delivery Partner",
                ),
                unsafe_allow_html=True,
            )
            if (
                payment_method_breakdown["cash_value"] == 0
                and payment_method_breakdown["card_payment_value"] == 0
                and payment_method_breakdown["bank_deposit_value"] == 0
            ):
                st.info(
                    "Payment method breakdown will appear here when gross sales data is available for the current filter."
                )
            else:
                payment_chart_column, payment_summary_column = st.columns(
                    [1.0, 1.0],
                    gap="medium",
                )
                with payment_chart_column:
                    st.plotly_chart(
                        _create_payment_method_breakdown_figure(
                            payment_method_breakdown
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False, "responsive": True},
                    )

                with payment_summary_column:
                    st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        _build_payment_method_summary_card_html(
                            label="Cash",
                            value_text=_format_currency(
                                float(payment_method_breakdown["cash_value"])
                            ),
                            share_text=_format_percent(
                                float(payment_method_breakdown["cash_share"])
                            ),
                            fill_ratio=float(payment_method_breakdown["cash_share"]),
                            tone_class="cash",
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        _build_payment_method_summary_card_html(
                            label="Card Payment",
                            value_text=_format_currency(
                                float(payment_method_breakdown["card_payment_value"])
                            ),
                            share_text=_format_percent(
                                float(payment_method_breakdown["card_payment_share"])
                            ),
                            fill_ratio=float(
                                payment_method_breakdown["card_payment_share"]
                            ),
                            tone_class="card",
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        _build_payment_method_summary_card_html(
                            label="Delivery Partner",
                            value_text=_format_currency(
                                float(payment_method_breakdown["bank_deposit_value"])
                            ),
                            share_text=_format_percent(
                                float(payment_method_breakdown["bank_deposit_share"])
                            ),
                            fill_ratio=float(
                                payment_method_breakdown["bank_deposit_share"]
                            ),
                            tone_class="bank",
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)

    monthly_orders_breakdown = _prepare_orders_breakdown_data(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )

    orders_title_column, orders_legend_column = st.columns([3.4, 2.2], gap="small")
    with orders_title_column:
        st.markdown(
            '<div class="sales-breakdown-title">Orders breakdown</div>',
            unsafe_allow_html=True,
        )
    with orders_legend_column:
        st.markdown(_build_sales_breakdown_legend_html(), unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    orders_chart_columns = st.columns(2, gap="large")

    with orders_chart_columns[0]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Monthly total orders",
                    subtitle="Offline + online combined per month",
                ),
                unsafe_allow_html=True,
            )
            if monthly_orders_breakdown.empty:
                st.info(
                    "Monthly total orders will appear here when monthly FY order data is available."
                )
            else:
                st.plotly_chart(
                    _create_monthly_total_orders_figure(monthly_orders_breakdown),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    with orders_chart_columns[1]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Online vs offline orders",
                    subtitle="Month-by-month channel comparison",
                ),
                unsafe_allow_html=True,
            )
            if monthly_orders_breakdown.empty:
                st.info(
                    "Online and offline monthly orders will appear here when monthly FY order data is available."
                )
            else:
                st.plotly_chart(
                    _create_online_vs_offline_orders_figure(monthly_orders_breakdown),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )


def render_performance_analysis_page(
    cleaned_tables: dict[str, pd.DataFrame],
    raw_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Render the operational performance shell page."""
    _apply_sage_mist_theme()
    if raw_tables is None:
        raw_tables = {}

    st.markdown(
        """
        <style>
        div.block-container {
            padding-top: 1rem;
        }
        .deep-insights-title {
            margin: 0;
            padding: 0;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1.2;
            color: #FFFFFF;
        }
        .deep-insights-filter-label {
            color: var(--sage-muted);
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.32rem;
        }
        .deep-section-title {
            color: var(--sage-text);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }
        .month-analysis-card-title {
            color: var(--sage-muted);
            font-size: 0.98rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 1.15rem;
        }
        .month-analysis-side-card {
            min-height: 13.25rem;
            padding: 0.45rem 0.2rem 0.35rem 0.2rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }
        .month-analysis-card-empty {
            color: var(--sage-muted);
            font-size: 0.98rem;
            line-height: 1.45;
        }
        .month-analysis-legend {
            display: flex;
            align-items: center;
            gap: 1.35rem;
            padding-top: 0.1rem;
            padding-bottom: 0.55rem;
        }
        .month-analysis-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.94rem;
            font-weight: 700;
            line-height: 1;
            color: var(--sage-text);
        }
        .month-analysis-legend-dot {
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 0.22rem;
            display: inline-block;
            border: 1px solid rgba(248, 250, 252, 0.12);
        }
        .month-analysis-legend-dot.total {
            background: var(--sage-secondary);
        }
        .month-analysis-legend-dot.offline {
            background: var(--sage-accent);
        }
        .month-analysis-legend-dot.online {
            background: var(--sage-online);
        }
        .week-analysis-legend {
            display: flex;
            align-items: center;
            gap: 1.15rem;
            padding-top: 0.08rem;
            padding-bottom: 0.6rem;
        }
        .week-analysis-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.94rem;
            font-weight: 700;
            line-height: 1;
            color: var(--sage-text);
        }
        .week-analysis-legend-dot {
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 0.22rem;
            display: inline-block;
            border: 1px solid rgba(248, 250, 252, 0.12);
        }
        .week-analysis-legend-dot.offline {
            background: var(--sage-accent);
        }
        .week-analysis-legend-dot.online {
            background: var(--sage-online);
        }
        .hourly-analysis-legend {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding-top: 0.2rem;
            white-space: nowrap;
        }
        .hourly-analysis-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--sage-text);
            font-size: 0.9rem;
            font-weight: 700;
            line-height: 1;
        }
        .hourly-analysis-legend-dot {
            width: 0.82rem;
            height: 0.82rem;
            border-radius: 999px;
            display: inline-block;
            border: 1px solid rgba(248, 250, 252, 0.12);
        }
        .hourly-analysis-legend-dot.high {
            background: var(--sage-accent);
        }
        .hourly-analysis-legend-dot.medium {
            background: var(--sage-secondary);
        }
        .hourly-analysis-legend-dot.low {
            background: var(--sage-alert);
        }
        .month-analysis-rank-stack {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .month-analysis-rank-row {
            display: grid;
            grid-template-columns: 4rem 1fr auto;
            align-items: center;
            gap: 1rem;
            min-height: 2.15rem;
        }
        .month-analysis-rank-month {
            color: var(--sage-text);
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .month-analysis-rank-value {
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.2;
            white-space: nowrap;
        }
        .month-analysis-rank-value.heaviest {
            color: var(--sage-secondary);
        }
        .month-analysis-rank-value.lighter {
            color: var(--sage-primary);
        }
        .month-analysis-rank-track {
            width: 100%;
            height: 0.4rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.14);
            overflow: hidden;
        }
        .month-analysis-rank-fill {
            height: 100%;
            border-radius: 999px;
        }
        .month-analysis-rank-fill.heaviest {
            background: var(--sage-secondary);
        }
        .month-analysis-rank-fill.lighter {
            background: var(--sage-primary);
        }
        .weekday-analysis-rank-stack {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .weekday-analysis-rank-row {
            display: grid;
            grid-template-columns: 6.75rem 1fr auto;
            align-items: center;
            gap: 0.95rem;
            min-height: 2.15rem;
        }
        .weekday-analysis-rank-day {
            color: var(--sage-text);
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .weekday-analysis-rank-value {
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.2;
            white-space: nowrap;
        }
        .weekday-analysis-rank-value.peak {
            color: var(--sage-secondary);
        }
        .weekday-analysis-rank-value.quiet {
            color: var(--sage-primary);
        }
        .weekday-analysis-rank-fill {
            height: 100%;
            border-radius: 999px;
        }
        .weekday-analysis-rank-fill.peak {
            background: var(--sage-secondary);
        }
        .weekday-analysis-rank-fill.quiet {
            background: var(--sage-primary);
        }
        .deep-hourly-kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 1rem;
        }
        .deep-hourly-kpi-card {
            background: linear-gradient(180deg, rgba(31,41,55,0.98) 0%, rgba(17,24,39,0.98) 100%);
            border: 1px solid var(--sage-border);
            border-radius: 22px;
            padding: 0.95rem 1.05rem 0.9rem 1.05rem;
            min-height: 7.75rem;
            box-sizing: border-box;
        }
        .deep-hourly-kpi-label {
            color: var(--sage-muted);
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.85rem;
        }
        .deep-hourly-kpi-value {
            color: var(--sage-text);
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.32rem;
        }
        .deep-hourly-kpi-support {
            color: var(--sage-muted);
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .operational-notes-card {
            min-height: 8.8rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }
        .operational-notes-title {
            color: var(--sage-text);
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.8rem;
        }
        .operational-notes-body {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        .operational-note-line {
            color: var(--sage-muted);
            font-size: 0.96rem;
            line-height: 1.45;
        }
        .deep-section-subtitle {
            color: var(--sage-muted);
            font-size: 1rem;
            line-height: 1.45;
            margin-top: 0.22rem;
        }
        .shift-panel-title {
            color: var(--sage-muted);
            font-size: 0.98rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }
        .shift-input-label {
            color: var(--sage-text);
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.35;
            margin-bottom: 0.42rem;
        }
        .shift-staff-header {
            color: var(--sage-muted);
            font-size: 0.94rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.45rem;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 2.35rem;
            border-radius: 14px;
            border: 1px solid var(--sage-border);
            background: rgba(31, 41, 55, 0.98);
            color: var(--sage-text);
            font-size: 0.96rem;
            font-weight: 700;
            box-shadow: none;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(20, 184, 166, 0.42);
            color: var(--sage-text);
        }
        div[data-testid="stButton"] > button:focus {
            box-shadow: none;
            border-color: rgba(20, 184, 166, 0.52);
        }
        div[data-testid="stNumberInput"] input {
            background: rgba(15, 23, 42, 0.75) !important;
            color: var(--sage-text) !important;
            border: 1px solid var(--sage-border) !important;
            border-radius: 18px !important;
        }
        .shift-summary-card,
        .shift-data-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            box-sizing: border-box;
        }
        .shift-data-card {
            min-height: 29rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .shift-summary-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.9rem;
            align-items: center;
            min-height: 1.95rem;
        }
        .shift-summary-label {
            color: var(--sage-muted);
            font-size: 0.98rem;
            line-height: 1.3;
        }
        .shift-summary-value {
            color: var(--sage-text);
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.3;
            text-align: right;
        }
        .shift-summary-row.total .shift-summary-label,
        .shift-summary-row.total .shift-summary-value {
            color: var(--sage-accent);
            font-weight: 700;
        }
        .shift-summary-divider {
            width: 100%;
            height: 1px;
            background: rgba(148, 163, 184, 0.14);
            margin: 0.8rem 0;
        }
        .shift-breakdown-title {
            color: var(--sage-muted);
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.6rem;
        }
        .shift-employee-breakdown-head,
        .shift-employee-breakdown-row {
            display: grid;
            grid-template-columns: 1.2fr 1.4fr 0.8fr 0.9fr 1fr;
            gap: 0.75rem;
            align-items: center;
        }
        .shift-employee-breakdown-head {
            color: var(--sage-muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding-bottom: 0.42rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            margin-bottom: 0.3rem;
        }
        .shift-employee-breakdown-row {
            min-height: 2.2rem;
        }
        .shift-employee-breakdown-name,
        .shift-employee-breakdown-time,
        .shift-employee-breakdown-hours,
        .shift-employee-breakdown-wage,
        .shift-employee-breakdown-cost {
            color: var(--sage-text);
            font-size: 0.9rem;
            line-height: 1.3;
        }
        .shift-employee-breakdown-name {
            font-weight: 700;
        }
        .shift-employee-breakdown-time,
        .shift-employee-breakdown-hours {
            color: var(--sage-muted);
        }
        .shift-employee-breakdown-wage,
        .shift-employee-breakdown-cost {
            text-align: right;
            font-weight: 700;
        }
        .shift-data-block-label {
            color: var(--sage-muted);
            font-size: 0.98rem;
            line-height: 1.3;
            margin-bottom: 0.78rem;
        }
        .shift-data-hour-stack {
            display: flex;
            flex-direction: column;
            gap: 0.38rem;
        }
        .shift-data-hour-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.9rem;
            align-items: center;
            min-height: 1.8rem;
        }
        .shift-data-hour-label {
            color: var(--sage-muted);
            font-size: 0.96rem;
            line-height: 1.3;
        }
        .shift-data-hour-value {
            color: var(--sage-secondary);
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1.3;
            text-align: right;
        }
        .shift-result-card {
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.14);
            min-height: 12.2rem;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .shift-result-card.profit {
            background: linear-gradient(180deg, rgba(5, 46, 22, 0.95) 0%, rgba(6, 78, 59, 0.95) 100%);
            border-color: rgba(52, 211, 153, 0.24);
        }
        .shift-result-card.loss {
            background: linear-gradient(180deg, rgba(76, 5, 25, 0.95) 0%, rgba(127, 29, 29, 0.95) 100%);
            border-color: rgba(251, 113, 133, 0.24);
        }
        .shift-result-card.neutral {
            background: linear-gradient(180deg, rgba(31, 41, 55, 0.98) 0%, rgba(17, 24, 39, 0.98) 100%);
        }
        .shift-result-title {
            color: var(--sage-muted);
            font-size: 0.98rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.95rem;
        }
        .shift-result-layout {
            display: grid;
            grid-template-columns: minmax(0, 1.6fr) auto;
            gap: 1.25rem;
            align-items: center;
        }
        .shift-result-value {
            color: var(--sage-text);
            font-size: clamp(2rem, 1.3vw + 1.3rem, 3.2rem);
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }
        .shift-result-card.profit .shift-result-value,
        .shift-result-card.profit .shift-result-label,
        .shift-result-card.profit .shift-result-margin-value {
            color: var(--sage-accent);
        }
        .shift-result-card.loss .shift-result-value,
        .shift-result-card.loss .shift-result-label,
        .shift-result-card.loss .shift-result-margin-value {
            color: var(--sage-alert);
        }
        .shift-result-label {
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }
        .shift-result-helper {
            color: var(--sage-muted);
            font-size: 0.98rem;
            line-height: 1.42;
        }
        .shift-result-margin-block {
            text-align: right;
        }
        .shift-result-margin-label {
            color: var(--sage-muted);
            font-size: 0.98rem;
            line-height: 1.25;
            margin-bottom: 0.2rem;
        }
        .shift-result-margin-value {
            color: var(--sage-text);
            font-size: clamp(1.8rem, 1.1vw + 1.2rem, 2.8rem);
            font-weight: 700;
            line-height: 1.05;
        }
        .scenario-kpi-card {
            background: linear-gradient(180deg, rgba(31, 41, 55, 0.98) 0%, rgba(17, 24, 39, 0.98) 100%);
            border: 1px solid var(--sage-border);
            border-radius: 22px;
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            min-height: 8.2rem;
            box-sizing: border-box;
        }
        .scenario-kpi-card.positive {
            border-color: rgba(52, 211, 153, 0.26);
        }
        .scenario-kpi-card.negative {
            border-color: rgba(251, 113, 133, 0.28);
        }
        .scenario-kpi-title {
            color: var(--sage-muted);
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.75rem;
        }
        .scenario-kpi-value {
            color: var(--sage-text);
            font-size: clamp(1.7rem, 1vw + 1.2rem, 2.4rem);
            font-weight: 700;
            line-height: 1.08;
            margin-bottom: 0.32rem;
        }
        .scenario-kpi-card.positive .scenario-kpi-value {
            color: var(--sage-accent);
        }
        .scenario-kpi-card.negative .scenario-kpi-value {
            color: var(--sage-alert);
        }
        .scenario-kpi-support {
            color: var(--sage-muted);
            font-size: 0.9rem;
            line-height: 1.38;
        }
        .scenario-split-card {
            background: linear-gradient(180deg, rgba(31, 41, 55, 0.98) 0%, rgba(17, 24, 39, 0.98) 100%);
            border: 1px solid var(--sage-border);
            border-radius: 22px;
            padding: 1rem 1.05rem 0.9rem 1.05rem;
            min-height: 10.8rem;
            box-sizing: border-box;
        }
        .scenario-split-title {
            color: var(--sage-text);
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.8rem;
        }
        .scenario-split-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.7rem;
            align-items: center;
            min-height: 2rem;
        }
        .scenario-split-label {
            color: var(--sage-muted);
            font-size: 0.94rem;
            line-height: 1.3;
        }
        .scenario-split-label.projected {
            color: var(--sage-text);
            font-weight: 600;
        }
        .scenario-split-value {
            color: var(--sage-text);
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.3;
            text-align: right;
        }
        .scenario-split-footer {
            margin-top: 0.85rem;
            padding-top: 0.7rem;
            border-top: 1px solid rgba(148, 163, 184, 0.12);
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.3;
            text-align: center;
        }
        .scenario-split-footer.positive {
            color: var(--sage-accent);
        }
        .scenario-split-footer.negative {
            color: var(--sage-alert);
        }
        .scenario-split-footer.neutral {
            color: var(--sage-text);
        }
        .scenario-summary-box {
            background: linear-gradient(180deg, rgba(31, 41, 55, 0.98) 0%, rgba(17, 24, 39, 0.98) 100%);
            border: 1px solid var(--sage-border);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-sizing: border-box;
        }
        .scenario-summary-line {
            color: var(--sage-text);
            font-size: 0.98rem;
            line-height: 1.55;
        }
        .scenario-summary-line + .scenario-summary-line {
            margin-top: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    title_column, quarter_column, channel_column = st.columns([6.4, 1.6, 1.6], gap="small")

    with title_column:
        st.markdown('<h1 class="deep-insights-title">Deep Insights</h1>', unsafe_allow_html=True)

    with quarter_column:
        st.markdown(
            '<div class="deep-insights-filter-label">Quarter</div>',
            unsafe_allow_html=True,
        )
        selected_quarter = st.selectbox(
            "Quarter",
            OVERALL_SALES_QUARTER_OPTIONS,
            index=0,
            key="deep_insights_quarter_filter",
            label_visibility="collapsed",
        )

    with channel_column:
        st.markdown(
            '<div class="deep-insights-filter-label">Channel</div>',
            unsafe_allow_html=True,
        )
        selected_channel = st.selectbox(
            "Channel",
            OVERALL_SALES_CHANNEL_OPTIONS,
            index=0,
            key="deep_insights_channel_filter",
            label_visibility="collapsed",
        )

    st.markdown("<div style='height: 0.95rem;'></div>", unsafe_allow_html=True)

    monthly_orders = _prepare_orders_breakdown_data(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )
    weekday_orders = _prepare_weekday_orders_analysis_data(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )
    hourly_orders = _prepare_hourly_orders_analysis_data(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )
    overall_kpi_values = _calculate_overall_sales_kpis(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )
    overall_average_order_value = float(
        overall_kpi_values["displayed_average_order_value"]
    )
    estimated_hourly_sales_series = (
        pd.to_numeric(hourly_orders["Total Average Orders"], errors="coerce").fillna(0)
        * overall_average_order_value
        if not hourly_orders.empty
        else pd.Series(dtype="float64")
    )
    average_hourly_sale = (
        float(estimated_hourly_sales_series.mean())
        if not estimated_hourly_sales_series.empty
        else 0.0
    )
    monthly_sales_for_kpis = _prepare_sales_breakdown_data(
        cleaned_tables=cleaned_tables,
        selected_quarter=selected_quarter,
        selected_channel=selected_channel,
    )
    selected_start_date, selected_end_date = _get_quarter_date_range(selected_quarter)
    weekly_start_date = selected_start_date
    weekly_end_date = selected_end_date
    if not monthly_sales_for_kpis.empty and "Month Start" in monthly_sales_for_kpis.columns:
        monthly_start_series = pd.to_datetime(
            monthly_sales_for_kpis["Month Start"], errors="coerce"
        ).dropna()
        if not monthly_start_series.empty:
            weekly_start_date = pd.Timestamp(monthly_start_series.min()).normalize()
            weekly_end_date = (
                pd.Timestamp(monthly_start_series.max()).normalize()
                + pd.offsets.MonthEnd(0)
            )

    # Use calendar week average over the active filtered sales span.
    # Partial weeks are handled through inclusive day-count / 7.
    total_calendar_days_in_scope = float(
        len(pd.date_range(weekly_start_date, weekly_end_date, freq="D"))
    )
    calendar_weeks_in_scope = (
        total_calendar_days_in_scope / 7.0
        if total_calendar_days_in_scope
        else 0.0
    )
    average_weekly_sale = (
        float(overall_kpi_values["displayed_gross_sale"]) / calendar_weeks_in_scope
        if calendar_weeks_in_scope
        else 0.0
    )
    # Use the business operating rule directly: 6 working days per week.
    average_daily_sale = average_weekly_sale / 6.0 if average_weekly_sale else 0.0
    peak_month_label = "N/A"
    if not monthly_sales_for_kpis.empty:
        peak_month_row = monthly_sales_for_kpis.sort_values("Gross Sale", ascending=False).iloc[0]
        peak_month_label = pd.to_datetime(peak_month_row["Month Start"]).strftime("%B")

    print(
        "DEEP_INSIGHTS_WEEKLY_SALE_DEBUG",
        {
            "filtered_start_date": str(pd.Timestamp(weekly_start_date).date()),
            "filtered_end_date": str(pd.Timestamp(weekly_end_date).date()),
            "total_gross_sale": float(overall_kpi_values["displayed_gross_sale"]),
            "total_calendar_days_in_scope": total_calendar_days_in_scope,
            "total_weeks_in_scope": calendar_weeks_in_scope,
            "final_average_weekly_sale": average_weekly_sale,
        },
    )

    deep_insights_kpi_columns = st.columns(4, gap="medium")
    deep_insights_kpi_specs = [
        ("Average Weekly Sale", _format_currency(average_weekly_sale), ""),
        ("Average Hourly Sale", _format_currency(average_hourly_sale), ""),
        ("Peak Month", peak_month_label, ""),
        ("Average Sale per Working Day", _format_currency(average_daily_sale), ""),
    ]

    for column, (label, value, support_text) in zip(
        deep_insights_kpi_columns, deep_insights_kpi_specs
    ):
        with column:
            support_display = support_text or "&nbsp;"
            st.markdown(
                (
                    '<div class="deep-hourly-kpi-card">'
                    f'<div class="deep-hourly-kpi-label">{label}</div>'
                    f'<div class="deep-hourly-kpi-value">{value}</div>'
                    f'<div class="deep-hourly-kpi-support">{support_display}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.15rem;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="deep-section-title">Month Analysis</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    chart_column, insights_column = st.columns([2.35, 1.0], gap="large")

    with chart_column:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Monthly order volume",
                    subtitle="Stacked monthly order load by channel",
                ),
                unsafe_allow_html=True,
            )
            if monthly_orders.empty:
                st.info(
                    "Monthly order volume will appear here when order data is available for the current filters."
                )
            else:
                st.markdown(_build_month_analysis_legend_html(), unsafe_allow_html=True)
                st.plotly_chart(
                    _create_monthly_order_volume_stacked_figure(monthly_orders),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    with insights_column:
        max_total_orders = (
            float(pd.to_numeric(monthly_orders["Total Orders"], errors="coerce").fillna(0).max())
            if not monthly_orders.empty
            else 0.0
        )
        heaviest_months = (
            monthly_orders.sort_values("Total Orders", ascending=False)
            .head(3)
            .reset_index(drop=True)
            if not monthly_orders.empty
            else pd.DataFrame()
        )
        lighter_months = (
            monthly_orders.sort_values("Total Orders", ascending=True)
            .head(3)
            .reset_index(drop=True)
            if not monthly_orders.empty
            else pd.DataFrame()
        )

        with st.container(border=True):
            st.markdown(
                _build_month_analysis_rank_card_html(
                    title="Heaviest months",
                    month_rows=heaviest_months,
                    tone_class="heaviest",
                    max_total_orders=max_total_orders,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                _build_month_analysis_rank_card_html(
                    title="Lighter months",
                    month_rows=lighter_months,
                    tone_class="lighter",
                    max_total_orders=max_total_orders,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="deep-section-title">Week Analysis</div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    weekday_chart_column, weekday_insights_column = st.columns([2.35, 1.0], gap="large")

    with weekday_chart_column:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Orders by day of week",
                    subtitle="Average orders per day · ranked busiest to lightest",
                ),
                unsafe_allow_html=True,
            )
            if weekday_orders.empty:
                st.info(
                    "Weekday order patterns will appear here when order data is available for the current filters."
                )
            else:
                st.markdown(_build_week_analysis_legend_html(), unsafe_allow_html=True)
                st.plotly_chart(
                    _create_weekday_order_volume_figure(weekday_orders),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    with weekday_insights_column:
        max_weekday_orders = (
            float(
                pd.to_numeric(
                    weekday_orders["Total Average Orders"], errors="coerce"
                ).fillna(0).max()
            )
            if not weekday_orders.empty
            else 0.0
        )
        peak_days = (
            weekday_orders.sort_values("Total Average Orders", ascending=False)
            .head(3)
            .reset_index(drop=True)
            if not weekday_orders.empty
            else pd.DataFrame()
        )
        quiet_days = (
            weekday_orders.sort_values("Total Average Orders", ascending=True)
            .head(3)
            .reset_index(drop=True)
            if not weekday_orders.empty
            else pd.DataFrame()
        )

        with st.container(border=True):
            st.markdown(
                _build_weekday_analysis_rank_card_html(
                    title="Peak days",
                    weekday_rows=peak_days,
                    tone_class="peak",
                    max_total_orders=max_weekday_orders,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                _build_weekday_analysis_rank_card_html(
                    title="Quiet days",
                    weekday_rows=quiet_days,
                    tone_class="quiet",
                    max_total_orders=max_weekday_orders,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)

    hourly_title_column, hourly_filter_column = st.columns([6.4, 1.6], gap="small")
    with hourly_title_column:
        st.markdown('<div class="deep-section-title">Hourly Analysis</div>', unsafe_allow_html=True)
    with hourly_filter_column:
        st.markdown(
            '<div class="deep-insights-filter-label">Channel</div>',
            unsafe_allow_html=True,
        )
        hourly_selected_channel = st.selectbox(
            "Hourly Analysis Channel",
            OVERALL_SALES_CHANNEL_OPTIONS,
            index=0,
            key="deep_insights_hourly_channel_filter",
            label_visibility="collapsed",
        )

    hourly_legend_row, _ = st.columns([6.4, 1.6], gap="small")
    with hourly_legend_row:
        st.markdown(_build_hourly_analysis_legend_html(), unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Hourly Analysis now uses exact Q4 hourly data only.
    # Sources used: POS table and the raw delivery_partner_raw_url table.
    # Approximate AOV-based hourly sales logic is intentionally excluded here.
    hourly_q4_exact = _prepare_hourly_exact_q4_data(
        cleaned_tables=cleaned_tables,
        raw_tables=raw_tables,
        selected_quarter=selected_quarter,
        selected_channel=hourly_selected_channel,
    )

    hourly_chart_columns = st.columns(2, gap="large")

    with hourly_chart_columns[0]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Average Orders per Hour",
                    subtitle=f"Exact Q4 hourly order pattern · Jan to Mar · {hourly_selected_channel}",
                ),
                unsafe_allow_html=True,
            )
            if hourly_q4_exact.empty:
                st.info(
                    "Exact Q4 hourly average orders will appear here when POS and raw delivery partner data are available for the current filters."
                )
            else:
                st.plotly_chart(
                    _create_hourly_total_orders_figure(
                        hourly_q4_exact,
                        selected_channel=hourly_selected_channel,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    with hourly_chart_columns[1]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Average Sales per Hour",
                    subtitle=f"Exact Q4 hourly sales pattern · Jan to Mar · {hourly_selected_channel}",
                ),
                unsafe_allow_html=True,
            )
            if hourly_q4_exact.empty:
                st.info(
                    "Exact Q4 hourly average sales will appear here when POS and raw delivery partner data are available for the current filters."
                )
            else:
                st.plotly_chart(
                    _create_hourly_estimated_sales_figure(
                        hourly_q4_exact,
                        selected_channel=hourly_selected_channel,
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )

    st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="deep-section-title">12pm – 4pm shift analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="deep-section-subtitle">Net revenue vs cost for this time bucket</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # This section uses exact Jan-to-Mar gross sales only from:
    # - POS table for offline/in-house sales
    # - delivery_partner_raw_url raw table for online sales
    # It calculates interactive shift profitability for the 12 PM to 4 PM bucket
    # using channel-adjusted average net sales per included hour slot, not summed quarter revenue.
    shift_analysis = _prepare_shift_analysis_q4_data(
        cleaned_tables=cleaned_tables,
        raw_tables=raw_tables,
    )
    shift_hourly = shift_analysis["hourly_revenue"]
    offline_shift_gross_revenue = float(shift_analysis["offline_gross_revenue"])
    online_shift_gross_revenue = float(shift_analysis["online_gross_revenue"])
    offline_shift_net_revenue = float(shift_analysis["offline_net_revenue"])
    online_shift_net_revenue = float(shift_analysis["online_net_revenue"])
    average_shift_net_revenue = float(shift_analysis["total_net_revenue"])
    # Shift staffing always runs through the fixed 4 PM shift end for this section.
    fixed_shift_end_time = time(16, 0)

    employee_rows_key = "deep_insights_shift_employee_rows"
    next_employee_id_key = "deep_insights_shift_next_employee_id"
    other_expenses_key = "deep_insights_shift_other_expenses"
    if employee_rows_key not in st.session_state:
        st.session_state[employee_rows_key] = [
            {
                "id": 1,
                "label": "Employee 1",
                "start_time": time(10, 0),
                "end_time": fixed_shift_end_time,
                "hourly_wage": 14.0,
            },
            {
                "id": 2,
                "label": "Employee 2",
                "start_time": time(11, 0),
                "end_time": fixed_shift_end_time,
                "hourly_wage": 14.0,
            },
        ]
    if next_employee_id_key not in st.session_state:
        st.session_state[next_employee_id_key] = 3
    if other_expenses_key not in st.session_state:
        st.session_state[other_expenses_key] = 0.0

    shift_top_left_column, shift_top_right_column = st.columns(2, gap="large")

    with shift_top_left_column:
        with st.container(border=True):
            st.markdown('<div class="shift-panel-title">Shift Setup</div>', unsafe_allow_html=True)
            shift_rows_header_column, shift_rows_action_column = st.columns(
                [2.4, 1.0], gap="small"
            )
            with shift_rows_header_column:
                st.markdown(
                    '<div class="shift-staff-header">Employee-wise staffing setup</div>',
                    unsafe_allow_html=True,
                )
            with shift_rows_action_column:
                if st.button("Add employee", key="deep_insights_shift_add_employee"):
                    next_employee_id = int(st.session_state[next_employee_id_key])
                    st.session_state[employee_rows_key] = [
                        *st.session_state[employee_rows_key],
                        {
                            "id": next_employee_id,
                            "label": f"Employee {next_employee_id}",
                            "start_time": time(12, 0),
                            "end_time": fixed_shift_end_time,
                            "hourly_wage": 14.0,
                        },
                    ]
                    st.session_state[next_employee_id_key] = next_employee_id + 1
                    st.rerun()

            employee_rows_state = list(st.session_state[employee_rows_key])
            rows_to_keep: list[dict[str, object]] = []
            employee_summary_rows: list[dict[str, object]] = []
            invalid_shift_labels: list[str] = []

            for employee_row in employee_rows_state:
                employee_id = int(employee_row["id"])
                label_key = f"deep_insights_shift_label_{employee_id}"
                start_key = f"deep_insights_shift_start_{employee_id}"
                wage_key = f"deep_insights_shift_wage_{employee_id}"

                if label_key not in st.session_state:
                    st.session_state[label_key] = str(employee_row["label"])
                if start_key not in st.session_state:
                    st.session_state[start_key] = employee_row["start_time"]
                if wage_key not in st.session_state:
                    st.session_state[wage_key] = float(employee_row["hourly_wage"])

                row_columns = st.columns([1.7, 1.1, 1.05, 0.38], gap="small")
                with row_columns[0]:
                    st.markdown('<div class="shift-input-label">Employee</div>', unsafe_allow_html=True)
                    employee_label = st.text_input(
                        f"Employee label {employee_id}",
                        key=label_key,
                        label_visibility="collapsed",
                    ).strip() or f"Employee {employee_id}"
                with row_columns[1]:
                    st.markdown('<div class="shift-input-label">Start</div>', unsafe_allow_html=True)
                    start_time_value = st.time_input(
                        f"Shift start {employee_id}",
                        key=start_key,
                        step=1800,
                        label_visibility="collapsed",
                    )
                with row_columns[2]:
                    st.markdown('<div class="shift-input-label">Wage (€)</div>', unsafe_allow_html=True)
                    hourly_wage_value = float(
                        st.number_input(
                            f"Hourly wage {employee_id}",
                            min_value=0.0,
                            step=1.0,
                            key=wage_key,
                            label_visibility="collapsed",
                        )
                    )
                with row_columns[3]:
                    st.markdown('<div class="shift-input-label">&nbsp;</div>', unsafe_allow_html=True)
                    remove_row = st.button(
                        "-",
                        key=f"deep_insights_shift_remove_{employee_id}",
                        help="Remove employee",
                    )

                if remove_row:
                    continue

                duration_hours = _calculate_shift_duration_hours(
                    start_time=start_time_value,
                    end_time=fixed_shift_end_time,
                )
                if duration_hours <= 0:
                    invalid_shift_labels.append(employee_label)
                employee_cost = duration_hours * hourly_wage_value

                rows_to_keep.append(
                    {
                        "id": employee_id,
                        "label": employee_label,
                        "start_time": start_time_value,
                        "end_time": fixed_shift_end_time,
                        "hourly_wage": hourly_wage_value,
                    }
                )
                employee_summary_rows.append(
                    {
                        "label": employee_label,
                        "start_label": _format_time_display(start_time_value),
                        "end_label": _format_time_display(fixed_shift_end_time),
                        "hours_label": _format_duration_label(duration_hours),
                        "hourly_wage": hourly_wage_value,
                        "employee_cost": employee_cost,
                        "duration_hours": duration_hours,
                    }
                )

                st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

            if len(rows_to_keep) != len(employee_rows_state):
                st.session_state[employee_rows_key] = rows_to_keep
                st.rerun()

            st.session_state[employee_rows_key] = rows_to_keep
            total_labour_hours = float(
                sum(float(employee_row["duration_hours"]) for employee_row in employee_summary_rows)
            )
            total_labour_cost = float(
                sum(float(employee_row["employee_cost"]) for employee_row in employee_summary_rows)
            )

            st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="shift-input-label">Other expenses (€)</div>',
                unsafe_allow_html=True,
            )
            other_expenses = float(
                st.number_input(
                    "Other expenses",
                    min_value=0.0,
                    step=1.0,
                    key=other_expenses_key,
                    label_visibility="collapsed",
                )
            )

            total_cost = total_labour_cost + other_expenses
            net_result = average_shift_net_revenue - total_cost
            margin_pct = (
                (net_result / average_shift_net_revenue) * 100.0
                if average_shift_net_revenue
                else 0.0
            )

            if invalid_shift_labels:
                st.warning(
                    "Start time must be earlier than 4:00 PM for: "
                    + ", ".join(invalid_shift_labels)
                )

            print(
                "SHIFT_STAFFING_DEBUG",
                {
                    "employee_rows": [
                        {
                            "label": employee_row["label"],
                            "start_time": employee_row["start_label"],
                            "end_time": employee_row["end_label"],
                            "duration_hours": float(employee_row["duration_hours"]),
                            "hourly_wage": float(employee_row["hourly_wage"]),
                            "employee_cost": float(employee_row["employee_cost"]),
                        }
                        for employee_row in employee_summary_rows
                    ],
                    "total_employees": int(len(employee_summary_rows)),
                    "total_labour_hours": total_labour_hours,
                    "total_labour_cost": total_labour_cost,
                    "other_expenses": float(other_expenses),
                    "total_cost": total_cost,
                    "average_shift_net_revenue": average_shift_net_revenue,
                    "net_result": net_result,
                    "margin_pct": margin_pct,
                },
            )

            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                _build_shift_setup_summary_html(
                    employee_rows=employee_summary_rows,
                    total_labour_hours=total_labour_hours,
                    total_labour_cost=total_labour_cost,
                    other_expenses=other_expenses,
                    total_cost=total_cost,
                ),
                unsafe_allow_html=True,
            )

    with shift_top_right_column:
        st.markdown(
            _build_shift_result_card_html(
                net_result=net_result,
                margin_pct=margin_pct,
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="12pm – 4pm · net revenue vs cost",
                    subtitle="Average shift net revenue vs shift cost",
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _create_shift_revenue_vs_cost_figure(
                    offline_gross_revenue=offline_shift_gross_revenue,
                    offline_net_revenue=offline_shift_net_revenue,
                    online_gross_revenue=online_shift_gross_revenue,
                    online_net_revenue=online_shift_net_revenue,
                    total_net_revenue=average_shift_net_revenue,
                    total_labour_cost=total_labour_cost,
                    other_expenses=other_expenses,
                    total_cost=total_cost,
                ),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

    st.markdown("<div style='height: 0.95rem;'></div>", unsafe_allow_html=True)

    shift_bottom_left_column, shift_bottom_right_column = st.columns(2, gap="large")

    with shift_bottom_left_column:
        with st.container(border=True):
            st.markdown(
                '<div class="shift-panel-title">Fetched from Data</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _build_shift_data_summary_html(
                    shift_hourly=shift_hourly,
                    offline_net_revenue=offline_shift_net_revenue,
                    online_net_revenue=online_shift_net_revenue,
                    total_net_revenue=average_shift_net_revenue,
                ),
                unsafe_allow_html=True,
            )

    with shift_bottom_right_column:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Hourly net revenue inside the shift",
                    subtitle="Average net revenue by included hour · Jan to Mar",
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _create_shift_hourly_revenue_figure(shift_hourly),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

    st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="deep-section-title">Scenario Impact Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="deep-section-subtitle">See how Jan–Mar baseline sales and orders change under a selected business up/down scenario</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # This section is a Jan-to-Mar scenario simulator.
    # The selected percentage is applied uniformly to baseline gross sale and orders.
    # Offline and online splits come only from POS + delivery_partner_raw_url raw data.
    scenario_baseline = _prepare_scenario_impact_q4_data(
        cleaned_tables=cleaned_tables,
        raw_tables=raw_tables,
    )
    scenario_selector_left, scenario_selector_center, scenario_selector_right = st.columns(
        [1.05, 4.9, 1.05],
        gap="small",
    )
    with scenario_selector_center:
        scenario_pct = int(
            st.select_slider(
                "Scenario Impact",
                options=SCENARIO_IMPACT_OPTIONS,
                value=0,
                key="deep_insights_scenario_impact_selector",
                format_func=lambda value: f"{int(value):+d}%" if int(value) else "0%",
                label_visibility="collapsed",
            )
        )

    scenario_factor = 1.0 + (float(scenario_pct) / 100.0)
    baseline_total_gross_sale = float(scenario_baseline["total_gross_sale"])
    baseline_total_orders = float(scenario_baseline["total_orders"])
    baseline_offline_gross_sale = float(scenario_baseline["offline_gross_sale"])
    baseline_online_gross_sale = float(scenario_baseline["online_gross_sale"])
    baseline_offline_orders = float(scenario_baseline["offline_orders"])
    baseline_online_orders = float(scenario_baseline["online_orders"])

    projected_total_gross_sale = baseline_total_gross_sale * scenario_factor
    projected_total_orders = baseline_total_orders * scenario_factor
    projected_offline_gross_sale = baseline_offline_gross_sale * scenario_factor
    projected_online_gross_sale = baseline_online_gross_sale * scenario_factor
    projected_offline_orders = baseline_offline_orders * scenario_factor
    projected_online_orders = baseline_online_orders * scenario_factor

    gross_sale_change = projected_total_gross_sale - baseline_total_gross_sale
    orders_change = projected_total_orders - baseline_total_orders
    offline_gross_sale_change = projected_offline_gross_sale - baseline_offline_gross_sale
    online_gross_sale_change = projected_online_gross_sale - baseline_online_gross_sale
    offline_orders_change = projected_offline_orders - baseline_offline_orders
    online_orders_change = projected_online_orders - baseline_online_orders
    scenario_tone_class = (
        "positive" if scenario_pct > 0 else "negative" if scenario_pct < 0 else "neutral"
    )

    print(
        "SCENARIO_IMPACT_PROJECTED_DEBUG",
        {
            "scenario_pct": int(scenario_pct),
            "scenario_factor": scenario_factor,
            "baseline_total_gross_sale": baseline_total_gross_sale,
            "projected_total_gross_sale": projected_total_gross_sale,
            "gross_sale_change": gross_sale_change,
            "baseline_total_orders": baseline_total_orders,
            "projected_total_orders": projected_total_orders,
            "orders_change": orders_change,
            "baseline_offline_gross_sale": baseline_offline_gross_sale,
            "projected_offline_gross_sale": projected_offline_gross_sale,
            "baseline_online_gross_sale": baseline_online_gross_sale,
            "projected_online_gross_sale": projected_online_gross_sale,
            "baseline_offline_orders": baseline_offline_orders,
            "projected_offline_orders": projected_offline_orders,
            "baseline_online_orders": baseline_online_orders,
            "projected_online_orders": projected_online_orders,
        },
    )

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

    scenario_kpi_columns = st.columns(4, gap="medium")
    scenario_kpi_specs = [
        (
            "Projected Gross Sale",
            _format_currency(projected_total_gross_sale),
            f"Jan–Mar baseline at {scenario_pct:+d}%" if scenario_pct else "Jan–Mar baseline unchanged",
            "neutral",
        ),
        (
            "Gross Sale Change",
            _format_signed_currency(gross_sale_change),
            "Projected vs Jan–Mar baseline",
            scenario_tone_class,
        ),
        (
            "Projected Orders",
            _format_whole_number(int(round(projected_total_orders))),
            f"Jan–Mar baseline at {scenario_pct:+d}%" if scenario_pct else "Jan–Mar baseline unchanged",
            "neutral",
        ),
        (
            "Orders Change",
            _format_signed_whole_number(orders_change),
            "Projected vs Jan–Mar baseline",
            scenario_tone_class,
        ),
    ]

    for scenario_column, (title, value_text, support_text, tone_class) in zip(
        scenario_kpi_columns, scenario_kpi_specs
    ):
        with scenario_column:
            st.markdown(
                _build_scenario_kpi_card_html(
                    title=title,
                    value_text=value_text,
                    support_text=support_text,
                    tone_class=tone_class,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    scenario_chart_columns = st.columns(2, gap="large")
    with scenario_chart_columns[0]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Baseline vs Projected Gross Sale",
                    subtitle="Jan–Mar baseline compared with the selected scenario",
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _create_scenario_comparison_figure(
                    baseline_value=baseline_total_gross_sale,
                    projected_value=projected_total_gross_sale,
                    scenario_pct=scenario_pct,
                    is_currency=True,
                ),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

    with scenario_chart_columns[1]:
        with st.container(border=True):
            st.markdown(
                _build_chart_card_header_html(
                    title="Baseline vs Projected Orders",
                    subtitle="Jan–Mar baseline compared with the selected scenario",
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                _create_scenario_comparison_figure(
                    baseline_value=baseline_total_orders,
                    projected_value=projected_total_orders,
                    scenario_pct=scenario_pct,
                    is_currency=False,
                ),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    scenario_split_columns = st.columns(4, gap="medium")
    scenario_split_specs = [
        (
            "Offline Gross Sale",
            _format_currency(baseline_offline_gross_sale),
            _format_currency(projected_offline_gross_sale),
            _format_signed_percentage(float(scenario_pct)),
        ),
        (
            "Online Gross Sale",
            _format_currency(baseline_online_gross_sale),
            _format_currency(projected_online_gross_sale),
            _format_signed_percentage(float(scenario_pct)),
        ),
        (
            "Offline Orders",
            _format_whole_number(int(round(baseline_offline_orders))),
            _format_whole_number(int(round(projected_offline_orders))),
            _format_signed_percentage(float(scenario_pct)),
        ),
        (
            "Online Orders",
            _format_whole_number(int(round(baseline_online_orders))),
            _format_whole_number(int(round(projected_online_orders))),
            _format_signed_percentage(float(scenario_pct)),
        ),
    ]

    for scenario_column, (title, baseline_text, projected_text, change_text) in zip(
        scenario_split_columns, scenario_split_specs
    ):
        with scenario_column:
            st.markdown(
                _build_scenario_split_card_html(
                    title=title,
                    baseline_text=baseline_text,
                    projected_text=projected_text,
                    change_text=change_text,
                    tone_class=scenario_tone_class,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        _build_scenario_summary_html(
            scenario_pct=scenario_pct,
            gross_sale_change=gross_sale_change,
            orders_change=orders_change,
            offline_sale_change=offline_gross_sale_change,
            online_sale_change=online_gross_sale_change,
            offline_order_change=offline_orders_change,
            online_order_change=online_orders_change,
        ),
        unsafe_allow_html=True,
    )


def render_forecast_planning_page(cleaned_tables: dict[str, pd.DataFrame]) -> None:
    """Render the forecast and planning shell page."""
    _apply_sage_mist_theme()
    del cleaned_tables

    st.title("Forecast & Planning")
    st.caption(
        "Forward-looking planning shell for forecast summaries, actual versus forecast tracking, and future channel or partner planning."
    )

    top_left, top_right = st.columns(2)
    with top_left:
        _show_placeholder_section(
            "Forecast Summary",
            "Reserved for top-level planning assumptions, growth expectations, and future revenue targets.",
        )
    with top_right:
        _show_placeholder_section(
            "Monthly Forecast Trend",
            "Reserved for future month-by-month forecast tracking once the forecasting layer is built.",
        )

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        _show_placeholder_section(
            "Actual vs Forecast",
            "Reserved for comparison blocks between forecasted performance and actual business results.",
        )
    with bottom_right:
        _show_placeholder_section(
            "Channel and Partner Forecasts",
            "Reserved for future planning splits across offline, online, and partner-level forecasts.",
        )
