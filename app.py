"""Main Streamlit app for the restaurant dashboard shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import streamlit as st

from src.cleaning import (
    prepare_delivery_financials,
    prepare_delivery_orders,
    prepare_pos_data,
)
from src.config import ResolvedSheetConfig, load_sheet_config_resolution
from src.dashboard_pages import (
    render_overall_sales_page,
    render_performance_analysis_page,
)
from src.data_loader import (
    SheetLoadResult,
    load_delivery_financials_with_debug,
    load_delivery_orders_with_debug,
    load_delivery_partner_raw_with_debug,
    load_pos_data_with_debug,
)


st.set_page_config(page_title="Restaurant Dashboard", layout="wide")

PAGE_OPTIONS = [
    "Overall Sales",
    "Deep Insights",
]


@dataclass
class DatasetBuildResult:
    """Internal result for one cleaned dataset."""

    cleaned_dataframe: pd.DataFrame | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.cleaned_dataframe is not None and self.error is None


@dataclass
class DashboardBackend:
    """Holds the cleaned datasets plus simple source availability state."""

    resolved_config: ResolvedSheetConfig
    datasets: dict[str, DatasetBuildResult]
    raw_sources: dict[str, pd.DataFrame]

    @property
    def cleaned_tables(self) -> dict[str, pd.DataFrame]:
        """Return the cleaned datasets in a simple dictionary."""
        return {
            dataset_name: dataset_result.cleaned_dataframe
            if dataset_result.cleaned_dataframe is not None
            else pd.DataFrame()
            for dataset_name, dataset_result in self.datasets.items()
        }

    @property
    def raw_tables(self) -> dict[str, pd.DataFrame]:
        """Return raw loaded tables that are intentionally kept uncleaned."""
        return {
            source_name: raw_dataframe.copy()
            for source_name, raw_dataframe in self.raw_sources.items()
        }

    @property
    def configured_source_count(self) -> int:
        configured_urls = [
            self.resolved_config.pos_url.value,
            self.resolved_config.delivery_financials_url.value,
            self.resolved_config.delivery_orders_url.value,
        ]
        return sum(1 for value in configured_urls if str(value).strip())

    @property
    def available_source_count(self) -> int:
        return sum(1 for dataset in self.datasets.values() if dataset.available)

    @property
    def has_no_live_data(self) -> bool:
        return self.available_source_count == 0

    @property
    def has_partial_live_data(self) -> bool:
        return (
            self.available_source_count > 0
            and self.available_source_count < self.configured_source_count
        )


def _load_and_prepare_dataset(
    sheet_url: str,
    load_function: Callable[[str], SheetLoadResult],
    prepare_function: Callable[[pd.DataFrame], pd.DataFrame],
) -> DatasetBuildResult:
    """
    Load and clean one dataset.

    The dashboard keeps this work in the background so the main interface
    can stay focused on the business view instead of the validation view.
    """
    if not str(sheet_url).strip():
        return DatasetBuildResult(
            cleaned_dataframe=None,
            error="Source URL is not configured.",
        )

    load_result = load_function(sheet_url)
    if not load_result.succeeded or load_result.dataframe is None:
        return DatasetBuildResult(
            cleaned_dataframe=None,
            error=load_result.fetch_debug.exception_message
            or "Source data could not be loaded.",
        )

    try:
        cleaned_dataframe = prepare_function(load_result.dataframe.copy())
    except Exception as error:
        return DatasetBuildResult(
            cleaned_dataframe=None,
            error=f"Data preparation failed: {error}",
        )

    return DatasetBuildResult(cleaned_dataframe=cleaned_dataframe)


def _load_raw_dataset(
    source_name: str,
    sheet_url: str,
    load_function: Callable[[str], SheetLoadResult],
) -> pd.DataFrame:
    """
    Load one raw dataset without cleaning or transformation.

    This keeps published CSV content available for future use while avoiding
    any dependency on it in the current dashboard calculations.
    """
    if not str(sheet_url).strip():
        print(
            "RAW_SOURCE_LOAD",
            {
                "source": source_name,
                "succeeded": False,
                "reason": "Source URL is not configured.",
            },
        )
        return pd.DataFrame()

    load_result = load_function(sheet_url)
    if not load_result.succeeded or load_result.dataframe is None:
        print(
            "RAW_SOURCE_LOAD",
            {
                "source": source_name,
                "succeeded": False,
                "reason": load_result.fetch_debug.exception_message
                or "Source data could not be loaded.",
            },
        )
        return pd.DataFrame()

    raw_dataframe = load_result.dataframe.copy()
    print(
        "RAW_SOURCE_LOAD",
        {
            "source": source_name,
            "succeeded": True,
            "rows": int(len(raw_dataframe.index)),
            "columns": int(len(raw_dataframe.columns)),
            "column_names": list(raw_dataframe.columns),
        },
    )
    return raw_dataframe


def build_dashboard_backend() -> DashboardBackend:
    """Run the full data pipeline silently and return cleaned datasets."""
    resolved_config = load_sheet_config_resolution()

    dataset_results = {
        "pos": _load_and_prepare_dataset(
            sheet_url=resolved_config.pos_url.value,
            load_function=load_pos_data_with_debug,
            prepare_function=prepare_pos_data,
        ),
        "delivery_financials": _load_and_prepare_dataset(
            sheet_url=resolved_config.delivery_financials_url.value,
            load_function=load_delivery_financials_with_debug,
            prepare_function=prepare_delivery_financials,
        ),
        "delivery_orders": _load_and_prepare_dataset(
            sheet_url=resolved_config.delivery_orders_url.value,
            load_function=load_delivery_orders_with_debug,
            prepare_function=prepare_delivery_orders,
        ),
    }

    raw_source_results = {
        # Raw published Google Sheet source for delivery partner data from Jan to Mar.
        "delivery_partner_raw": _load_raw_dataset(
            source_name="delivery_partner_raw",
            sheet_url=resolved_config.delivery_partner_raw_url.value,
            load_function=load_delivery_partner_raw_with_debug,
        ),
    }

    return DashboardBackend(
        resolved_config=resolved_config,
        datasets=dataset_results,
        raw_sources=raw_source_results,
    )


def render_sidebar_navigation() -> str:
    """Render the main dashboard navigation."""
    st.sidebar.title("IBC Dashboard")
    return st.sidebar.radio(
        "Navigation",
        PAGE_OPTIONS,
        index=0,
    )


def render_dashboard_banner(backend: DashboardBackend) -> None:
    """Show only simple business-friendly status messaging."""
    if backend.has_no_live_data:
        st.warning(
            "Live source data is not available right now. The dashboard layout is ready and sections will populate when the feeds load."
        )
        return

    if backend.has_partial_live_data:
        st.warning(
            "Some data sources are currently unavailable. The dashboard is showing the data that loaded successfully."
        )
        return


def render_selected_page(selected_page: str, backend: DashboardBackend) -> None:
    """Dispatch rendering to the selected dashboard page."""
    if selected_page == "Overall Sales":
        render_overall_sales_page(backend.cleaned_tables)
        return

    render_performance_analysis_page(backend.cleaned_tables, backend.raw_tables)


def main() -> None:
    """Render the main dashboard app."""
    selected_page = render_sidebar_navigation()

    with st.spinner("Loading dashboard data..."):
        backend = build_dashboard_backend()

    render_dashboard_banner(backend)
    render_selected_page(selected_page, backend)


if __name__ == "__main__":
    main()
