#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Ads Agency Dashboard Pro v7.0

- CPC calculation from Spend/Clicks
- ASIN negative handling
- Supports 7/14/30-day sales & orders columns (big-date reports)
- TCoAS (Total Cost of Advertising Sales) support
- Amazon BULK upload–ready bid optimization export
- S.No starts from 1 in all tables
- Premium dark-glass UI with clear colours (red for negatives)
- Placement-wise recommendations when placement data is present
"""

import io
import re
import traceback
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Ads Dashboard Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def load_custom_css():
    css = """
    <style>
    .main {
        padding-top: 0.5rem;
        background: radial-gradient(circle at top, #020617 0, #020617 40%, #000000 100%);
    }

    /* Header */
    .agency-header {
        background:
            radial-gradient(circle at top left, #a855f7 0, #1e293b 35%, #020617 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        color: #e5e7eb;
        box-shadow: 0 18px 50px rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.5);
    }
    .agency-header h1 {
        margin: 0;
        font-size: 1.5rem;
    }
    .agency-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        color: #e0e7ff;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background:
            radial-gradient(circle at top left, #0f172a 0, #020617 55%, #020617 100%);
        border-radius: 16px;
        padding: 1.1rem 0.9rem;
        border: 1px solid #1f2937;
        box-shadow: 0 16px 40px rgba(15,23,42,0.9);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #e5e7eb !important;   /* brighter for readability */
        white-space: normal !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        color: #f9fafb !important;
        white-space: normal !important;
        word-break: break-word !important;
    }

    /* Target number inputs (sidebar) */
    div[data-testid="stNumberInput"] > label {
        font-size: 0.75rem;
        line-height: 1.15;
        color: #e5e7eb;
    }
    div[data-testid="stNumberInput"] > div {
        background: #020617;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    div[data-testid="stNumberInput"] input {
        text-align: center;
        padding: 0.3rem 0.4rem;
        font-size: 0.85rem;
        color: #f9fafb;
    }

    /* Info / status boxes */
    .info-box {
        background: radial-gradient(circle at top left, rgba(59,130,246,0.35), rgba(15,23,42,0.98));
        border-left: 4px solid #3b82f6;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
    }
    .success-box {
        background: radial-gradient(circle at top left, rgba(34,197,94,0.35), rgba(6,78,59,0.98));
        border-left: 4px solid #22c55e;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
        color: #dcfce7;
    }
    .warning-box {
        background: radial-gradient(circle at top left, rgba(250,204,21,0.35), rgba(77,54,10,0.98));
        border-left: 4px solid #eab308;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
        color: #fef9c3;
    }
    .danger-box {
        background: radial-gradient(circle at top left, rgba(248,113,113,0.40), rgba(127,29,29,0.99));
        border-left: 4px solid #ef4444;   /* strong red for negatives */
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
        color: #fee2e2;
    }

    .purple-box {
        background: radial-gradient(circle at top left, rgba(129,140,248,0.35), rgba(30,64,175,0.98));
        border-left: 4px solid #8b5cf6;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
        color: #e0e7ff;
    }
    .cyan-box {
        background: radial-gradient(circle at top left, rgba(6,182,212,0.35), rgba(8,47,73,0.98));
        border-left: 4px solid #06b6d4;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
        color: #cffafe;
    }

    /* DataFrames rounded corners */
    .blank > div[data-testid="stDataFrame"] table {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value) or value == "" or value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        val_str = (
            str(value)
            .replace("₹", "")
            .replace("$", "")
            .replace(",", "")
            .replace("%", "")
            .strip()
        )
        return float(val_str) if val_str else default
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value) or value == "" or value is None:
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return int(
            float(
                str(value)
                .replace(",", "")
                .replace("₹", "")
                .replace("$", "")
                .strip()
            )
        )
    except Exception:
        return default


def safe_str(value, default: str = "N/A") -> str:
    try:
        if pd.isna(value) or value == "" or value is None:
            return default
        return str(value).strip()
    except Exception:
        return default


def format_currency(value) -> str:
    try:
        val = safe_float(value, 0)
        if val >= 10_000_000:
            return f"₹{val/10_000_000:.2f}Cr"
        if val >= 100_000:
            return f"₹{val/100_000:.2f}L"
        return f"₹{val:,.2f}"
    except Exception:
        return "₹0.00"


def format_number(value) -> str:
    try:
        val = safe_int(value, 0)
        if val >= 10_000_000:
            return f"{val/10_000_000:.2f}Cr"
        if val >= 100_000:
            return f"{val/100_000:.2f}L"
        if val >= 1_000:
            return f"{val:,}"
        return str(val)
    except Exception:
        return "0"


def is_asin(value) -> bool:
    if not value or pd.isna(value):
        return False
    val_str = str(value).strip().upper()
    return bool(re.match(r"^[B][0-9A-Z]{9,}$", val_str))


def get_negative_type(value) -> str:
    return "PRODUCT" if is_asin(value) else "KEYWORD"


def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add S.No starting from 1 for nicer tables."""
    if df is None or len(df) == 0:
        return df
    df = df.reset_index(drop=True).copy()
    df.insert(0, "S.No", df.index + 1)
    return df

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
class ClientData:
    def __init__(self, name, industry: str = "E-commerce", budget: float = 50_000):
        self.name = name
        self.industry = industry
        self.monthly_budget = budget
        self.analyzer: "CompleteAnalyzer | None" = None
        self.added_date = datetime.now()
        self.contact_email: str = ""
        self.target_acos: float | None = None
        self.target_roas: float | None = None
        self.target_cpa: float | None = None
        self.target_tcoas: float | None = None

class CompleteAnalyzer:
    REQUIRED_COLUMNS = ["Customer Search Term", "Campaign Name", "Spend", "Clicks"]

    def __init__(
        self,
        df: pd.DataFrame,
        client_name: str,
        target_acos: float | None = None,
        target_roas: float | None = None,
        target_cpa: float | None = None,
        target_tcoas: float | None = None,
    ):
        self.client_name = client_name
        self.target_acos = target_acos
        self.target_roas = target_roas
        self.target_cpa = target_cpa
        self.target_tcoas = target_tcoas
        self.df: pd.DataFrame | None = None
        self.raw_df: pd.DataFrame | None = None
        self.error: str | None = None
        self.column_mapping: Dict[str, str] = {}
        try:
            self.raw_df = df.copy(deep=True)
            self.df = self._validate_and_prepare_data(df.copy(deep=True))
        except Exception as e:
            self.error = str(e)
            raise ValueError(f"Validation failed: {e}")

    # ------------------------------------------------------------------ #
    # Data prep
    # ------------------------------------------------------------------ #
    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame")

        df.columns = df.columns.str.strip()

        # Mapping extended for 7 / 14 / 30 day reports so big-date files work.[file:70][file:1]
        mapping = {
            # search term
            "customer search term": "Customer Search Term",
            "search term": "Customer Search Term",
            "keyword": "Customer Search Term",
            "searchterm": "Customer Search Term",
            "customer_search_term": "Customer Search Term",
            "search terms": "Customer Search Term",
            # campaign / ad group
            "campaign": "Campaign Name",
            "campaign name": "Campaign Name",
            "campaign_name": "Campaign Name",
            "ad group": "Ad Group Name",
            "ad group name": "Ad Group Name",
            "adgroup": "Ad Group Name",
            "ad_group_name": "Ad Group Name",
            # match type
            "match type": "Match Type",
            "matchtype": "Match Type",
            "match_type": "Match Type",
            # SALES – 7/14/30 day variants
            "7 day total sales": "Sales",
            "7 day total sales (₹)": "Sales",
            "7 day total sales ($)": "Sales",
            "7 day sales": "Sales",
            "7 day total revenue": "Sales",
            "7 day total sales ": "Sales",
            "7 day total sales(₹)": "Sales",
            "14 day total sales": "Sales",
            "14 day total sales (₹)": "Sales",
            "14 day total sales ($)": "Sales",
            "30 day total sales": "Sales",
            "30 day total sales (₹)": "Sales",
            "30 day total sales ($)": "Sales",
            "total sales": "Sales",
            "sales": "Sales",
            "revenue": "Sales",
            # ORDERS – 7/14/30 day variants
            "7 day total orders": "Orders",
            "7 day total orders (#)": "Orders",
            "7 day orders": "Orders",
            "7 day total units": "Orders",
            "7 day ordered units": "Orders",
            "14 day total orders": "Orders",
            "14 day total orders (#)": "Orders",
            "30 day total orders": "Orders",
            "30 day total orders (#)": "Orders",
            "total orders": "Orders",
            "orders": "Orders",
            "units": "Orders",
            # Spend/Cost
            "cost": "Spend",
            "spend": "Spend",
            "ad spend": "Spend",
            "spend (₹)": "Spend",
            "spend ($)": "Spend",
            # Impressions & clicks
            "impressions": "Impressions",
            "imps": "Impressions",
            "clicks": "Clicks",
            # CPC
            "cpc": "CPC",
            "cost per click": "CPC",
            "avg cpc": "CPC",
            "average cpc": "CPC",
        }

        df.columns = df.columns.str.lower().str.strip()
        for old, new in mapping.items():
            key = old.lower().strip()
            if key in df.columns:
                df.rename(columns={key: new}, inplace=True)
                self.column_mapping[old] = new

        df.columns = [c.title() for c in df.columns]

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Available: {list(df.columns)}"
            )

        optional_numeric = ["Sales", "Orders", "Impressions", "Cpc"]
        optional_text = ["Ad Group Name", "Match Type"]
        for col in optional_numeric:
            if col not in df.columns:
                df[col] = 0
        for col in optional_text:
            if col not in df.columns:
                df[col] = "N/A"

        if "Cpc" in df.columns and "CPC" not in df.columns:
            df["CPC"] = df["Cpc"]

        numeric_cols = ["Spend", "Sales", "Clicks", "Impressions", "Orders", "CPC"]
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace("₹", "")
                        .str.replace("$", "")
                        .str.replace(",", "")
                        .str.replace("%", "")
                        .str.replace("(", "-")
                        .str.replace(")", "")
                        .str.strip()
                    )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # CPC from Spend/Clicks when missing.[file:70]
        df["Cpc_Calculated"] = df.apply(
            lambda x: safe_float(x.get("Spend", 0))
            / safe_float(x.get("Clicks", 1))
            if safe_float(x.get("Clicks", 0)) > 0
            else 0,
            axis=1,
        )
        df["CPC"] = df.apply(
            lambda x: safe_float(x.get("CPC", 0))
            if safe_float(x.get("CPC", 0)) > 0
            else safe_float(x.get("Cpc_Calculated", 0)),
            axis=1,
        )

        # keep active rows
        df = df[(df["Spend"] > 0) | (df["Clicks"] > 0)].copy()
        if len(df) == 0:
            raise ValueError("No valid data after filtering")

        # derived metrics
        df["Profit"] = df["Sales"] - df["Spend"]
        df["Wastage"] = df.apply(
            lambda x: safe_float(x.get("Spend", 0))
            if safe_float(x.get("Sales", 0)) == 0
            else 0,
            axis=1,
        )
        df["CVR"] = df.apply(
            lambda x: (
                safe_float(x.get("Orders", 0)) / safe_float(x.get("Clicks", 1)) * 100
            )
            if safe_float(x.get("Clicks", 0)) > 0
            else 0,
            axis=1,
        )
        df["ROAS"] = df.apply(
            lambda x: (
                safe_float(x.get("Sales", 0)) / safe_float(x.get("Spend", 1))
            )
            if safe_float(x.get("Spend", 0)) > 0
            else 0,
            axis=1,
        )
        df["ACOS"] = df.apply(
            lambda x: (
                safe_float(x.get("Spend", 0)) / safe_float(x.get("Sales", 1)) * 100
            )
            if safe_float(x.get("Sales", 0)) > 0
            else 0,
            axis=1,
        )
        df["CTR"] = df.apply(
            lambda x: (
                safe_float(x.get("Clicks", 0))
                / safe_float(x.get("Impressions", 1))
                * 100
            )
            if safe_float(x.get("Impressions", 0)) > 0
            else 0,
            axis=1,
        )
        df["CPA"] = df.apply(
            lambda x: (
                safe_float(x.get("Spend", 0)) / safe_float(x.get("Orders", 1))
            )
            if safe_float(x.get("Orders", 0)) > 0
            else 0,
            axis=1,
        )
        df["TCoAS"] = df["ACOS"]
        df["Negative_Type"] = df["Customer Search Term"].apply(get_negative_type)
        df["Client"] = self.client_name
        df["Processed_Date"] = datetime.now()

        return df

    # ------------------------------------------------------------------ #
    # Summary & health
    # ------------------------------------------------------------------ #
    def _empty_summary(self) -> Dict[str, float]:
        return {
            k: 0
            for k in [
                "total_spend",
                "total_sales",
                "total_profit",
                "total_orders",
                "total_clicks",
                "total_impressions",
                "total_wastage",
                "roas",
                "acos",
                "tcoas",
                "avg_cpc",
                "avg_ctr",
                "avg_cvr",
                "avg_cpa",
                "conversion_rate",
                "keywords_count",
                "campaigns_count",
                "ad_groups_count",
            ]
        }

    def get_client_summary(self) -> Dict[str, float]:
        try:
            if self.df is None or len(self.df) == 0:
                return self._empty_summary()

            ts = safe_float(self.df["Spend"].sum())
            tsa = safe_float(self.df["Sales"].sum())
            to = safe_int(self.df["Orders"].sum())
            tc = safe_int(self.df["Clicks"].sum())
            ti = safe_int(self.df["Impressions"].sum())
            tw = safe_float(self.df["Wastage"].sum())
            tp = safe_float(self.df["Profit"].sum())

            avg_cpc = 0.0
            if "CPC" in self.df.columns:
                cpc_values = self.df[self.df["CPC"] > 0]["CPC"]
                if len(cpc_values) > 0:
                    avg_cpc = safe_float(cpc_values.mean())
            if avg_cpc == 0 and tc > 0:
                avg_cpc = ts / tc

            avg_ctr = safe_float(self.df["CTR"].mean())
            avg_cvr = safe_float(self.df["CVR"].mean())
            avg_cpa = ts / to if to > 0 else 0
            avg_acos = ts / tsa * 100 if tsa > 0 else 0
            avg_roas = tsa / ts if ts > 0 else 0

            return {
                "total_spend": ts,
                "total_sales": tsa,
                "total_profit": tp,
                "total_orders": to,
                "total_clicks": tc,
                "total_impressions": ti,
                "total_wastage": tw,
                "roas": avg_roas,
                "acos": avg_acos,
                "tcoas": avg_acos,
                "avg_cpc": avg_cpc,
                "avg_ctr": avg_ctr,
                "avg_cvr": avg_cvr,
                "avg_cpa": avg_cpa,
                "conversion_rate": to / tc * 100 if tc > 0 else 0,
                "keywords_count": len(self.df),
                "campaigns_count": safe_int(self.df["Campaign Name"].nunique()),
                "ad_groups_count": safe_int(
                    self.df["Ad Group Name"].nunique()
                )
                if "Ad Group Name" in self.df.columns
                else 0,
            }
        except Exception as e:
            st.error(f"Summary error: {e}")
            return self._empty_summary()

    def get_health_score(self) -> int:
        try:
            s = self.get_client_summary()
            score = 0

            r = s["roas"]
            if r >= 4.0:
                score += 40
            elif r >= 3.0:
                score += 35
            elif r >= 2.5:
                score += 30
            elif r >= 2.0:
                score += 25
            elif r >= 1.5:
                score += 15
            elif r > 0:
                score += 5

            wp = s["total_wastage"] / s["total_spend"] * 100 if s["total_spend"] > 0 else 0
            if wp <= 5:
                score += 25
            elif wp <= 15:
                score += 20
            elif wp <= 25:
                score += 15
            elif wp <= 35:
                score += 10
            else:
                score += 5

            ctr = s["avg_ctr"]
            if ctr >= 5:
                score += 20
            elif ctr >= 3:
                score += 15
            elif ctr >= 1.5:
                score += 10
            elif ctr >= 0.5:
                score += 5

            cvr = s["avg_cvr"]
            if cvr >= 10:
                score += 15
            elif cvr >= 5:
                score += 10
            elif cvr >= 2:
                score += 5

            return int(min(score, 100))
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    # Insights & placement recommendations
    # ------------------------------------------------------------------ #
    def get_performance_insights(self) -> Dict:
        s = self.get_client_summary()
        insights = {
            "ctr_insights": [],
            "cvr_insights": [],
            "roas_insights": [],
            "acos_insights": [],
            "cpa_insights": [],
            "tcoas_insights": [],
            "content_suggestions": [],
        }

        try:
            avg_ctr = s["avg_ctr"]
            if avg_ctr < 0.3:
                insights["ctr_insights"].append(
                    {
                        "level": "critical",
                        "metric": "CTR",
                        "value": f"{avg_ctr:.2f}%",
                        "issue": "Extremely low CTR",
                        "action": "Revamp creatives and targeting",
                    }
                )
                insights["content_suggestions"].extend(
                    [
                        'Add power words: "Best", "Top-Rated", "Premium"',
                        'Show pricing: "Under ₹999", "Free Shipping"',
                        "Use high‑res lifestyle images",
                    ]
                )
            elif avg_ctr < 0.8:
                insights["ctr_insights"].append(
                    {
                        "level": "warning",
                        "metric": "CTR",
                        "value": f"{avg_ctr:.2f}%",
                        "issue": "CTR below normal (1–2%)",
                        "action": "Test new titles and main images",
                    }
                )
            elif avg_ctr >= 2.0:
                insights["ctr_insights"].append(
                    {
                        "level": "success",
                        "metric": "CTR",
                        "value": f"{avg_ctr:.2f}%",
                        "issue": "Strong CTR",
                        "action": "Scale winning campaigns",
                    }
                )

            avg_cvr = s["avg_cvr"]
            if avg_cvr < 1.0:
                insights["cvr_insights"].append(
                    {
                        "level": "critical",
                        "metric": "CVR",
                        "value": f"{avg_cvr:.2f}%",
                        "issue": "Very low conversion rate",
                        "action": "Fix pricing, listing, or audience",
                    }
                )
            elif avg_cvr < 3.0:
                insights["cvr_insights"].append(
                    {
                        "level": "warning",
                        "metric": "CVR",
                        "value": f"{avg_cvr:.2f}%",
                        "issue": "Below average conversion",
                        "action": "Improve images and A+ content",
                    }
                )

            roas = s["roas"]
            if roas < 1.0:
                insights["roas_insights"].append(
                    {
                        "level": "critical",
                        "metric": "ROAS",
                        "value": f"{roas:.2f}x",
                        "issue": "Losing money",
                        "action": "Pause or heavily cut bids",
                    }
                )
            elif roas < 2.0:
                insights["roas_insights"].append(
                    {
                        "level": "warning",
                        "metric": "ROAS",
                        "value": f"{roas:.2f}x",
                        "issue": "Weak profit",
                        "action": "Reduce bids and add negatives",
                    }
                )
            elif roas >= 3.0:
                insights["roas_insights"].append(
                    {
                        "level": "success",
                        "metric": "ROAS",
                        "value": f"{roas:.2f}x",
                        "issue": "Good profitability",
                        "action": "Scale winners",
                    }
                )

            acos = s["acos"]
            target_acos = self.target_acos or 30
            if acos > target_acos * 1.5:
                insights["acos_insights"].append(
                    {
                        "level": "critical",
                        "metric": "ACOS",
                        "value": f"{acos:.1f}%",
                        "issue": "ACOS far above target",
                        "action": "Cut bids, pause waste, tighten targeting",
                    }
                )
            elif acos > target_acos:
                insights["acos_insights"].append(
                    {
                        "level": "warning",
                        "metric": "ACOS",
                        "value": f"{acos:.1f}%",
                        "issue": "ACOS slightly above target",
                        "action": "Gradually reduce bids on poor performers",
                    }
                )

            avg_cpa = s["avg_cpa"]
            if self.target_cpa and avg_cpa > self.target_cpa:
                insights["cpa_insights"].append(
                    {
                        "level": "warning",
                        "metric": "CPA",
                        "value": format_currency(avg_cpa),
                        "issue": f"CPA above target {format_currency(self.target_cpa)}",
                        "action": "Lower bids or improve CVR",
                    }
                )

            tcoas = s["tcoas"]
            if self.target_tcoas and tcoas > self.target_tcoas:
                insights["tcoas_insights"].append(
                    {
                        "level": "warning",
                        "metric": "TCoAS",
                        "value": f"{tcoas:.1f}%",
                        "issue": f"TCoAS above target {self.target_tcoas:.1f}%",
                        "action": "Increase organic sales or trim ad spend",
                    }
                )

            return insights
        except Exception as e:
            st.error(f"Insights error: {e}")
            return insights

    def get_placement_recommendations(self) -> Dict:
        """
        Placement-wise summary & simple recommendations.

        Works if the uploaded report has a column containing 'placement'
        in its name (e.g., 'Placement', 'Placement Type'). Otherwise, it
        returns an info message so the UI can explain the limitation.
        """
        res = {"available": False, "table": pd.DataFrame(), "recs": [], "message": ""}
        try:
            if self.df is None or len(self.df) == 0:
                res["message"] = "No data for placements."
                return res

            placement_cols = [c for c in self.df.columns if "placement" in c.lower()]
            if not placement_cols:
                res["message"] = (
                    "Current report has no placement column. "
                    "Use a placement/campaign report for placement-wise tips."
                )
                return res

            col = placement_cols[0]
            dfp = self.df.copy()
            dfp[col] = dfp[col].fillna("UNKNOWN").astype(str).str.strip()

            g = dfp.groupby(col).agg(
                {
                    "Spend": "sum",
                    "Sales": "sum",
                    "Orders": "sum",
                    "Clicks": "sum",
                    "Impressions": "sum",
                }
            )
            g["ROAS"] = g.apply(
                lambda x: safe_float(x["Sales"]) / safe_float(x["Spend"])
                if safe_float(x["Spend"]) > 0
                else 0,
                axis=1,
            )
            g["ACOS"] = g.apply(
                lambda x: safe_float(x["Spend"]) / safe_float(x["Sales"]) * 100
                if safe_float(x["Sales"]) > 0
                else 0,
                axis=1,
            )
            g["CVR"] = g.apply(
                lambda x: safe_float(x["Orders"]) / safe_float(x["Clicks"]) * 100
                if safe_float(x["Clicks"]) > 0
                else 0,
                axis=1,
            )
            g["CTR"] = g.apply(
                lambda x: safe_float(x["Clicks"]) / safe_float(x["Impressions"]) * 100
                if safe_float(x["Impressions"]) > 0
                else 0,
                axis=1,
            )

            g_reset = g.reset_index().rename(columns={col: "Placement"})
            res["table"] = g_reset
            res["available"] = True

            ta = self.target_acos or 30
            tr = self.target_roas or 3.0

            recs = []
            for _, r in g_reset.iterrows():
                name = safe_str(r["Placement"])
                ro = safe_float(r["ROAS"])
                ac = safe_float(r["ACOS"])
                sp = safe_float(r["Spend"])

                if sp < 1:
                    continue

                if ro >= tr or ac <= ta:
                    recs.append(
                        {
                            "Placement": name,
                            "Action": "INCREASE",
                            "Recommendation": f"Strong performance (ROAS {ro:.2f}x, ACOS {ac:.1f}%). Test +10–20% placement bid.",
                        }
                    )
                elif ro < 1.0 or ac > ta * 1.5:
                    recs.append(
                        {
                            "Placement": name,
                            "Action": "REDUCE",
                            "Recommendation": f"Weak performance (ROAS {ro:.2f}x, ACOS {ac:.1f}%). Reduce placement bid by 15–30% or pause.",
                        }
                    )
                else:
                    recs.append(
                        {
                            "Placement": name,
                            "Action": "HOLD",
                            "Recommendation": f"Average performance (ROAS {ro:.2f}x, ACOS {ac:.1f}%). Keep bids stable and monitor.",
                        }
                    )

            res["recs"] = recs
            return res
        except Exception as e:
            res["message"] = f"Error while building placement view: {e}"
            return res

    # ------------------------------------------------------------------ #
    # Classification & bids
    # ------------------------------------------------------------------ #
    def classify_keywords_improved(self) -> Dict[str, List[Dict]]:
        cats = {
            "high_potential": [],
            "low_potential": [],
            "wastage": [],
            "opportunities": [],
            "future_watch": [],
        }
        try:
            if self.df is None or len(self.df) == 0:
                return cats

            for _, r in self.df.iterrows():
                try:
                    sp = safe_float(r.get("Spend", 0))
                    sa = safe_float(r.get("Sales", 0))
                    ro = safe_float(r.get("ROAS", 0))
                    o = safe_int(r.get("Orders", 0))
                    c = safe_int(r.get("Clicks", 0))
                    cv = safe_float(r.get("CVR", 0))
                    kw = safe_str(r.get("Customer Search Term"))
                    camp = safe_str(r.get("Campaign Name"))
                    mt = safe_str(r.get("Match Type"))
                    cpc = safe_float(r.get("CPC", 0))

                    if sp <= 0 and c <= 0:
                        continue

                    kd = {
                        "Keyword": kw,
                        "Spend": format_currency(sp),
                        "Sales": format_currency(sa),
                        "ROAS": f"{ro:.2f}x",
                        "Orders": o,
                        "Clicks": c,
                        "CVR": f"{cv:.2f}%",
                        "CPC": format_currency(cpc),
                        "Campaign": camp,
                        "Match Type": mt,
                        "Reason": "",
                    }

                    if ro >= 2.5 and o >= 1 and sp >= 20:
                        kd["Reason"] = f"Champion! ROAS {ro:.2f}x, {o} orders"
                        cats["high_potential"].append(kd)
                    elif sp >= 50 and sa == 0 and c >= 3:
                        kd["Reason"] = f"₹{sp:.0f} spent, ZERO sales - PAUSE"
                        cats["wastage"].append(kd)
                    elif sp >= 30 and ro < 1.0 and c >= 5:
                        kd["Reason"] = f"Poor ROAS {ro:.2f}x - reduce 30%"
                        cats["low_potential"].append(kd)
                    elif sp >= 20 and 1.5 <= ro < 2.5 and c >= 3:
                        kd["Reason"] = f"Good potential ROAS {ro:.2f}x - test +10-15%"
                        cats["opportunities"].append(kd)
                    elif c >= 3 and sp < 50 and sa == 0:
                        kd["Reason"] = f"{c} clicks, ₹{sp:.0f} - gather more data"
                        cats["future_watch"].append(kd)
                except Exception:
                    continue

            return cats
        except Exception as e:
            st.error(f"Classify error: {e}")
            return cats

    def get_bid_suggestions_improved(self) -> List[Dict]:
        sug: List[Dict] = []
        try:
            if self.df is None:
                return sug

            ta = self.target_acos or 30.0
            tr = self.target_roas or 3.0

            for _, r in self.df.iterrows():
                try:
                    sp = safe_float(r.get("Spend", 0))
                    sa = safe_float(r.get("Sales", 0))
                    ro = safe_float(r.get("ROAS", 0))
                    o = safe_int(r.get("Orders", 0))
                    c = safe_int(r.get("Clicks", 0))
                    cv = safe_float(r.get("CVR", 0))
                    cpc = safe_float(r.get("CPC", 0))

                    if sp < 20 or c < 3 or cpc <= 0:
                        continue

                    s = {
                        "Keyword": safe_str(r.get("Customer Search Term")),
                        "Campaign": safe_str(r.get("Campaign Name")),
                        "Ad Group": safe_str(r.get("Ad Group Name")),
                        "Match Type": safe_str(r.get("Match Type")),
                        "Current CPC": format_currency(cpc),
                        "Spend": format_currency(sp),
                        "Sales": format_currency(sa),
                        "ROAS": f"{ro:.2f}x",
                        "CVR": f"{cv:.2f}%",
                        "Orders": o,
                        "Action": "",
                        "Suggested Bid": "",
                        "Change (%)": 0,
                        "Reason": "",
                    }

                    ac = sp / sa * 100 if sa > 0 else 999

                    if ro >= 3.0 and cv >= 2.0 and o >= 2:
                        nb = cpc * 1.25
                        s.update(
                            {
                                "Action": "INCREASE",
                                "Suggested Bid": format_currency(nb),
                                "Change (%)": 25,
                                "Reason": f"Champion keyword! ROAS {ro:.2f}x",
                            }
                        )
                        sug.append(s)
                    elif ro >= tr and cv >= 1.0 and o >= 1:
                        nb = cpc * 1.15
                        s.update(
                            {
                                "Action": "INCREASE",
                                "Suggested Bid": format_currency(nb),
                                "Change (%)": 15,
                                "Reason": "Above target ROAS",
                            }
                        )
                        sug.append(s)
                    elif sa == 0 and sp >= 50:
                        s.update(
                            {
                                "Action": "PAUSE",
                                "Suggested Bid": "₹0.00",
                                "Change (%)": -100,
                                "Reason": f"₹{sp:.0f} wasted, no sales",
                            }
                        )
                        sug.append(s)
                    elif ro < 1.5 and sp >= 30:
                        nb = cpc * 0.7
                        s.update(
                            {
                                "Action": "REDUCE",
                                "Suggested Bid": format_currency(nb),
                                "Change (%)": -30,
                                "Reason": f"Poor ROAS {ro:.2f}x",
                            }
                        )
                        sug.append(s)
                    elif ac > ta and sp >= 30:
                        red = min(30, (ac - ta) / ta * 100)
                        nb = cpc * (1 - red / 100)
                        s.update(
                            {
                                "Action": "REDUCE",
                                "Suggested Bid": format_currency(nb),
                                "Change (%)": int(-red),
                                "Reason": f"ACOS {ac:.1f}% above target {ta:.1f}%",
                            }
                        )
                        sug.append(s)
                except Exception:
                    continue

            return sorted(
                sug,
                key=lambda x: safe_float(
                    x["Spend"]
                    .replace("₹", "")
                    .replace(",", "")
                    .replace("L", "")
                    .replace("Cr", "")
                ),
                reverse=True,
            )
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # Amazon BULK file builder for bids
    # ------------------------------------------------------------------ #
    def build_amazon_bulk_from_bids(self, bid_suggestions: List[Dict]) -> pd.DataFrame:
        rows: List[Dict] = []
        for s in bid_suggestions:
            try:
                campaign = safe_str(s.get("Campaign"))
                ad_group = safe_str(s.get("Ad Group"))
                keyword = safe_str(s.get("Keyword"))
                match_type = safe_str(s.get("Match Type"))
                action = safe_str(s.get("Action")).upper()
                bid_val = safe_float(s.get("Suggested Bid", 0))

                if not campaign or not ad_group or not keyword:
                    continue

                row = {
                    "Record Type": "Keyword",
                    "Campaign Name": campaign,
                    "Ad Group Name": ad_group,
                    "Keyword or Product Targeting": keyword,
                    "Match Type": match_type.title() if match_type else "",
                    "Bid": bid_val if bid_val > 0 else "",
                    "State": "enabled",
                    "Operation": "update",
                }

                if action == "PAUSE":
                    row["State"] = "paused"
                    row["Bid"] = ""

                rows.append(row)
            except Exception:
                continue

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    def generate_client_report(self) -> str:
        try:
            s = self.get_client_summary()
            h = self.get_health_score()
            c = self.classify_keywords_improved()
            hs = "EXCELLENT" if h >= 70 else "GOOD" if h >= 50 else "NEEDS ATTENTION"

            tas = f"{self.target_acos:.1f}%" if self.target_acos else "Not Set (30% default)"
            trs = f"{self.target_roas:.1f}x" if self.target_roas else "Not Set (3.0x default)"
            tcpa = format_currency(self.target_cpa) if self.target_cpa else "Not Set"
            ttcoas = (
                f"{self.target_tcoas:.1f}%" if self.target_tcoas else "Not Set"
            )

            return f"""
===============================================================================
AMAZON PPC PERFORMANCE REPORT
Client: {self.client_name}
Date: {datetime.now().strftime('%B %d, %Y')}
===============================================================================

OVERALL HEALTH: {h}/100 - {hs}

TARGETS
-------
ACOS : {tas}
ROAS : {trs}
CPA  : {tcpa}
TCoAS: {ttcoas}

FINANCIAL
---------
Spend : {format_currency(s['total_spend'])}
Sales : {format_currency(s['total_sales'])}
Profit: {format_currency(s['total_profit'])}
ROAS  : {s['roas']:.2f}x
ACOS  : {s['acos']:.1f}%
TCoAS : {s['tcoas']:.1f}%

METRICS
-------
Orders : {format_number(s['total_orders'])}
Clicks : {format_number(s['total_clicks'])}
Impr.  : {format_number(s['total_impressions'])}
CVR    : {s['avg_cvr']:.2f}%
CTR    : {s['avg_ctr']:.2f}%
CPA    : {format_currency(s['avg_cpa'])}
Avg CPC: {format_currency(s['avg_cpc'])}

KEYWORDS
--------
Scale Now : {len(c['high_potential'])}
Test      : {len(c['opportunities'])}
Watch     : {len(c['future_watch'])}
Reduce    : {len(c['low_potential'])}
Pause     : {len(c['wastage'])}

===============================================================================
Generated by Amazon Ads Dashboard Pro v7.0
===============================================================================
"""
        except Exception as e:
            return f"Error generating report: {e}"

# -----------------------------------------------------------------------------
# Session & layout helpers
# -----------------------------------------------------------------------------
def init_session_state():
    if "clients" not in st.session_state:
        st.session_state.clients: Dict[str, ClientData] = {}
    if "active_client" not in st.session_state:
        st.session_state.active_client: str | None = None
    if "agency_name" not in st.session_state:
        st.session_state.agency_name = "Your Agency"

def render_agency_header():
    st.markdown(
        f"""
        <div class="agency-header">
            <h1>🏢 {st.session_state.agency_name} – Amazon Ads Dashboard Pro v7.0</h1>
            <p>CPC & negatives fixed • Big-date reports supported • Amazon bulk‑ready bid exports</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar():
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=True):
            new_name = st.text_input(
                "Agency name", value=st.session_state.agency_name
            )
            if new_name != st.session_state.agency_name:
                st.session_state.agency_name = new_name
                st.rerun()

        st.markdown("---")
        st.markdown("### 👥 Clients")

        if st.session_state.clients:
            names = list(st.session_state.clients.keys())
            sel = st.selectbox("Active client", names)
            st.session_state.active_client = sel

        st.markdown("---")
        with st.expander("➕ Add client", expanded=False):
            nm = st.text_input("Client name*", key="add_client_name")
            ind = st.selectbox(
                "Industry",
                [
                    "E-commerce",
                    "Electronics",
                    "Fashion",
                    "Beauty",
                    "Home",
                    "Sports",
                    "Books",
                    "Health",
                    "Other",
                ],
            )
            bug = st.number_input(
                "Monthly budget (₹)", value=50_000, step=5_000, min_value=0
            )

            st.info("🎯 Targets (optional; 0 = use smart defaults)")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                t_acos = st.number_input(
                    "Target\nACOS %",
                    value=0.0,
                    step=5.0,
                    format="%.1f",
                    key="t_acos",
                )
            with c2:
                t_roas = st.number_input(
                    "Target\nROAS",
                    value=0.0,
                    step=0.5,
                    format="%.1f",
                    key="t_roas",
                )
            with c3:
                t_cpa = st.number_input(
                    "Target\nCPA ₹",
                    value=0.0,
                    step=50.0,
                    format="%.0f",
                    key="t_cpa",
                )
            with c4:
                t_tcoas = st.number_input(
                    "Target\nTCoAS %",
                    value=0.0,
                    step=5.0,
                    format="%.1f",
                    key="t_tcoas",
                )

            em = st.text_input("Client email (optional)")
            up = st.file_uploader(
                "Upload Sponsored Products search term / placement report*",
                type=["xlsx", "xls", "csv"],
            )

            if st.button("✅ Create client", use_container_width=True):
                if not nm:
                    st.error("Enter client name.")
                elif nm in st.session_state.clients:
                    st.error("Client with this name already exists.")
                elif not up:
                    st.error("Upload a report file.")
                else:
                    try:
                        if up.name.lower().endswith(".csv"):
                            df = pd.read_csv(up)
                        else:
                            df = pd.read_excel(up)

                        cd = ClientData(nm, ind, bug)
                        cd.contact_email = em
                        cd.target_acos = t_acos if t_acos > 0 else None
                        cd.target_roas = t_roas if t_roas > 0 else None
                        cd.target_cpa = t_cpa if t_cpa > 0 else None
                        cd.target_tcoas = t_tcoas if t_tcoas > 0 else None

                        analyzer = CompleteAnalyzer(
                            df,
                            nm,
                            cd.target_acos,
                            cd.target_roas,
                            cd.target_cpa,
                            cd.target_tcoas,
                        )
                        cd.analyzer = analyzer

                        st.session_state.clients[nm] = cd
                        st.session_state.active_client = nm
                        st.success("Client added.")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error while adding client: {e}")
                        with st.expander("Details"):
                            st.code(traceback.format_exc())

        if st.session_state.clients:
            st.markdown("---")
            st.markdown("### 📋 All clients")
            for name in list(st.session_state.clients.keys()):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"📊 {name}")
                with c2:
                    if st.button("❌", key=f"del_{name}"):
                        del st.session_state.clients[name]
                        if st.session_state.active_client == name:
                            st.session_state.active_client = None
                        st.rerun()

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
def render_dashboard_tab(cl: ClientData, an: CompleteAnalyzer):
    s = an.get_client_summary()
    h = an.get_health_score()

    tad = f"{cl.target_acos:.1f}%" if cl.target_acos else "30% (default)"
    trd = f"{cl.target_roas:.1f}x" if cl.target_roas else "3.0x (default)"
    tcpa = format_currency(cl.target_cpa) if cl.target_cpa else "Not set"
    ttcoas = f"{cl.target_tcoas:.1f}%" if cl.target_tcoas else "Not set"

    st.subheader("💰 Financial performance")

    st.markdown(
        f"""
        <div class="info-box">
        <strong>Health score:</strong> {h}/100<br>
        <strong>Targets:</strong> ACOS {tad} • ROAS {trd} • CPA {tcpa} • TCoAS {ttcoas}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Spend", format_currency(s["total_spend"]))
    with c2:
        st.metric("Sales", format_currency(s["total_sales"]))
    with c3:
        st.metric("ROAS", f"{s['roas']:.2f}x")
    with c4:
        st.metric("ACOS", f"{s['acos']:.1f}%")
    with c5:
        st.metric("TCoAS", f"{s['tcoas']:.1f}%")

    st.subheader("📈 Key metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Orders", format_number(s["total_orders"]))
    with c2:
        st.metric("Clicks", format_number(s["total_clicks"]))
    with c3:
        st.metric("CTR", f"{s['avg_ctr']:.2f}%")
    with c4:
        st.metric("CVR", f"{s['avg_cvr']:.2f}%")
    with c5:
        st.metric("Avg CPC", format_currency(s["avg_cpc"]))

    wp = s["total_wastage"] / s["total_spend"] * 100 if s["total_spend"] > 0 else 0
    st.markdown(
        f"""
        <div class="danger-box">
            <strong>Wastage (zero‑sales spend)</strong><br>
            {format_currency(s["total_wastage"])} ({wp:.1f}%)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("💡 Performance insights")
    ins = an.get_performance_insights()

    for bucket, style in [
        ("ctr_insights", "warning-box"),
        ("cvr_insights", "warning-box"),
        ("roas_insights", "warning-box"),
        ("acos_insights", "warning-box"),
        ("cpa_insights", "warning-box"),
        ("tcoas_insights", "warning-box"),
    ]:
        for item in ins[bucket]:
            lev = item["level"]
            box = (
                "danger-box"
                if lev == "critical"
                else "success-box"
                if lev == "success"
                else style
            )
            st.markdown(
                f"""
                <div class="{box}">
                    <strong>{item['metric']}: {item['value']}</strong><br>
                    Issue: {item['issue']}<br>
                    Action: {item['action']}
                </div>
                """,
                unsafe_allow_html=True,
            )

    plac = an.get_placement_recommendations()
    st.subheader("📍 Placement recommendations")
    if plac["available"]:
        st.dataframe(add_serial_column(plac["table"]), use_container_width=True)
        if plac["recs"]:
            for r in plac["recs"]:
                style = "success-box" if r["Action"] == "INCREASE" else (
                    "danger-box" if r["Action"] == "REDUCE" else "info-box"
                )
                st.markdown(
                    f"""
                    <div class="{style}">
                        <strong>{r['Placement']}</strong> – {r['Action']}<br>
                        {r['Recommendation']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            f'<div class="info-box">{plac["message"]}</div>',
            unsafe_allow_html=True,
        )

    if ins["content_suggestions"]:
        st.subheader("📝 Content & ad suggestions")
        st.markdown(
            '<div class="cyan-box"><strong>Ideas to improve CTR & CVR</strong></div>',
            unsafe_allow_html=True,
        )
        for txt in ins["content_suggestions"]:
            st.markdown(f"- {txt}")

def render_keywords_tab(an: CompleteAnalyzer):
    st.subheader("🎯 Keyword groups")
    cats = an.classify_keywords_improved()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🏆 Scale", len(cats["high_potential"]))
    with c2:
        st.metric("⚡ Test", len(cats["opportunities"]))
    with c3:
        st.metric("👀 Watch", len(cats["future_watch"]))
    with c4:
        st.metric("⚠️ Reduce", len(cats["low_potential"]))
    with c5:
        st.metric("🚨 Pause", len(cats["wastage"]))

    tabs = st.tabs(
        [
            f"🏆 Scale ({len(cats['high_potential'])})",
            f"⚡ Test ({len(cats['opportunities'])})",
            f"👀 Watch ({len(cats['future_watch'])})",
            f"⚠️ Reduce ({len(cats['low_potential'])})",
            f"🚨 Pause ({len(cats['wastage'])})",
        ]
    )

    with tabs[0]:
        if cats["high_potential"]:
            st.success("These are your winners. Increase bids 15–25%.")
            st.dataframe(
                add_serial_column(pd.DataFrame(cats["high_potential"])),
                use_container_width=True,
            )
        else:
            st.info("No clear champions yet.")

    with tabs[1]:
        if cats["opportunities"]:
            st.info("Keywords with good potential – test +10–15% bids.")
            st.dataframe(
                add_serial_column(pd.DataFrame(cats["opportunities"])),
                use_container_width=True,
            )
        else:
            st.info("No opportunity keywords yet.")

    with tabs[2]:
        if cats["future_watch"]:
            st.dataframe(
                add_serial_column(pd.DataFrame(cats["future_watch"])),
                use_container_width=True,
            )
        else:
            st.info("No watchlist keywords.")

    with tabs[3]:
        if cats["low_potential"]:
            st.warning("Consider reducing bids by ~30%.")
            st.dataframe(
                add_serial_column(pd.DataFrame(cats["low_potential"])),
                use_container_width=True,
            )
        else:
            st.success("No low‑potential keywords.")

    with tabs[4]:
        if cats["wastage"]:
            tw = sum(
                safe_float(x["Spend"].replace("₹", "").replace(",", ""))
                for x in cats["wastage"]
            )
            st.error(
                f"Wastage on zero‑sales keywords: {format_currency(tw)}"
            )
            st.dataframe(
                add_serial_column(pd.DataFrame(cats["wastage"])),
                use_container_width=True,
            )
        else:
            st.success("No pure wastage keywords – great!")

def render_bid_tab(an: CompleteAnalyzer):
    st.subheader("💡 Bid optimization")
    sug = an.get_bid_suggestions_improved()

    tad = f"{an.target_acos:.1f}%" if an.target_acos else "30% (default)"
    trd = f"{an.target_roas:.1f}x" if an.target_roas else "3.0x (default)"
    tcpa = format_currency(an.target_cpa) if an.target_cpa else "Not set"

    st.markdown(
        f"""
        <div class="info-box">
        <strong>Targets:</strong> ACOS {tad} • ROAS {trd} • CPA {tcpa}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not sug:
        st.info("No bid suggestions – data may be too small or already optimized.")
        return

    act_filter = st.selectbox(
        "Filter by action",
        ["All", "INCREASE", "REDUCE", "PAUSE"],
        index=0,
    )
    if act_filter != "All":
        view = [x for x in sug if x["Action"] == act_filter]
    else:
        view = sug

    inc = len([x for x in sug if x["Action"] == "INCREASE"])
    red = len([x for x in sug if x["Action"] == "REDUCE"])
    pau = len([x for x in sug if x["Action"] == "PAUSE"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("⬆️ Increase", inc)
    with c2:
        st.metric("⬇️ Reduce", red)
    with c3:
        st.metric("⏸️ Pause", pau)

    st.markdown(f"Showing **{len(view)}** of {len(sug)} suggestions.")
    st.dataframe(
        add_serial_column(pd.DataFrame(view)),
        use_container_width=True,
        height=500,
    )

def render_exports_tab(an: CompleteAnalyzer, client_name: str):
    st.subheader("📥 Exports")

    cats = an.classify_keywords_improved()
    sug = an.get_bid_suggestions_improved()

    c1, c2, c3 = st.columns(3)

    # Negatives
    with c1:
        st.markdown("#### 🚫 Negative keywords")
        wast = cats["wastage"]
        if wast:
            rows = [
                {
                    "Record Type": "Keyword",
                    "Campaign Name": k["Campaign"],
                    "Ad Group Name": "",
                    "Keyword or Product Targeting": k["Keyword"],
                    "Match Type": "Negative Exact",
                    "State": "enabled",
                    "Operation": "create",
                }
                for k in wast
            ]
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as wr:
                pd.DataFrame(rows).to_excel(
                    wr,
                    index=False,
                    sheet_name="Negatives",
                )
            out.seek(0)
            st.download_button(
                f"Download negatives ({len(rows)})",
                data=out,
                file_name=f"Negatives_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.info("No pure wastage keywords → no negatives needed.")

    # Amazon bulk bid changes
    with c2:
        st.markdown("#### 💰 Bid adjustments (Amazon bulk)")
        if sug:
            bulk_df = an.build_amazon_bulk_from_bids(sug)
            if len(bulk_df) == 0:
                st.info("Suggestions exist but not enough data to build bulk file.")
            else:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="xlsxwriter") as wr:
                    bulk_df.to_excel(
                        wr,
                        index=False,
                        sheet_name="Sponsored Products",
                    )
                out.seek(0)
                st.download_button(
                    f"Download bulk bid file ({len(bulk_df)} rows)",
                    data=out,
                    file_name=f"Bulk_Bids_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.success(
                    "File is in Amazon Sponsored Products bulk format – "
                    "upload directly in Ads console."
                )
        else:
            st.info("No bid suggestions → nothing to export.")

    # Full data CSV
    with c3:
        st.markdown("#### 📊 Raw dataset")
        if an.df is not None:
            csv = an.df.to_csv(index=False)
            st.download_button(
                f"Download CSV ({len(an.df)} rows)",
                data=csv,
                file_name=f"Full_Data_{client_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No data for this client.")

def render_report_tab(cl: ClientData, an: CompleteAnalyzer):
    st.subheader("📝 Client report")
    txt = an.generate_client_report()
    st.text_area("Report", txt, height=500)
    st.download_button(
        "Download TXT",
        data=txt,
        file_name=f"Report_{cl.name}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

def render_all_clients_tab():
    st.subheader("👥 All clients overview")
    data = []
    for name, c in st.session_state.clients.items():
        if not c.analyzer or c.analyzer.df is None:
            continue
        s = c.analyzer.get_client_summary()
        h = c.analyzer.get_health_score()
        data.append(
            {
                "Client": name,
                "Health": f"{h}/100",
                "Spend": format_currency(s["total_spend"]),
                "Sales": format_currency(s["total_sales"]),
                "ROAS": f"{s['roas']:.2f}x",
                "ACOS": f"{s['acos']:.1f}%",
                "TCoAS": f"{s['tcoas']:.1f}%",
                "CVR": f"{s['avg_cvr']:.2f}%",
                "Keywords": format_number(s["keywords_count"]),
            }
        )
    if not data:
        st.info("No clients with data yet.")
        return
    st.dataframe(add_serial_column(pd.DataFrame(data)), use_container_width=True)

# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------
def render_dashboard():
    render_agency_header()

    if not st.session_state.clients:
        st.info("Add a client from the left sidebar to start.")
        return

    if not st.session_state.active_client:
        st.warning("Select a client from the left sidebar.")
        return

    cl = st.session_state.clients[st.session_state.active_client]
    if not cl.analyzer or cl.analyzer.df is None:
        st.error("No data loaded for this client.")
        return

    an = cl.analyzer

    tabs = st.tabs(
        [
            "📊 Dashboard",
            "🎯 Keywords",
            "💡 Bids",
            "📝 Report",
            "📥 Exports",
            "👥 All clients",
        ]
    )

    with tabs[0]:
        render_dashboard_tab(cl, an)
    with tabs[1]:
        render_keywords_tab(an)
    with tabs[2]:
        render_bid_tab(an)
    with tabs[3]:
        render_report_tab(cl, an)
    with tabs[4]:
        render_exports_tab(an, cl.name)
    with tabs[5]:
        render_all_clients_tab()

def main():
    load_custom_css()
    init_session_state()
    render_sidebar()
    render_dashboard()

    st.markdown(
        """
        <hr>
        <div style="text-align:center;color:#64748b;font-size:0.8rem;padding:0.6rem 0;">
        Amazon Ads Dashboard Pro v7.0 – tuned for big-date reports and premium visuals.
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
