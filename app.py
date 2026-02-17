#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amazon Ads Agency Dashboard Pro (app.py)
Based on your amazon_ads_dashboard_v6.py, with fixes:
- Supports 7/14/30 Day Total Sales/Orders columns (maps to Sales/Orders)
- More robust numeric parsing for ₹/$, commas, percent, ( ) negatives
- Keeps CPC calculation from Spend/Clicks
"""

import io
import re
import traceback
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Amazon Ads Dashboard Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# UI helpers
# =============================================================================
def load_custom_css():
    # Keep minimal to avoid bugs; you can add your CSS
    css = """
    <style>
    .block-container {padding-top: 1.2rem;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# Safe parsing helpers (improved)
# =============================================================================
def safe_float(value, default=0.0):
    try:
        if value is None or value == '':
            return default
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return default
            return float(value)

        s = str(value).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default

        # negatives like (123.45)
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        # remove commas/currency/percent tokens
        s = s.replace(",", "")
        s = s.replace("₹", "").replace("$", "").replace("%", "")
        s = s.replace("INR", "").replace("Rs.", "").replace("Rs", "")

        # extract first valid number
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(round(safe_float(value, default)))
    except Exception:
        return default


def safe_str(value, default="N/A"):
    try:
        if value is None:
            return default
        if isinstance(value, float) and pd.isna(value):
            return default
        s = str(value).strip()
        return s if s else default
    except Exception:
        return default


def format_currency(value):
    try:
        val = safe_float(value, 0)
        if val >= 10000000:
            return f"₹{val/10000000:.2f}Cr"
        elif val >= 100000:
            return f"₹{val/100000:.2f}L"
        else:
            return f"₹{val:,.2f}"
    except Exception:
        return "₹0.00"


def format_number(value):
    try:
        val = safe_int(value, 0)
        if val >= 10000000:
            return f"{val/10000000:.2f}Cr"
        elif val >= 100000:
            return f"{val/100000:.2f}L"
        elif val >= 1000:
            return f"{val:,}"
        else:
            return str(val)
    except Exception:
        return "0"


def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.reset_index(drop=True).copy()
    out.insert(0, "S.No", range(1, len(out) + 1))
    return out


def is_asin(value):
    if not value or (isinstance(value, float) and pd.isna(value)):
        return False
    val_str = str(value).strip().upper()
    return bool(re.match(r"^[B][0-9A-Z]{9,}$", val_str))


def get_negative_type(value):
    return "PRODUCT" if is_asin(value) else "KEYWORD"


# =============================================================================
# Data models
# =============================================================================
class ClientData:
    def __init__(self, name, industry="E-commerce", budget=50000):
        self.name = name
        self.industry = industry
        self.monthly_budget = budget
        self.analyzer = None
        self.added_date = datetime.now()
        self.contact_email = ""
        self.target_acos = None
        self.target_roas = None
        self.target_cpa = None
        self.target_tcoas = None


class CompleteAnalyzer:
    REQUIRED_COLUMNS = ["Customer Search Term", "Campaign Name", "Spend", "Clicks"]

    def __init__(self, df, client_name, target_acos=None, target_roas=None, target_cpa=None, target_tcoas=None):
        self.client_name = client_name
        self.target_acos = target_acos
        self.target_roas = target_roas
        self.target_cpa = target_cpa
        self.target_tcoas = target_tcoas

        self.df = None
        self.raw_df = None
        self.error = None
        self.column_mapping = {}

        # diagnostics
        self.detected_sales_col = None
        self.detected_orders_col = None

        try:
            self.raw_df = df.copy(deep=True)
            self.df = self._validate_and_prepare_data(df.copy(deep=True))
        except Exception as e:
            self.error = str(e)
            raise ValueError(f"Validation failed: {e}")

    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame")

        # Clean column names
        df.columns = df.columns.str.strip()
        original_columns = list(df.columns)

        # Build lowercase columns for mapping
        df.columns = df.columns.str.lower().str.strip()

        # EXTENDED column mapping (UPDATED: 7/14/30 day)
        mapping = {
            # Core
            "customer search term": "Customer Search Term",
            "search term": "Customer Search Term",
            "search terms": "Customer Search Term",
            "keyword": "Customer Search Term",
            "campaign": "Campaign Name",
            "campaign name": "Campaign Name",
            "campaign_name": "Campaign Name",
            "ad group": "Ad Group Name",
            "ad group name": "Ad Group Name",
            "adgroup": "Ad Group Name",
            "ad_group_name": "Ad Group Name",
            "match type": "Match Type",
            "matchtype": "Match Type",
            "match_type": "Match Type",

            # Spend/clicks/imps
            "cost": "Spend",
            "spend": "Spend",
            "ad spend": "Spend",
            "spend (₹)": "Spend",
            "spend ($)": "Spend",
            "clicks": "Clicks",
            "impressions": "Impressions",
            "imps": "Impressions",

            # CPC
            "cpc": "CPC",
            "cost per click": "CPC",
            "avg cpc": "CPC",
            "average cpc": "CPC",

            # SALES (7/14/30 day + generic)
            "7 day total sales": "Sales",
            "7 day total sales (₹)": "Sales",
            "7 day total sales ($)": "Sales",
            "7 day sales": "Sales",
            "7 day total revenue": "Sales",
            "14 day total sales": "Sales",
            "14 day total sales (₹)": "Sales",
            "14 day total sales ($)": "Sales",
            "30 day total sales": "Sales",
            "30 day total sales (₹)": "Sales",
            "30 day total sales ($)": "Sales",
            "total sales": "Sales",
            "sales": "Sales",
            "revenue": "Sales",
            "sales (₹)": "Sales",
            "sales ($)": "Sales",

            # ORDERS (7/14/30 day + generic)
            "7 day total orders": "Orders",
            "7 day total orders (#)": "Orders",
            "7 day orders": "Orders",
            "7 day total units": "Orders",
            "7 day ordered units": "Orders",
            "14 day total orders": "Orders",
            "14 day total orders (#)": "Orders",
            "14 day total units": "Orders",
            "30 day total orders": "Orders",
            "30 day total orders (#)": "Orders",
            "30 day total units": "Orders",
            "total orders": "Orders",
            "orders": "Orders",
            "units": "Orders",
        }

        # Rename columns if present
        for old, new in mapping.items():
            old_clean = old.lower().strip()
            if old_clean in df.columns:
                df.rename(columns={old_clean: new}, inplace=True)
                self.column_mapping[old] = new

        # Title-case remaining columns to match existing logic
        df.columns = [c.title() for c in df.columns]

        # Check required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

        # Optional columns
        optional_numeric = ["Sales", "Orders", "Impressions", "Cpc"]
        optional_text = ["Ad Group Name", "Match Type"]
        for col in optional_numeric:
            if col not in df.columns:
                df[col] = 0
        for col in optional_text:
            if col not in df.columns:
                df[col] = "N/A"

        # Ensure CPC column exists
        if "Cpc" in df.columns and "CPC" not in df.columns:
            df["CPC"] = df["Cpc"]
        if "CPC" not in df.columns:
            df["CPC"] = 0

        # Convert numeric columns
        numeric_cols = ["Spend", "Sales", "Clicks", "Impressions", "Orders", "CPC"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_float(x, 0.0))

        # Calculate CPC from Spend/Clicks if missing/0
        df["CPC_Calculated"] = df.apply(
            lambda x: safe_float(x.get("Spend", 0)) / safe_float(x.get("Clicks", 1))
            if safe_float(x.get("Clicks", 0)) > 0 else 0,
            axis=1
        )
        df["CPC"] = df.apply(
            lambda x: safe_float(x.get("CPC", 0)) if safe_float(x.get("CPC", 0)) > 0 else safe_float(x.get("CPC_Calculated", 0)),
            axis=1
        )

        # Filter rows with activity
        df = df[(df["Spend"] > 0) | (df["Clicks"] > 0)].copy()
        if len(df) == 0:
            raise ValueError("No valid data after filtering")

        # Derived metrics
        df["Profit"] = df["Sales"] - df["Spend"]
        df["Wastage"] = df.apply(
            lambda x: safe_float(x.get("Spend", 0)) if safe_float(x.get("Sales", 0)) == 0 else 0,
            axis=1
        )
        df["CVR"] = df.apply(
            lambda x: (safe_float(x.get("Orders", 0)) / safe_float(x.get("Clicks", 1)) * 100)
            if safe_float(x.get("Clicks", 0)) > 0 else 0,
            axis=1
        )
        df["ROAS"] = df.apply(
            lambda x: (safe_float(x.get("Sales", 0)) / safe_float(x.get("Spend", 1)))
            if safe_float(x.get("Spend", 0)) > 0 else 0,
            axis=1
        )
        df["ACOS"] = df.apply(
            lambda x: (safe_float(x.get("Spend", 0)) / safe_float(x.get("Sales", 1)) * 100)
            if safe_float(x.get("Sales", 0)) > 0 else 0,
            axis=1
        )
        df["CTR"] = df.apply(
            lambda x: (safe_float(x.get("Clicks", 0)) / safe_float(x.get("Impressions", 1)) * 100)
            if safe_float(x.get("Impressions", 0)) > 0 else 0,
            axis=1
        )
        df["CPA"] = df.apply(
            lambda x: (safe_float(x.get("Spend", 0)) / safe_float(x.get("Orders", 1)))
            if safe_float(x.get("Orders", 0)) > 0 else 0,
            axis=1
        )
        df["TCoAS"] = df["ACOS"]

        # Negative type
        df["Negative_Type"] = df["Customer Search Term"].apply(get_negative_type)
        df["Client"] = self.client_name
        df["Processed_Date"] = datetime.now()

        return df

    def _empty_summary(self):
        return {k: 0 for k in [
            "total_spend", "total_sales", "total_profit", "total_orders", "total_clicks",
            "total_impressions", "total_wastage", "roas", "acos", "tcoas",
            "avg_cpc", "avg_ctr", "avg_cvr", "avg_cpa", "conversion_rate",
            "keywords_count", "campaigns_count", "ad_groups_count"
        ]}

    def get_client_summary(self) -> Dict:
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

            # Avg CPC (ignore zero)
            avg_cpc = 0.0
            if "CPC" in self.df.columns:
                cpc_values = self.df[self.df["CPC"] > 0]["CPC"]
                if len(cpc_values) > 0:
                    avg_cpc = safe_float(cpc_values.mean())
            if avg_cpc == 0 and tc > 0:
                avg_cpc = ts / tc

            avg_ctr = safe_float(self.df["CTR"].mean())
            avg_cvr = safe_float(self.df["CVR"].mean())
            avg_cpa = (ts / to) if to > 0 else 0
            avg_acos = (ts / tsa * 100) if tsa > 0 else 0
            avg_roas = (tsa / ts) if ts > 0 else 0
            avg_tcoas = avg_acos

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
                "tcoas": avg_tcoas,
                "avg_cpc": avg_cpc,
                "avg_ctr": avg_ctr,
                "avg_cvr": avg_cvr,
                "avg_cpa": avg_cpa,
                "conversion_rate": (to / tc * 100) if tc > 0 else 0,
                "keywords_count": len(self.df),
                "campaigns_count": safe_int(self.df["Campaign Name"].nunique()),
                "ad_groups_count": safe_int(self.df["Ad Group Name"].nunique()) if "Ad Group Name" in self.df.columns else 0,
            }
        except Exception:
            return self._empty_summary()

    # Keeping your v6.0 logic (shortened: you can extend more later)
    def classify_keywords_improved(self):
        cats = {
            "high_potential": [],
            "low_potential": [],
            "wastage": [],
            "opportunities": [],
            "future_watch": [],
        }
        if self.df is None or len(self.df) == 0:
            return cats

        for _, r in self.df.iterrows():
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
                "Reason": ""
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
            elif sp >= 20 and (1.5 <= ro < 2.5) and c >= 3:
                kd["Reason"] = f"Good potential ROAS {ro:.2f}x - test +10-15%"
                cats["opportunities"].append(kd)
            elif c >= 3 and sp < 50 and sa == 0:
                kd["Reason"] = f"{c} clicks, ₹{sp:.0f} - gather more data"
                cats["future_watch"].append(kd)

        return cats

    def get_bid_suggestions_improved(self):
        sug = []
        if self.df is None or len(self.df) == 0:
            return sug

        ta = self.target_acos or 30.0
        tr = self.target_roas or 3.0

        for _, r in self.df.iterrows():
            sp = safe_float(r.get("Spend", 0))
            sa = safe_float(r.get("Sales", 0))
            ro = safe_float(r.get("ROAS", 0))
            o = safe_int(r.get("Orders", 0))
            c = safe_int(r.get("Clicks", 0))
            cv = safe_float(r.get("CVR", 0))
            cpc = safe_float(r.get("CPC", 0))

            if sp < 20 or c < 3 or cpc <= 0:
                continue

            ac = (sp / sa * 100) if sa > 0 else 999

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
                "Reason": ""
            }

            if ro >= 3.0 and cv >= 2.0 and o >= 2:
                nb = cpc * 1.25
                s.update({
                    "Action": "⬆️ INCREASE",
                    "Suggested Bid": format_currency(nb),
                    "Change (%)": 25,
                    "Reason": f"Champion keyword! ROAS {ro:.2f}x"
                })
                sug.append(s)
            elif ro >= tr and cv >= 1.0 and o >= 1:
                nb = cpc * 1.15
                s.update({
                    "Action": "⬆️ INCREASE",
                    "Suggested Bid": format_currency(nb),
                    "Change (%)": 15,
                    "Reason": "Above target ROAS"
                })
                sug.append(s)
            elif sa == 0 and sp >= 50:
                s.update({
                    "Action": "⏸️ PAUSE",
                    "Suggested Bid": "₹0.00",
                    "Change (%)": -100,
                    "Reason": f"₹{sp:.0f} wasted, no sales"
                })
                sug.append(s)
            elif ro < 1.5 and sp >= 30:
                nb = cpc * 0.7
                s.update({
                    "Action": "⬇️ REDUCE",
                    "Suggested Bid": format_currency(nb),
                    "Change (%)": -30,
                    "Reason": f"Poor ROAS {ro:.2f}x"
                })
                sug.append(s)
            elif ac > ta and sp >= 30:
                red = min(30, (ac - ta) / ta * 100)
                nb = cpc * (1 - red / 100)
                s.update({
                    "Action": "⬇️ REDUCE",
                    "Suggested Bid": format_currency(nb),
                    "Change (%)": -int(red),
                    "Reason": f"ACOS {ac:.1f}% above target {ta:.1f}%"
                })
                sug.append(s)

        # Sort by spend (numeric)
        def _spend_to_num(x):
            return safe_float(str(x).replace("₹", "").replace(",", "").replace("L", "").replace("Cr", ""), 0)

        return sorted(sug, key=lambda x: _spend_to_num(x.get("Spend", 0)), reverse=True)


# =============================================================================
# Session state
# =============================================================================
def init_session_state():
    if "clients" not in st.session_state:
        st.session_state.clients = {}
    if "active_client" not in st.session_state:
        st.session_state.active_client = None
    if "agency_name" not in st.session_state:
        st.session_state.agency_name = "Your Agency"
    if "show_diagnostics" not in st.session_state:
        st.session_state.show_diagnostics = False


# =============================================================================
# File loading (cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# =============================================================================
# Sidebar
# =============================================================================
def sidebar_ui():
    with st.sidebar:
        st.header("Client Setup")

        st.session_state.agency_name = st.text_input("Agency name", st.session_state.agency_name)
        st.session_state.show_diagnostics = st.checkbox("Show diagnostics", value=st.session_state.show_diagnostics)

        client_name = st.text_input("Client name", value="Client 1")

        col1, col2 = st.columns(2)
        with col1:
            target_acos = st.number_input("Target ACOS %", value=30.0, step=1.0)
        with col2:
            target_roas = st.number_input("Target ROAS", value=3.0, step=0.1)

        uploaded = st.file_uploader("Upload Search Term report (CSV/XLSX)", type=["csv", "xlsx", "xls"])

        if st.button("Add / Update Client", use_container_width=True):
            if not uploaded:
                st.error("Please upload a report file.")
                return

            try:
                with st.spinner("Processing file…"):
                    raw = read_uploaded_file(uploaded)
                    analyzer = CompleteAnalyzer(
                        raw,
                        client_name=client_name,
                        target_acos=float(target_acos),
                        target_roas=float(target_roas),
                    )

                cd = ClientData(client_name)
                cd.target_acos = float(target_acos)
                cd.target_roas = float(target_roas)
                cd.analyzer = analyzer

                st.session_state.clients[client_name] = cd
                st.session_state.active_client = client_name
                st.success("Client loaded successfully.")
                st.rerun()

            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.code(traceback.format_exc())

        st.divider()

        if st.session_state.clients:
            names = list(st.session_state.clients.keys())
            idx = names.index(st.session_state.active_client) if st.session_state.active_client in names else 0
            st.session_state.active_client = st.selectbox("Active client", names, index=idx)

        if st.button("Clear all clients", use_container_width=True):
            st.session_state.clients = {}
            st.session_state.active_client = None
            st.rerun()


# =============================================================================
# Main dashboard
# =============================================================================
def render_header():
    st.markdown(f"## {st.session_state.agency_name} — Amazon Ads Dashboard Pro")


def render_dashboard(client: ClientData):
    an = client.analyzer
    if an is None or an.df is None or len(an.df) == 0:
        st.warning("No data available.")
        return

    s = an.get_client_summary()

    st.subheader("Performance Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spend", format_currency(s["total_spend"]))
    c2.metric("Sales", format_currency(s["total_sales"]))
    c3.metric("ROAS", f"{s['roas']:.2f}x")
    c4.metric("ACOS", f"{s['acos']:.1f}%")
    c5.metric("Wastage", format_currency(s["total_wastage"]))

    st.subheader("Traffic & Efficiency")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Orders", format_number(s["total_orders"]))
    k2.metric("Clicks", format_number(s["total_clicks"]))
    k3.metric("Impressions", format_number(s["total_impressions"]))
    k4.metric("Avg CPC", format_currency(s["avg_cpc"]))
    k5.metric("CVR", f"{s['avg_cvr']:.2f}%")

    if st.session_state.show_diagnostics:
        with st.expander("Diagnostics", expanded=False):
            st.write("Rows:", len(an.df))
            st.write("Columns:", list(an.raw_df.columns)[:30] if an.raw_df is not None else [])
            st.write("Tip: This build supports 7/14/30 Day Total Sales/Orders → Sales/Orders.")

    st.subheader("Keyword Buckets")
    cats = an.classify_keywords_improved()
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Scale now", len(cats["high_potential"]))
    b2.metric("Opportunities", len(cats["opportunities"]))
    b3.metric("Watch", len(cats["future_watch"]))
    b4.metric("Reduce", len(cats["low_potential"]))
    b5.metric("Pause", len(cats["wastage"]))

    st.subheader("Bid suggestions")
    bids = an.get_bid_suggestions_improved()
    if bids:
        st.dataframe(add_serial_column(pd.DataFrame(bids).head(300)), use_container_width=True, hide_index=True)
    else:
        st.info("No bid suggestions (need enough clicks/spend and CPC > 0).")

    st.subheader("Data preview")
    cols = ["Customer Search Term", "Campaign Name", "Ad Group Name", "Match Type",
            "Spend", "Sales", "Orders", "Clicks", "Impressions", "CPC", "ROAS", "ACOS", "CVR", "CTR"]
    cols = [c for c in cols if c in an.df.columns]
    st.dataframe(add_serial_column(an.df[cols].head(500)), use_container_width=True, hide_index=True)


def main():
    load_custom_css()
    init_session_state()
    sidebar_ui()
    render_header()

    if not st.session_state.active_client or st.session_state.active_client not in st.session_state.clients:
        st.info("Upload a report from the sidebar to start.")
        return

    client = st.session_state.clients[st.session_state.active_client]
    st.caption(f"Active: {client.name}")
    render_dashboard(client)


if __name__ == "__main__":
    main()
