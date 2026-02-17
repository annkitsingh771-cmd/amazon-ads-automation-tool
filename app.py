#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amazon Ads Agency Dashboard Pro v6.2 (7/14/30-day Sales fix)

Fixes:
- Sales/Orders detection for 7/14/30 day search term reports
- Robust currency/number parsing (₹, commas, excel formats)
- CPC calculation from Spend/Clicks when missing
- S.No starts from 1 (hide default index)
"""

import io
import re
import traceback
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
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
# Styling (keep simple, you can paste your premium CSS here)
# =============================================================================
def load_custom_css():
    css = """
    <style>
    .main {background: #0b1220;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# Robust parsing helpers (IMPORTANT FIX)
# =============================================================================
def safe_float(value, default=0.0):
    try:
        if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
            return default
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        s = str(value).strip()
        s = s.replace(',', '')
        # handle negatives like (123.45)
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        # remove common currency/percent tokens
        s = s.replace('₹', '').replace('$', '').replace('%', '').replace('INR', '').replace('Rs.', '').replace('Rs', '')
        # Excel style tokens e.g. [$₹-en-US]123.4 -> just extract number
        m = re.search(r'-?\d+(?:\.\d+)?', s)
        return float(m.group(0)) if m else default
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(round(safe_float(value, default)))
    except Exception:
        return default


def safe_str(value, default='N/A'):
    try:
        if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
            return default
        return str(value).strip()
    except Exception:
        return default


def format_currency(value):
    val = safe_float(value, 0)
    if val >= 10000000:
        return f"₹{val/10000000:.2f}Cr"
    if val >= 100000:
        return f"₹{val/100000:.2f}L"
    return f"₹{val:,.2f}"


def format_number(value):
    val = safe_int(value, 0)
    if val >= 10000000:
        return f"{val/10000000:.2f}Cr"
    if val >= 100000:
        return f"{val/100000:.2f}L"
    if val >= 1000:
        return f"{val:,}"
    return str(val)


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
    return bool(re.match(r'^[B][0-9A-Z]{9,}$', val_str))


def get_negative_type(value):
    return 'PRODUCT' if is_asin(value) else 'KEYWORD'


# =============================================================================
# Core classes
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
    REQUIRED_COLUMNS = ['Customer Search Term', 'Campaign Name', 'Spend', 'Clicks']

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
        self.sales_source = ""
        self.orders_source = ""

        try:
            self.raw_df = df.copy(deep=True)
            self.df = self._validate_and_prepare_data(df.copy(deep=True))
        except Exception as e:
            self.error = str(e)
            raise ValueError(f"Validation failed: {e}")

    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame")

        # Normalize column names
        df.columns = df.columns.str.strip()
        original_cols = list(df.columns)

        # Lowercase for matching
        df.columns = df.columns.str.lower().str.strip()

        # -------------------------
        # EXTENDED mapping (FIXED)
        # Now supports 7/14/30 day Sales/Orders names.
        # -------------------------
        mapping = {
            # Core dimensions
            'customer search term': 'Customer Search Term',
            'search term': 'Customer Search Term',
            'keyword': 'Customer Search Term',
            'campaign name': 'Campaign Name',
            'campaign': 'Campaign Name',
            'ad group name': 'Ad Group Name',
            'ad group': 'Ad Group Name',
            'match type': 'Match Type',

            # Spend/Clicks/Imps
            'spend': 'Spend',
            'cost': 'Spend',
            'ad spend': 'Spend',
            'clicks': 'Clicks',
            'impressions': 'Impressions',
            'imps': 'Impressions',

            # CPC
            'cost per click (cpc)': 'CPC',
            'cost per click': 'CPC',
            'avg cpc': 'CPC',
            'cpc': 'CPC',

            # Sales (7/14/30 day + generic)
            '7 day total sales (₹)': 'Sales',
            '7 day total sales ($)': 'Sales',
            '7 day total sales': 'Sales',
            '14 day total sales (₹)': 'Sales',
            '14 day total sales ($)': 'Sales',
            '14 day total sales': 'Sales',
            '30 day total sales (₹)': 'Sales',
            '30 day total sales ($)': 'Sales',
            '30 day total sales': 'Sales',
            'total sales': 'Sales',
            'sales': 'Sales',
            'revenue': 'Sales',

            # Orders (7/14/30 day + generic)
            '7 day total orders (#)': 'Orders',
            '7 day total orders': 'Orders',
            '7 day total units (#)': 'Orders',
            '7 day total units': 'Orders',
            '14 day total orders (#)': 'Orders',
            '14 day total orders': 'Orders',
            '14 day total units (#)': 'Orders',
            '14 day total units': 'Orders',
            '30 day total orders (#)': 'Orders',
            '30 day total orders': 'Orders',
            '30 day total units (#)': 'Orders',
            '30 day total units': 'Orders',
            'total orders': 'Orders',
            'orders': 'Orders',
            'units': 'Orders',
        }

        # Rename using mapping if exact match exists
        for old_lower, new_name in mapping.items():
            if old_lower in df.columns:
                df.rename(columns={old_lower: new_name}, inplace=True)

        # After rename, title-case the remaining columns to match old logic
        df.columns = [c.title() for c in df.columns]

        # Detect which sales/orders columns we used (for debugging)
        # If original had 14 day sales, it got renamed to 'Sales'
        self.sales_source = "Sales"
        self.orders_source = "Orders"

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

        # Optional columns
        for col in ['Sales', 'Orders', 'Impressions', 'Cpc', 'CPC']:
            if col not in df.columns:
                df[col] = 0
        for col in ['Ad Group Name', 'Match Type']:
            if col not in df.columns:
                df[col] = 'N/A'

        # Ensure CPC column exists
        if 'Cpc' in df.columns and 'CPC' not in df.columns:
            df['CPC'] = df['Cpc']

        # Numeric conversion (robust)
        numeric_cols = ['Spend', 'Sales', 'Clicks', 'Impressions', 'Orders', 'CPC']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_float(x, 0.0))

        # CPC fallback
        df['CPC_Calculated'] = df.apply(
            lambda x: safe_float(x.get('Spend', 0)) / safe_float(x.get('Clicks', 1))
            if safe_float(x.get('Clicks', 0)) > 0 else 0,
            axis=1
        )
        df['CPC'] = df.apply(
            lambda x: safe_float(x.get('CPC', 0)) if safe_float(x.get('CPC', 0)) > 0 else safe_float(x.get('CPC_Calculated', 0)),
            axis=1
        )

        # Filter activity
        df = df[(df['Spend'] > 0) | (df['Clicks'] > 0)].copy()
        if len(df) == 0:
            raise ValueError("No valid data after filtering")

        # Derived metrics
        df['Profit'] = df['Sales'] - df['Spend']
        df['Wastage'] = df.apply(lambda x: safe_float(x.get('Spend', 0)) if safe_float(x.get('Sales', 0)) == 0 else 0, axis=1)
        df['CVR'] = df.apply(lambda x: (safe_float(x.get('Orders', 0)) / safe_float(x.get('Clicks', 1)) * 100) if safe_float(x.get('Clicks', 0)) > 0 else 0, axis=1)
        df['ROAS'] = df.apply(lambda x: (safe_float(x.get('Sales', 0)) / safe_float(x.get('Spend', 1))) if safe_float(x.get('Spend', 0)) > 0 else 0, axis=1)
        df['ACOS'] = df.apply(lambda x: (safe_float(x.get('Spend', 0)) / safe_float(x.get('Sales', 1)) * 100) if safe_float(x.get('Sales', 0)) > 0 else 0, axis=1)
        df['CTR'] = df.apply(lambda x: (safe_float(x.get('Clicks', 0)) / safe_float(x.get('Impressions', 1)) * 100) if safe_float(x.get('Impressions', 0)) > 0 else 0, axis=1)
        df['CPA'] = df.apply(lambda x: (safe_float(x.get('Spend', 0)) / safe_float(x.get('Orders', 1))) if safe_float(x.get('Orders', 0)) > 0 else 0, axis=1)
        df['TCoAS'] = df['ACOS']

        df['Negative_Type'] = df['Customer Search Term'].apply(get_negative_type)
        df['Client'] = self.client_name
        df['Processed_Date'] = datetime.now()
        return df

    def _empty_summary(self):
        return {k: 0 for k in [
            'total_spend', 'total_sales', 'total_profit', 'total_orders', 'total_clicks', 'total_impressions',
            'total_wastage', 'roas', 'acos', 'tcoas', 'avg_cpc', 'avg_ctr', 'avg_cvr', 'avg_cpa',
            'conversion_rate', 'keywords_count', 'campaigns_count', 'ad_groups_count'
        ]}

    def get_client_summary(self) -> Dict:
        try:
            if self.df is None or len(self.df) == 0:
                return self._empty_summary()

            ts = safe_float(self.df['Spend'].sum())
            tsa = safe_float(self.df['Sales'].sum())
            to = safe_int(self.df['Orders'].sum())
            tc = safe_int(self.df['Clicks'].sum())
            ti = safe_int(self.df['Impressions'].sum())
            tw = safe_float(self.df['Wastage'].sum())
            tp = safe_float(self.df['Profit'].sum())

            # Avg CPC
            cpc_values = self.df[self.df['CPC'] > 0]['CPC']
            avg_cpc = safe_float(cpc_values.mean()) if len(cpc_values) > 0 else (ts / tc if tc > 0 else 0)

            avg_ctr = safe_float(self.df['CTR'].mean())
            avg_cvr = safe_float(self.df['CVR'].mean())
            avg_cpa = (ts / to) if to > 0 else 0
            avg_acos = (ts / tsa * 100) if tsa > 0 else 0
            avg_roas = (tsa / ts) if ts > 0 else 0
            avg_tcoas = avg_acos

            return {
                'total_spend': ts,
                'total_sales': tsa,
                'total_profit': tp,
                'total_orders': to,
                'total_clicks': tc,
                'total_impressions': ti,
                'total_wastage': tw,
                'roas': avg_roas,
                'acos': avg_acos,
                'tcoas': avg_tcoas,
                'avg_cpc': avg_cpc,
                'avg_ctr': avg_ctr,
                'avg_cvr': avg_cvr,
                'avg_cpa': avg_cpa,
                'conversion_rate': (to / tc * 100) if tc > 0 else 0,
                'keywords_count': len(self.df),
                'campaigns_count': safe_int(self.df['Campaign Name'].nunique()),
                'ad_groups_count': safe_int(self.df['Ad Group Name'].nunique()) if 'Ad Group Name' in self.df.columns else 0,
            }
        except Exception:
            return self._empty_summary()


# =============================================================================
# App
# =============================================================================
def init_session():
    if "clients" not in st.session_state:
        st.session_state.clients = {}
    if "active_client" not in st.session_state:
        st.session_state.active_client = None
    if "debug" not in st.session_state:
        st.session_state.debug = True


def sidebar():
    with st.sidebar:
        st.title("Clients")
        st.session_state.debug = st.checkbox("Show diagnostics", value=st.session_state.debug)

        if st.button("Reset app (clear clients)", use_container_width=True):
            st.session_state.clients = {}
            st.session_state.active_client = None
            st.rerun()

        st.markdown("---")
        client_name = st.text_input("Client name", value="Client 1")
        target_acos = st.number_input("Target ACOS %", value=30.0, step=5.0)
        target_roas = st.number_input("Target ROAS", value=3.0, step=0.5)

        uploaded = st.file_uploader("Upload Search Term report", type=["xlsx", "xls", "csv"])

        if st.button("Create/Update client", use_container_width=True):
            if not uploaded:
                st.error("Upload a file first.")
                return

            try:
                if uploaded.name.lower().endswith(".csv"):
                    raw = pd.read_csv(uploaded)
                else:
                    raw = pd.read_excel(uploaded)

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
                st.success("Client ready.")
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

        if st.session_state.clients:
            st.markdown("---")
            st.caption("Active client")
            st.session_state.active_client = st.selectbox("Select", list(st.session_state.clients.keys()), index=0)


def main():
    load_custom_css()
    init_session()
    sidebar()

    if not st.session_state.active_client:
        st.info("Add a client from the sidebar.")
        return

    client = st.session_state.clients[st.session_state.active_client]
    an = client.analyzer
    if an is None or an.df is None:
        st.error("No analyzer data.")
        return

    s = an.get_client_summary()

    st.subheader("💰 Financial performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spend", format_currency(s["total_spend"]))
    c2.metric("Sales", format_currency(s["total_sales"]))
    c3.metric("ROAS", f"{s['roas']:.2f}x")
    c4.metric("ACOS", f"{s['acos']:.1f}%")
    c5.metric("TCoAS", f"{s['tcoas']:.1f}%")

    st.subheader("📈 Key metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Orders", format_number(s["total_orders"]))
    k2.metric("Clicks", format_number(s["total_clicks"]))
    k3.metric("CTR", f"{s['avg_ctr']:.2f}%")
    k4.metric("CVR", f"{s['avg_cvr']:.2f}%")
    k5.metric("Avg CPC", format_currency(s["avg_cpc"]))

    wastage_pct = (s["total_wastage"] / s["total_spend"] * 100) if s["total_spend"] > 0 else 0
    st.markdown(f"**Wastage:** {format_currency(s['total_wastage'])} ({wastage_pct:.1f}%)")

    if st.session_state.debug:
        st.markdown("### Diagnostics")
        st.write({
            "Sales total": s["total_sales"],
            "Orders total": s["total_orders"],
            "Raw columns sample": list(an.raw_df.columns)[:10] if an.raw_df is not None else [],
            "Note": "This build supports 7/14/30 day Sales/Orders columns.",
        })

    st.markdown("### Data preview")
    preview = an.df[["Customer Search Term", "Campaign Name", "Spend", "Sales", "Orders", "Clicks", "ROAS", "ACOS"]].head(200).copy()
    st.dataframe(add_serial_column(preview), use_container_width=True, hide_index=True, height=520)


if __name__ == "__main__":
    main()
