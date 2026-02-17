#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import traceback
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Amazon Ads Dashboard", page_icon="🏢", layout="wide")
st.title("Amazon Ads Dashboard")


# ------------------------------------------------------------
# Helpers (robust parsing)
# ------------------------------------------------------------
def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
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

        s = s.replace(",", "")
        s = s.replace("₹", "").replace("$", "").replace("%", "")
        s = s.replace("INR", "").replace("Rs.", "").replace("Rs", "")

        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(round(safe_float(value, default)))
    except Exception:
        return default


def format_currency(v):
    v = safe_float(v, 0)
    if v >= 10000000:
        return f"₹{v/10000000:.2f}Cr"
    if v >= 100000:
        return f"₹{v/100000:.2f}L"
    return f"₹{v:,.2f}"


def parse_num_series(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: safe_float(x, 0.0)).astype(float)


def find_col(df: pd.DataFrame, patterns: List[str]):
    cols = list(df.columns)
    cols_l = [str(c).lower().strip() for c in cols]
    for pat in patterns:
        rx = re.compile(pat)
        for i, c in enumerate(cols_l):
            if rx.search(c):
                return cols[i]
    return None


def detect_sales_orders_cols(df: pd.DataFrame):
    sales_col = find_col(df, [
        r"14\s*day\s*total\s*sales",
        r"7\s*day\s*total\s*sales",
        r"30\s*day\s*total\s*sales",
        r"total\s*sales",
        r"^sales$",
        r"revenue",
    ])
    orders_col = find_col(df, [
        r"14\s*day\s*total\s*orders",
        r"7\s*day\s*total\s*orders",
        r"30\s*day\s*total\s*orders",
        r"total\s*orders",
        r"^orders$",
        r"units",
    ])
    return sales_col, orders_col


def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.reset_index(drop=True).copy()
    out.insert(0, "S.No", range(1, len(out) + 1))
    return out


# ------------------------------------------------------------
# Safe file reading (won’t crash on missing openpyxl)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_report(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    # Excel: prefer openpyxl (most common)
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        # last attempt
        try:
            return pd.read_excel(uploaded_file)
        except Exception as e:
            raise RuntimeError("Excel read failed. Install: pip install openpyxl") from e


# ------------------------------------------------------------
# Analysis (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analyze(df: pd.DataFrame) -> Dict:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lowmap = {c.lower(): c for c in df.columns}

    def col_ci(*names):
        for n in names:
            c = lowmap.get(n.lower())
            if c:
                return c
        return None

    # Required fields
    spend_col = col_ci("Spend", "Cost", "Ad Spend")
    clicks_col = col_ci("Clicks")
    cst_col = col_ci("Customer Search Term", "Search Term", "Search term")
    camp_col = col_ci("Campaign Name", "Campaign")

    if not spend_col or not clicks_col or not cst_col or not camp_col:
        raise ValueError(f"Missing required columns. Available: {list(df.columns)}")

    sales_col, orders_col = detect_sales_orders_cols(df)
    if not sales_col or not orders_col:
        raise ValueError("Sales/Orders column not found (7/14/30 day).")

    imps_col = col_ci("Impressions", "Imps")

    spend = parse_num_series(df[spend_col])
    clicks = parse_num_series(df[clicks_col])
    sales = parse_num_series(df[sales_col])
    orders = parse_num_series(df[orders_col])
    imps = parse_num_series(df[imps_col]) if imps_col else pd.Series([0.0] * len(df))

    total_spend = float(spend.sum())
    total_sales = float(sales.sum())
    total_clicks = int(clicks.sum())
    total_orders = int(orders.sum())
    total_imps = int(imps.sum())

    roas = (total_sales / total_spend) if total_spend > 0 else 0.0
    acos = (total_spend / total_sales * 100) if total_sales > 0 else 0.0
    ctr = (total_clicks / total_imps * 100) if total_imps > 0 else 0.0
    cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0.0
    avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0.0
    wastage = float(spend[sales == 0].sum())
    wastage_pct = (wastage / total_spend * 100) if total_spend > 0 else 0.0

    preview_cols = [cst_col, camp_col, spend_col, sales_col, orders_col, clicks_col]
    if imps_col:
        preview_cols.append(imps_col)

    preview = df[preview_cols].head(500).copy()

    return {
        "sales_col": sales_col,
        "orders_col": orders_col,
        "spend_col": spend_col,
        "clicks_col": clicks_col,
        "imps_col": imps_col,
        "cst_col": cst_col,
        "camp_col": camp_col,
        "rows": int(len(df)),
        "total_spend": total_spend,
        "total_sales": total_sales,
        "total_clicks": total_clicks,
        "total_orders": total_orders,
        "total_impressions": total_imps,
        "roas": roas,
        "acos": acos,
        "ctr": ctr,
        "cvr": cvr,
        "avg_cpc": avg_cpc,
        "wastage": wastage,
        "wastage_pct": wastage_pct,
        "preview": preview,
    }


# ------------------------------------------------------------
# UI (reboot-safe)
# ------------------------------------------------------------
st.caption("Status: app started successfully. Upload a file only when needed.")

with st.sidebar:
    st.header("Run")
    uploaded = st.file_uploader("Upload Search Term report", type=["csv", "xlsx", "xls"])
    run_btn = st.button("Run analysis", use_container_width=True)
    show_diag = st.checkbox("Diagnostics", value=False)

if uploaded is None:
    st.info("No file uploaded. App is idle (fast boot).")
    st.stop()

if not run_btn:
    st.warning("Click 'Run analysis' to start (prevents auto rerun loops).")
    st.stop()

try:
    with st.spinner("Loading & processing…"):
        df = load_report(uploaded)
        result = analyze(df)

    st.success(f"Detected: Sales = {result['sales_col']} | Orders = {result['orders_col']}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spend", format_currency(result["total_spend"]))
    c2.metric("Sales", format_currency(result["total_sales"]))
    c3.metric("ROAS", f"{result['roas']:.2f}x")
    c4.metric("ACOS", f"{result['acos']:.2f}%")
    c5.metric("Wastage", f"{format_currency(result['wastage'])} ({result['wastage_pct']:.1f}%)")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Orders", f"{result['total_orders']:,}")
    k2.metric("Clicks", f"{result['total_clicks']:,}")
    k3.metric("Impressions", f"{result['total_impressions']:,}")
    k4.metric("CTR", f"{result['ctr']:.2f}%")
    k5.metric("CVR", f"{result['cvr']:.2f}%")

    st.subheader("Preview (first 500 rows)")
    st.dataframe(add_serial_column(result["preview"]), use_container_width=True, hide_index=True, height=520)

    if show_diag:
        st.subheader("Diagnostics")
        st.write({k: v for k, v in result.items() if k != "preview"})
        st.write("All columns:", list(df.columns))

except Exception as e:
    st.error(str(e))
    st.code(traceback.format_exc())
