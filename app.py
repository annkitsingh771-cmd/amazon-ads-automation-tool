#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Amazon Ads Dashboard", layout="wide")
st.title("Amazon Ads Dashboard")
st.caption("If you can see this, the app started successfully.")


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

        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        s = s.replace(",", "")
        s = s.replace("₹", "").replace("$", "").replace("%", "")
        s = s.replace("INR", "").replace("Rs.", "").replace("Rs", "")

        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default


def parse_num_series(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: safe_float(x, 0.0)).astype(float)


def find_first_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_l = [str(c).lower().strip() for c in cols]
    for pat in patterns:
        rx = re.compile(pat)
        for i, c in enumerate(cols_l):
            if rx.search(c):
                return cols[i]
    return None


def format_currency(v):
    v = safe_float(v, 0.0)
    if v >= 10_000_000:
        return f"₹{v/10_000_000:.2f}Cr"
    if v >= 100_000:
        return f"₹{v/100_000:.2f}L"
    return f"₹{v:,.2f}"


def can_import_openpyxl() -> bool:
    try:
        import openpyxl  # noqa: F401
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_report(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith((".xlsx", ".xls")):
        if not can_import_openpyxl():
            raise RuntimeError("Excel needs openpyxl. Add openpyxl in requirements.txt or upload CSV.")
        return pd.read_excel(uploaded_file, engine="openpyxl")

    raise RuntimeError("Unsupported file type. Upload CSV or XLSX.")


@st.cache_data(show_spinner=False)
def analyze(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lowmap = {c.lower(): c for c in df.columns}

    def col_ci(*names):
        for n in names:
            c = lowmap.get(n.lower())
            if c:
                return c
        return None

    spend_col = col_ci("Spend", "Cost", "Ad Spend")
    clicks_col = col_ci("Clicks")
    cst_col = col_ci("Customer Search Term", "Search Term", "Keyword")
    camp_col = col_ci("Campaign Name", "Campaign")
    imps_col = col_ci("Impressions", "Imps")

    if not spend_col or not clicks_col or not cst_col or not camp_col:
        raise ValueError(f"Missing required columns. Available: {list(df.columns)}")

    sales_col = find_first_col(df, [
        r"14\s*day\s*total\s*sales",
        r"7\s*day\s*total\s*sales",
        r"30\s*day\s*total\s*sales",
        r"total\s*sales",
        r"^sales$",
        r"revenue",
    ])
    orders_col = find_first_col(df, [
        r"14\s*day\s*total\s*orders",
        r"7\s*day\s*total\s*orders",
        r"30\s*day\s*total\s*orders",
        r"total\s*orders",
        r"^orders$",
        r"units",
    ])

    if not sales_col:
        raise ValueError("Sales column not found (7/14/30 day).")
    if not orders_col:
        raise ValueError("Orders column not found (7/14/30 day).")

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
    avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0.0

    preview = df[[cst_col, camp_col, spend_col, sales_col, orders_col, clicks_col]].head(300).copy()

    return {
        "sales_col": sales_col,
        "orders_col": orders_col,
        "total_spend": total_spend,
        "total_sales": total_sales,
        "total_clicks": total_clicks,
        "total_orders": total_orders,
        "total_impressions": total_imps,
        "roas": roas,
        "acos": acos,
        "avg_cpc": avg_cpc,
        "preview": preview,
    }


with st.sidebar:
    st.header("Upload")
    st.write("If your app crashes, check: Manage app → Logs.")
    uploaded = st.file_uploader("Upload report", type=["csv", "xlsx", "xls"])
    run_btn = st.button("Run", use_container_width=True)

if uploaded is None:
    st.info("No upload yet. App is idle (fast boot).")
    st.stop()

if not run_btn:
    st.warning("Click Run to start processing.")
    st.stop()

try:
    with st.spinner("Loading…"):
        df = load_report(uploaded)
    with st.spinner("Analyzing…"):
        r = analyze(df)

    st.success(f"Detected Sales: {r['sales_col']} | Orders: {r['orders_col']}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spend", format_currency(r["total_spend"]))
    c2.metric("Sales", format_currency(r["total_sales"]))
    c3.metric("ROAS", f"{r['roas']:.2f}x")
    c4.metric("ACOS", f"{r['acos']:.2f}%")
    c5.metric("Avg CPC", format_currency(r["avg_cpc"]))

    st.subheader("Preview")
    st.dataframe(r["preview"], use_container_width=True, hide_index=True)

except Exception as e:
    st.error(str(e))
    st.code(traceback.format_exc())
