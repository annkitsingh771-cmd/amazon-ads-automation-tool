#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Amazon Ads Dashboard Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Amazon Ads Dashboard Pro")
st.caption("If you can see this message, the app started correctly (no crash).")


# -----------------------------
# Helpers
# -----------------------------
def safe_float(value, default: float = 0.0) -> float:
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

        # negatives like (123)
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        s = s.replace(",", "")
        s = s.replace("₹", "").replace("$", "").replace("%", "")
        s = s.replace("INR", "").replace("Rs.", "").replace("Rs", "")

        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(round(safe_float(value, default)))
    except Exception:
        return default


def format_currency(value) -> str:
    v = safe_float(value, 0.0)
    if v >= 10_000_000:
        return f"₹{v/10_000_000:.2f}Cr"
    if v >= 100_000:
        return f"₹{v/100_000:.2f}L"
    return f"₹{v:,.2f}"


def format_number(value) -> str:
    v = safe_int(value, 0)
    if v >= 10_000_000:
        return f"{v/10_000_000:.2f}Cr"
    if v >= 100_000:
        return f"{v/100_000:.2f}L"
    if v >= 1_000:
        return f"{v:,}"
    return str(v)


def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.reset_index(drop=True).copy()
    out.insert(0, "S.No", range(1, len(out) + 1))
    return out


def find_first_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_l = [str(c).lower().strip() for c in cols]
    for pat in patterns:
        rx = re.compile(pat)
        for i, c in enumerate(cols_l):
            if rx.search(c):
                return cols[i]
    return None


def parse_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: safe_float(x, 0.0)).astype(float)
    return out


# -----------------------------
# Loaders (BOOT SAFE)
# -----------------------------
def can_read_excel() -> bool:
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
        # Avoid crash if openpyxl not installed
        if not can_read_excel():
            raise RuntimeError("Excel upload needs openpyxl. Install it or upload CSV.")
        return pd.read_excel(uploaded_file, engine="openpyxl")

    raise RuntimeError("Unsupported file type.")


# -----------------------------
# Analyzer (simple + reliable)
# -----------------------------
@st.cache_data(show_spinner=False)
def prepare_dataframe(df: pd.DataFrame) -> Dict:
    if df is None or len(df) == 0:
        raise ValueError("Empty file")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Map core columns (case-insensitive)
    lowmap = {c.lower(): c for c in df.columns}

    def col_ci(*names):
        for n in names:
            c = lowmap.get(n.lower())
            if c:
                return c
        return None

    cst = col_ci("Customer Search Term", "Search Term", "Keyword")
    camp = col_ci("Campaign Name", "Campaign")
    spend = col_ci("Spend", "Cost", "Ad Spend")
    clicks = col_ci("Clicks")
    imps = col_ci("Impressions", "Imps")

    if not cst or not camp or not spend or not clicks:
        raise ValueError(f"Missing required columns. Available: {list(df.columns)}")

    # Detect sales/orders across 7/14/30
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

    # Extra fallback (some reports split sales)
    adv_sales = find_first_col(df, [r"advertised\s*sku\s*sales"])
    other_sales = find_first_col(df, [r"other\s*sku\s*sales"])

    if not sales_col and (adv_sales or other_sales):
        sales_col = "__computed_sales__"
        df[sales_col] = 0

    if not orders_col:
        raise ValueError("Orders column not found (7/14/30 day).")

    # Create normalized columns
    df_norm = df.rename(columns={
        cst: "Customer Search Term",
        camp: "Campaign Name",
        spend: "Spend",
        clicks: "Clicks",
    }).copy()

    if imps:
        df_norm.rename(columns={imps: "Impressions"}, inplace=True)
    else:
        df_norm["Impressions"] = 0

    # Build Sales
    if sales_col == "__computed_sales__":
        df_norm["Sales_Advertised"] = df[adv_sales] if adv_sales else 0
        df_norm["Sales_Other"] = df[other_sales] if other_sales else 0
        df_norm = parse_numeric_columns(df_norm, ["Sales_Advertised", "Sales_Other"])
        df_norm["Sales"] = df_norm["Sales_Advertised"] + df_norm["Sales_Other"]
        detected_sales = f"Computed from: {adv_sales or '0'} + {other_sales or '0'}"
    else:
        df_norm["Sales"] = df[sales_col] if sales_col else 0
        detected_sales = sales_col or "Not found (set to 0)"

    df_norm["Orders"] = df[orders_col]
    detected_orders = orders_col

    # Numeric convert
    df_norm = parse_numeric_columns(df_norm, ["Spend", "Clicks", "Impressions", "Sales", "Orders"])

    # Filter active
    df_norm = df_norm[(df_norm["Spend"] > 0) | (df_norm["Clicks"] > 0)].copy()
    if len(df_norm) == 0:
        raise ValueError("No active rows after filtering (Spend/Clicks all 0).")

    # Metrics
    df_norm["CPC"] = df_norm.apply(
        lambda x: (x["Spend"] / x["Clicks"]) if x["Clicks"] > 0 else 0.0,
        axis=1
    )
    df_norm["ROAS"] = df_norm.apply(
        lambda x: (x["Sales"] / x["Spend"]) if x["Spend"] > 0 else 0.0,
        axis=1
    )
    df_norm["ACOS"] = df_norm.apply(
        lambda x: (x["Spend"] / x["Sales"] * 100) if x["Sales"] > 0 else 0.0,
        axis=1
    )
    df_norm["CTR"] = df_norm.apply(
        lambda x: (x["Clicks"] / x["Impressions"] * 100) if x["Impressions"] > 0 else 0.0,
        axis=1
    )
    df_norm["CVR"] = df_norm.apply(
        lambda x: (x["Orders"] / x["Clicks"] * 100) if x["Clicks"] > 0 else 0.0,
        axis=1
    )

    summary = {
        "rows": int(len(df_norm)),
        "total_spend": float(df_norm["Spend"].sum()),
        "total_sales": float(df_norm["Sales"].sum()),
        "total_orders": int(df_norm["Orders"].sum()),
        "total_clicks": int(df_norm["Clicks"].sum()),
        "total_impressions": int(df_norm["Impressions"].sum()),
    }
    summary["roas"] = (summary["total_sales"] / summary["total_spend"]) if summary["total_spend"] > 0 else 0.0
    summary["acos"] = (summary["total_spend"] / summary["total_sales"] * 100) if summary["total_sales"] > 0 else 0.0
    summary["avg_cpc"] = (summary["total_spend"] / summary["total_clicks"]) if summary["total_clicks"] > 0 else 0.0

    wastage = float(df_norm.loc[df_norm["Sales"] == 0, "Spend"].sum())
    wastage_pct = (wastage / summary["total_spend"] * 100) if summary["total_spend"] > 0 else 0.0

    return {
        "df": df_norm,
        "summary": summary,
        "detected_sales": detected_sales,
        "detected_orders": detected_orders,
        "wastage": wastage,
        "wastage_pct": wastage_pct,
    }


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Upload")
    st.write("Tip: CSV opens faster and avoids Excel dependency issues.")
    uploaded = st.file_uploader("Upload Search Term report", type=["csv", "xlsx", "xls"])
    run_btn = st.button("Run", use_container_width=True)
    diag = st.checkbox("Show diagnostics", value=False)

# Idle state (fast reboot)
if uploaded is None:
    st.info("No file uploaded. App is idle and should open instantly.")
    st.stop()

# Prevent auto-rerun heavy work
if not run_btn:
    st.warning("Click Run to start processing.")
    st.stop()

# -----------------------------
# Run
# -----------------------------
try:
    with st.spinner("Loading…"):
        raw = load_report(uploaded)

    with st.spinner("Analyzing…"):
        out = prepare_dataframe(raw)

    df_norm = out["df"]
    s = out["summary"]

    st.success(f"Sales column: {out['detected_sales']} | Orders column: {out['detected_orders']}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spend", format_currency(s["total_spend"]))
    c2.metric("Sales", format_currency(s["total_sales"]))
    c3.metric("ROAS", f"{s['roas']:.2f}x")
    c4.metric("ACOS", f"{s['acos']:.2f}%")
    c5.metric("Avg CPC", format_currency(s["avg_cpc"]))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders", format_number(s["total_orders"]))
    k2.metric("Clicks", format_number(s["total_clicks"]))
    k3.metric("Impressions", format_number(s["total_impressions"]))
    k4.metric("Wastage", f"{format_currency(out['wastage'])} ({out['wastage_pct']:.1f}%)")

    st.subheader("Preview")
    show_cols = ["Customer Search Term", "Campaign Name", "Spend", "Sales", "Orders", "Clicks", "Impressions", "CPC", "ROAS", "ACOS", "CVR", "CTR"]
    show_cols = [c for c in show_cols if c in df_norm.columns]
    st.dataframe(add_serial_column(df_norm[show_cols].head(500)), use_container_width=True, hide_index=True, height=560)

    if diag:
        st.subheader("Diagnostics")
        st.write("Rows:", len(raw))
        st.write("Columns:", list(raw.columns))

except Exception as e:
    st.error(str(e))
    st.code(traceback.format_exc())
