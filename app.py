#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Ads Dashboard – v1.0
- Fixes truncated Spend/Sales by using custom metric cards (not st.metric).
- Shows TACoS and ideal TACoS benchmarks by product category.
- Explains how TACoS impacts organic rankings and how to move TACoS under 10%.
"""

import math
import re
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Amazon Ads TACoS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple TACoS benchmark ranges by category (guideline values, not strict rules). [web:127][web:129][web:130][web:131]
TACOS_BENCHMARKS = {
    "Electronics": (5, 8, 12),     # aggressive, typical, upper-comfort
    "Clothing / Fashion": (8, 12, 18),
    "Beauty / Personal Care": (10, 15, 20),
    "Grocery / FMCG": (6, 10, 15),
    "Home & Kitchen": (8, 12, 18),
    "Supplements / Health": (10, 15, 22),
    "Generic / Other": (8, 12, 18),
}


# =============================================================================
# STYLES
# =============================================================================

def load_css():
    css = """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #020617 0, #020617 40%, #000000 100%);
    }
    .main {
        padding-top: 0.5rem;
    }
    .app-header {
        background: radial-gradient(circle at top left, #a855f7 0, #1e293b 35%, #020617 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.1rem;
        color: #e5e7eb;
        box-shadow: 0 18px 40px rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.5);
    }
    .app-header h1 { margin: 0; font-size: 1.5rem; }
    .app-header p { margin: 0.35rem 0 0 0; font-size: 0.9rem; color: #e0e7ff; }

    .metric-card {
        background: radial-gradient(circle at top left, rgba(15,23,42,0.95), rgba(15,23,42,0.95));
        border-radius: 14px;
        padding: 0.9rem 1.0rem;
        border: 1px solid rgba(148,163,184,0.55);
        box-shadow: 0 10px 30px rgba(15,23,42,0.85);
        color: #e5e7eb;
        white-space: nowrap;      /* KEY: avoid cutting numbers */
        overflow: visible;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .info-box, .warning-box, .danger-box {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
    }
    .info-box {
        background: radial-gradient(circle at top left, rgba(59,130,246,0.33), rgba(15,23,42,0.98));
        border-left: 4px solid #3b82f6;
        color: #e5e7eb;
    }
    .warning-box {
        background: radial-gradient(circle at top left, rgba(250,204,21,0.33), rgba(77,54,10,0.98));
        border-left: 4px solid #eab308;
        color: #fef9c3;
    }
    .danger-box {
        background: radial-gradient(circle at top left, rgba(248,113,113,0.40), rgba(127,29,29,0.99));
        border-left: 4px solid #ef4444;
        color: #fee2e2;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        border: 1px solid rgba(148,163,184,0.35);
        overflow: hidden;
        box-shadow: 0 14px 36px rgba(15,23,42,0.65);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HELPERS
# =============================================================================

def safe_str(v, default: str = "N/A") -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            return default
        return str(v).strip()
    except Exception:
        return default


def parse_number(v, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(",", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return default
        return float(m.group(0))
    except Exception:
        return default


def to_numeric(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: parse_number(x, 0.0)).astype(float)


def format_inr_full(v) -> str:
    """Full INR with commas, no L/Cr shortening – this is what fixes the cut numbers."""
    x = parse_number(v, 0.0)
    return f"₹{x:,.2f}"


def format_inr_short(v) -> str:
    """Shorter for tables."""
    x = parse_number(v, 0.0)
    if x >= 10_000_000:
        return f"₹{x/10_000_000:.2f}Cr"
    if x >= 100_000:
        return f"₹{x/100_000:.2f}L"
    return f"₹{x:,.0f}"


def format_int(v) -> str:
    x = int(parse_number(v, 0.0))
    if x >= 10_000_000:
        return f"{x/10_000_000:.2f}Cr"
    if x >= 100_000:
        return f"{x/100_000:.2f}L"
    if x >= 1000:
        return f"{x:,}"
    return str(x)


# =============================================================================
# FILE HANDLING
# =============================================================================

def auto_detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = [safe_str(c, "").strip() for c in df.columns]
    lower = [c.lower() for c in cols]
    mapping = {c.lower(): c for c in cols}

    def pick(patterns):
        for p in patterns:
            r = re.compile(p)
            matches = [c for c in lower if r.search(c)]
            if matches:
                return mapping[matches[0]]
        return ""

    return {
        "search_term": pick([r"customer search term", r"search term"]),
        "campaign": pick([r"campaign name"]),
        "clicks": pick([r"^clicks$", r"clicks"]),
        "spend": pick([r"^spend$", r"\bcost\b"]),
        "impressions": pick([r"^impressions$", r"impressions"]),
        "sales": pick(
            [
                r"7 day total sales",
                r"14 day total sales",
                r"30 day total sales",
                r"total sales",
                r"^sales$",
            ]
        ),
        "orders": pick(
            [
                r"7 day total orders",
                r"14 day total orders",
                r"30 day total orders",
                r"ordered units",
                r"^orders$",
            ]
        ),
    }


def read_report(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)

    xls = pd.ExcelFile(uploaded)
    # choose the sheet with "customer search term" etc.
    best_df, best_score = None, -1
    hints = ["customer search term", "search term", "spend", "clicks"]

    for sh in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=sh)
        cols = [str(c).lower() for c in tmp.columns]
        score = sum(any(h in c for c in cols) for h in hints)
        if score > best_score:
            best_df, best_score = tmp, score

    return best_df if best_df is not None else pd.read_excel(uploaded)


# =============================================================================
# CORE LOGIC
# =============================================================================

def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy().dropna(how="all")
    mapping = auto_detect_columns(raw)

    for key in ["search_term", "campaign", "clicks", "spend"]:
        if not mapping.get(key):
            raise ValueError(f"Could not auto‑detect required column for {key!r}")

    df = pd.DataFrame()
    df["Search Term"] = raw[mapping["search_term"]].astype(str).fillna("").str.strip()
    df["Campaign"] = raw[mapping["campaign"]].astype(str).fillna("").str.strip()
    df["Clicks"] = to_numeric(raw[mapping["clicks"]])
    df["Spend"] = to_numeric(raw[mapping["spend"]])
    df["Impressions"] = to_numeric(raw[mapping["impressions"]]) if mapping["impressions"] else 0.0
    df["Sales"] = to_numeric(raw[mapping["sales"]]) if mapping["sales"] else 0.0
    df["Orders"] = to_numeric(raw[mapping["orders"]]) if mapping["orders"] else 0.0

    # basic metrics
    df = df[(df["Clicks"] > 0) | (df["Spend"] > 0)].copy()
    df["CTR"] = df.apply(
        lambda r: (r["Clicks"] / r["Impressions"] * 100) if r["Impressions"] > 0 else 0.0, axis=1
    )
    df["CVR"] = df.apply(
        lambda r: (r["Orders"] / r["Clicks"] * 100) if r["Clicks"] > 0 else 0.0, axis=1
    )
    df["ROAS"] = df.apply(
        lambda r: (r["Sales"] / r["Spend"]) if r["Spend"] > 0 else 0.0, axis=1
    )
    df["ACOS"] = df.apply(
        lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 0.0, axis=1
    )
    df["Wastage"] = df.apply(lambda r: r["Spend"] if r["Orders"] == 0 else 0.0, axis=1)

    return df


def summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    spend = float(df["Spend"].sum())
    sales = float(df["Sales"].sum())
    orders = float(df["Orders"].sum())
    clicks = float(df["Clicks"].sum())
    imps = float(df["Impressions"].sum())
    wastage = float(df["Wastage"].sum())
    roas = sales / spend if spend > 0 else 0.0
    acos = spend / sales * 100 if sales > 0 else 0.0
    ctr = (clicks / imps * 100) if imps > 0 else 0.0
    cpc = spend / clicks if clicks > 0 else 0.0
    return {
        "spend": spend,
        "sales": sales,
        "orders": orders,
        "clicks": clicks,
        "impressions": imps,
        "wastage": wastage,
        "roas": roas,
        "acos": acos,
        "ctr": ctr,
        "cpc": cpc,
    }


# =============================================================================
# PAGES
# =============================================================================

def render_dashboard(df: pd.DataFrame, category: str, total_sales_overall: float):
    stats = summary_stats(df)

    # TACoS: ad spend / total sales (ad + organic) × 100. [web:127][web:130][web:131]
    ad_spend = stats["spend"]
    tacos = ad_spend / total_sales_overall * 100 if total_sales_overall > 0 else 0.0

    st.markdown(
        """
        <div class="app-header">
            <h1>📊 Amazon Ads TACoS Dashboard</h1>
            <p>Full‑value metrics (no truncation) + TACoS benchmarks by category.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("💰 Financial performance")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Spend", format_inr_full(stats["spend"]))
    with c2:
        metric_card("Sales (ad)", format_inr_full(stats["sales"]))
    with c3:
        metric_card("ROAS", f"{stats['roas']:.2f}x")
    with c4:
        metric_card("ACOS", f"{stats['acos']:.1f}%")
    with c5:
        metric_card("TACoS", f"{tacos:.1f}%")

    st.subheader("📈 Key metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Orders", format_int(stats["orders"]))
    with k2:
        metric_card("Clicks", format_int(stats["clicks"]))
    with k3:
        metric_card("Impressions", format_int(stats["impressions"]))
    with k4:
        metric_card("CTR", f"{stats['ctr']:.2f}%")
    with k5:
        metric_card("Avg CPC", format_inr_full(stats["cpc"]))

    wastage_pct = stats["wastage"] / stats["spend"] * 100 if stats["spend"] > 0 else 0.0
    st.markdown(
        f"""
        <div class="danger-box">
        <strong>Wastage (zero‑order spend)</strong><br>
        {format_inr_full(stats["wastage"])} ({wastage_pct:.1f}% of ad spend)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # TACoS benchmarks
    lo, mid, hi = TACOS_BENCHMARKS.get(category, TACOS_BENCHMARKS["Generic / Other"])
    st.markdown(
        f"""
        <div class="info-box">
        <strong>Ideal TACoS benchmarks for {category}</strong><br>
        • Aggressive / very efficient: &lt; {lo}%<br>
        • Healthy range for scaling: {lo} – {mid}%<br>
        • Upper comfort zone: {mid} – {hi}% (beyond this, margin pressure increases). [web:127][web:129][web:130][web:131]
        </div>
        """,
        unsafe_allow_html=True,
    )

    # TACoS → organic rankings explanation. [web:127][web:128][web:129]
    st.markdown(
        """
        <div class="info-box">
        <strong>How TACoS impacts organic rankings</strong><br>
        When your ads generate consistent sales on strong keywords, Amazon's algorithm reads those sales as a signal of relevance, 
        which helps push your organic rankings up for the same queries. A moderate TACoS is acceptable if total profit and organic 
        lift are growing; an increasing TACoS with flat organic sales is a warning sign that ads are not compounding your rank.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Strategies to reach TACoS &lt; 10%. [web:127][web:128][web:130]
    st.markdown(
        """
        <div class="warning-box">
        <strong>Strategies to optimize TACoS under 10%</strong><br>
        • Fix product page: higher CTR & conversion (better images, title, bullets, reviews) reduce ACOS and TACoS.<br>
        • Consolidate spend on top converting search terms and exact‑match campaigns; cut broad, non‑converting traffic.<br>
        • Use negative keywords to block obviously irrelevant queries so budget flows to profitable ones.<br>
        • Lower bids or placement multipliers on poor‑performing placements instead of pausing good keywords.<br>
        • Track TACoS weekly at ASIN level – if TACoS rises while organic sales stay flat, trim bids and test new keywords.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simple wastage table
    st.subheader("🔥 Zero‑order wastage terms")
    wastage_df = df[df["Wastage"] > 0].copy()
    wastage_df = wastage_df.sort_values("Wastage", ascending=False)
    if wastage_df.empty:
        st.info("No zero‑order wastage terms in this report.")
    else:
        view = wastage_df[
            ["Search Term", "Campaign", "Spend", "Clicks", "Impressions", "Sales", "Orders", "ACOS", "ROAS"]
        ].copy()
        view["Spend"] = view["Spend"].apply(format_inr_short)
        view["Sales"] = view["Sales"].apply(format_inr_short)
        view["ACOS"] = view["ACOS"].apply(lambda x: f"{float(x):.1f}%")
        view["ROAS"] = view["ROAS"].apply(lambda x: f"{float(x):.2f}x")
        st.dataframe(view, use_container_width=True, hide_index=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    load_css()

    with st.sidebar:
        st.title("⚙️ Settings")

        uploaded = st.file_uploader(
            "Upload Amazon Search Term report (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
        )

        category = st.selectbox(
            "Product category for TACoS benchmark",
            list(TACOS_BENCHMARKS.keys()),
            index=0,
        )

    if uploaded is None:
        st.markdown(
            """
            <div class="info-box">
            Upload your Sponsored Products Search Term report in the sidebar to see full spend/sales numbers,
            TACoS, and wastage terms.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        raw = read_report(uploaded)
        df = prepare_data(raw)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.code(str(e))
        return

    # Ask for organic sales to compute proper TACoS.
    with st.sidebar:
        est_org = st.number_input(
            "Organic sales in same period (₹)",
            min_value=0.0,
            value=float(df["Sales"].sum()),
            step=100.0,
            format="%.2f",
            help="If you know your organic sales, enter them; else leave equal to ad sales to approximate TACoS.",
        )
        total_sales_overall = float(df["Sales"].sum()) + est_org

    render_dashboard(df, category, total_sales_overall)


if __name__ == "__main__":
    main()
