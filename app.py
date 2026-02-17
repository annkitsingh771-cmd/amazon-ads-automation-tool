#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =============================================================================
# CONFIG + THEME
# =============================================================================
st.set_page_config(page_title="Amazon Ads Dashboard PRO", page_icon="🏢", layout="wide")

CSS = """
<style>
.main {background: radial-gradient(circle at top, #020617 0, #020617 45%, #000 100%); padding-top: 0.5rem;}
.header {
  background: radial-gradient(circle at top left, #a855f7 0, #1e293b 35%, #020617 100%);
  border: 1px solid rgba(148,163,184,0.5);
  box-shadow: 0 18px 50px rgba(15,23,42,0.9);
  color: #e5e7eb; border-radius: 18px; padding: 1.15rem 1.5rem; margin-bottom: 1rem;
}
.header h1{margin:0;font-size:1.5rem;}
.header p{margin:0.35rem 0 0 0;color:#e0e7ff;font-size:0.9rem;}

div[data-testid="stMetric"]{
  background: radial-gradient(circle at top left,#0f172a 0,#020617 55%,#020617 100%);
  border:1px solid #1f2937; border-radius:16px; padding:1.05rem 0.9rem;
  box-shadow:0 16px 40px rgba(15,23,42,0.9);
}
div[data-testid="stMetricLabel"]{color:#e5e7eb !important; font-size:0.82rem !important; white-space:normal !important;}
div[data-testid="stMetricValue"]{color:#f9fafb !important; font-size:1.55rem !important;}

.box {border-radius:12px;padding:0.9rem 1rem;margin:0.8rem 0;font-size:0.92rem;}
.info {background: radial-gradient(circle at top left, rgba(59,130,246,0.30), rgba(15,23,42,0.98)); border-left:4px solid #3b82f6; color:#e5e7eb;}
.warn {background: radial-gradient(circle at top left, rgba(250,204,21,0.30), rgba(77,54,10,0.98)); border-left:4px solid #eab308; color:#fef9c3;}
.danger{background: radial-gradient(circle at top left, rgba(248,113,113,0.40), rgba(127,29,29,0.99)); border-left:4px solid #ef4444; color:#fee2e2;}
.good {background: radial-gradient(circle at top left, rgba(34,197,94,0.35), rgba(6,78,59,0.98)); border-left:4px solid #22c55e; color:#dcfce7;}

div[data-testid="stDataFrame"]{
  border-radius:14px; border: 1px solid rgba(148,163,184,0.35);
  overflow:hidden; box-shadow:0 14px 36px rgba(15,23,42,0.65);
}
div[data-testid="stDataFrame"] thead tr th{
  background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(2,6,23,0.95)) !important;
  color:#f8fafc !important; font-weight:700 !important;
}
div[data-testid="stDataFrame"] tbody tr td{
  background: rgba(2,6,23,0.65) !important;
  color:#e5e7eb !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="header">
      <h1>🏢 Amazon Ads Dashboard PRO</h1>
      <p>Search Term report • Forces Sales/Orders columns • Premium UI • S.No from 1</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================
def parse_number(x, default: float = 0.0) -> float:
    """Robust: handles ₹, commas, Excel formats like '[$₹-en-US]123.4', text etc."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default


def to_num(s: pd.Series) -> pd.Series:
    return s.apply(lambda v: parse_number(v, 0.0)).astype(float)


def money(v) -> str:
    v = parse_number(v, 0.0)
    if v >= 10_000_000:
        return f"₹{v/10_000_000:.2f}Cr"
    if v >= 100_000:
        return f"₹{v/100_000:.2f}L"
    return f"₹{v:,.2f}"


def add_sno(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.reset_index(drop=True).copy()
    out.insert(0, "S.No", range(1, len(out) + 1))
    return out


def read_best_sheet(uploaded_file) -> pd.DataFrame:
    """If multiple sheets, pick the one that looks like a Search Term sheet."""
    xls = pd.ExcelFile(uploaded_file)
    hints = ["customer search term", "campaign name", "spend", "clicks"]
    best_df, best_score, best_rows = None, -1, -1
    for sh in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=sh)
        if tmp is None or len(tmp) == 0:
            continue
        cols = [str(c).lower().strip() for c in tmp.columns]
        score = sum(1 for h in hints if any(h == c or h in c for c in cols))
        if score > best_score or (score == best_score and len(tmp) > best_rows):
            best_df, best_score, best_rows = tmp, score, len(tmp)
    return best_df if best_df is not None else pd.read_excel(uploaded_file)


def find_sales_col(columns: List[str]) -> Optional[str]:
    """Hard priority to your common columns; then fallback by regex."""
    priority = [
        "14 Day Total Sales (₹)",
        "7 Day Total Sales (₹)",
        "30 Day Total Sales (₹)",
        "14 Day Total Sales",
        "7 Day Total Sales",
        "30 Day Total Sales",
        "Total Sales",
        "Sales",
    ]
    for p in priority:
        if p in columns:
            return p

    cols_l = [c.lower() for c in columns]
    for i, c in enumerate(cols_l):
        if "total sales" in c or (c.strip() == "sales"):
            return columns[i]
    return None


def find_orders_col(columns: List[str]) -> Optional[str]:
    priority = [
        "14 Day Total Orders (#)",
        "7 Day Total Orders (#)",
        "30 Day Total Orders (#)",
        "14 Day Total Orders",
        "7 Day Total Orders",
        "30 Day Total Orders",
        "Total Orders",
        "Orders",
    ]
    for p in priority:
        if p in columns:
            return p

    cols_l = [c.lower() for c in columns]
    for i, c in enumerate(cols_l):
        if "total orders" in c or c.strip() == "orders":
            return columns[i]
    return None


def placement_col(columns: List[str]) -> Optional[str]:
    cols_l = [c.lower() for c in columns]
    for i, c in enumerate(cols_l):
        if "placement" in c:
            return columns[i]
    return None


# =============================================================================
# CORE LOGIC
# =============================================================================
def build_dataset(raw: pd.DataFrame, sales_col: str, orders_col: str) -> pd.DataFrame:
    # Required columns in your file
    required = ["Customer Search Term", "Campaign Name", "Spend", "Clicks"]
    miss = [c for c in required if c not in raw.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df = pd.DataFrame()
    df["Customer Search Term"] = raw["Customer Search Term"].astype(str).fillna("").str.strip()
    df["Campaign Name"] = raw["Campaign Name"].astype(str).fillna("").str.strip()
    df["Ad Group Name"] = raw["Ad Group Name"].astype(str).fillna("N/A").str.strip() if "Ad Group Name" in raw.columns else "N/A"
    df["Match Type"] = raw["Match Type"].astype(str).fillna("N/A").str.strip() if "Match Type" in raw.columns else "N/A"

    df["Spend"] = to_num(raw["Spend"])
    df["Clicks"] = to_num(raw["Clicks"])
    df["Impressions"] = to_num(raw["Impressions"]) if "Impressions" in raw.columns else 0.0

    df["Sales"] = to_num(raw[sales_col]) if sales_col else 0.0
    df["Orders"] = to_num(raw[orders_col]) if orders_col else 0.0

    # CPC
    if "Cost Per Click (CPC)" in raw.columns:
        df["CPC"] = to_num(raw["Cost Per Click (CPC)"])
    elif "CPC" in raw.columns:
        df["CPC"] = to_num(raw["CPC"])
    else:
        df["CPC"] = 0.0

    df["CPC"] = df.apply(lambda r: float(r["CPC"]) if float(r["CPC"]) > 0 else (float(r["Spend"]) / float(r["Clicks"]) if float(r["Clicks"]) > 0 else 0.0), axis=1)

    # Keep active rows
    df = df[(df["Spend"] > 0) | (df["Clicks"] > 0)].copy()

    # Derived metrics
    df["ROAS"] = df.apply(lambda r: (r["Sales"] / r["Spend"]) if r["Spend"] > 0 else 0.0, axis=1)
    df["ACOS"] = df.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 0.0, axis=1)
    df["TCoAS"] = df["ACOS"]
    df["CVR"] = df.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r["Clicks"] > 0 else 0.0, axis=1)
    df["CTR"] = df.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r["Impressions"] > 0 else 0.0, axis=1)
    df["Wastage"] = df.apply(lambda r: r["Spend"] if r["Sales"] == 0 else 0.0, axis=1)

    return df


def keyword_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df is None or len(df) == 0:
        return {k: pd.DataFrame() for k in ["Scale", "Test", "Watch", "Reduce", "Pause"]}

    base = df.copy()
    base["Keyword"] = base["Customer Search Term"]

    scale = base[(base["Spend"] >= 20) & (base["Orders"] >= 1) & (base["ROAS"] >= 2.5)]
    pause = base[(base["Spend"] >= 50) & (base["Sales"] == 0) & (base["Clicks"] >= 3)]
    reduce = base[(base["Spend"] >= 30) & (base["ROAS"] < 1.0) & (base["Clicks"] >= 5)]
    test = base[(base["Spend"] >= 20) & (base["ROAS"].between(1.5, 2.49)) & (base["Clicks"] >= 3)]

    used = pd.concat([scale, pause, reduce, test]).drop_duplicates()
    watch = base.drop(index=used.index, errors="ignore")

    def view(dfx):
        if len(dfx) == 0:
            return pd.DataFrame()
        cols = ["Keyword", "Campaign Name", "Ad Group Name", "Match Type", "Spend", "Sales", "Orders", "Clicks", "ROAS", "CVR", "CPC"]
        v = dfx[cols].copy()
        v["Spend"] = v["Spend"].apply(money)
        v["Sales"] = v["Sales"].apply(money)
        v["ROAS"] = v["ROAS"].apply(lambda x: f"{x:.2f}x")
        v["CVR"] = v["CVR"].apply(lambda x: f"{x:.2f}%")
        v["CPC"] = v["CPC"].apply(money)
        return v.sort_values(by="Spend", ascending=False)

    return {"Scale": view(scale), "Test": view(test), "Watch": view(watch), "Reduce": view(reduce), "Pause": view(pause)}


def bids(df: pd.DataFrame, target_roas: float, target_acos: float) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        if r["Spend"] < 20 or r["Clicks"] < 3 or r["CPC"] <= 0:
            continue

        acos = (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 999.0
        action, change, new_bid, reason = "", 0, r["CPC"], ""

        if r["Sales"] == 0 and r["Spend"] >= 50:
            action, change, new_bid, reason = "PAUSE", -100, 0.0, "High spend, zero sales."
        elif r["ROAS"] >= target_roas and r["Orders"] >= 1 and r["CVR"] >= 1:
            action, change, new_bid, reason = "INCREASE", 15, r["CPC"] * 1.15, "Above target ROAS."
        elif r["ROAS"] < 1.5 and r["Spend"] >= 30:
            action, change, new_bid, reason = "REDUCE", -30, r["CPC"] * 0.70, "Low ROAS."
        elif acos > target_acos and r["Spend"] >= 30:
            red = min(30, (acos - target_acos) / target_acos * 100)
            action, change, new_bid, reason = "REDUCE", int(-red), r["CPC"] * (1 - red / 100), f"ACOS {acos:.1f}% above target."

        if action:
            rows.append({
                "Keyword": r["Customer Search Term"],
                "Campaign Name": r["Campaign Name"],
                "Ad Group Name": r["Ad Group Name"],
                "Match Type": r["Match Type"],
                "Spend": money(r["Spend"]),
                "Sales": money(r["Sales"]),
                "Orders": int(r["Orders"]),
                "ROAS": f"{r['ROAS']:.2f}x",
                "CVR": f"{r['CVR']:.2f}%",
                "Current CPC": money(r["CPC"]),
                "Action": action,
                "Change (%)": change,
                "Suggested Bid": money(new_bid) if new_bid > 0 else "₹0.00",
                "Reason": reason,
            })

    return pd.DataFrame(rows)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.subheader("Settings")
    debug = st.checkbox("Show diagnostics", value=True)
    target_roas = st.number_input("Target ROAS", value=3.0, step=0.5)
    target_acos = st.number_input("Target ACOS %", value=30.0, step=5.0)

    st.markdown("---")
    uploaded = st.file_uploader("Upload Search Term report", type=["xlsx", "xls", "csv"])

if not uploaded:
    st.markdown('<div class="box info">Upload your Search Term report from the sidebar.</div>', unsafe_allow_html=True)
    st.stop()


# =============================================================================
# LOAD FILE
# =============================================================================
if uploaded.name.lower().endswith(".csv"):
    raw = pd.read_csv(uploaded)
else:
    raw = read_best_sheet(uploaded)

raw.columns = [str(c).strip() for c in raw.columns]
cols = list(raw.columns)

sales_col = find_sales_col(cols)
orders_col = find_orders_col(cols)

if not sales_col or not orders_col:
    st.markdown('<div class="box danger"><b>Sales/Orders column not found.</b><br>Please upload correct Search Term report.</div>', unsafe_allow_html=True)
    st.write("Columns:", cols)
    st.stop()

# Build dataset
df = build_dataset(raw, sales_col=sales_col, orders_col=orders_col)

# =============================================================================
# SANITY CHECK (IMPORTANT)
# =============================================================================
spend_total = float(df["Spend"].sum())
sales_total = float(df["Sales"].sum())
orders_total = int(df["Orders"].sum())
clicks_total = int(df["Clicks"].sum())

if debug:
    st.markdown(
        f"""
        <div class="box info">
        <b>Diagnostics</b><br>
        Sales column: <code>{sales_col}</code><br>
        Orders column: <code>{orders_col}</code><br>
        Spend sum: <code>{spend_total:.2f}</code><br>
        Sales sum: <code>{sales_total:.2f}</code><br>
        Orders sum: <code>{orders_total}</code><br>
        Clicks sum: <code>{clicks_total}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# DASHBOARD
# =============================================================================
roas = (sales_total / spend_total) if spend_total > 0 else 0.0
acos = (spend_total / sales_total * 100) if sales_total > 0 else 0.0
tcoas = acos
ctr = (df["Clicks"].sum() / df["Impressions"].sum() * 100) if df["Impressions"].sum() > 0 else 0.0
cvr = (orders_total / clicks_total * 100) if clicks_total > 0 else 0.0
avg_cpc = (spend_total / clicks_total) if clicks_total > 0 else 0.0

st.subheader("💰 Financial performance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spend", money(spend_total))
c2.metric("Sales", money(sales_total))
c3.metric("ROAS", f"{roas:.2f}x")
c4.metric("ACOS", f"{acos:.2f}%")
c5.metric("TCoAS", f"{tcoas:.2f}%")

st.subheader("📈 Key metrics")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Orders", f"{orders_total:,}")
k2.metric("Clicks", f"{clicks_total:,}")
k3.metric("CTR", f"{ctr:.2f}%")
k4.metric("CVR", f"{cvr:.2f}%")
k5.metric("Avg CPC", money(avg_cpc))

wastage = float(df.loc[df["Sales"] == 0, "Spend"].sum())
waste_pct = (wastage / spend_total * 100) if spend_total > 0 else 0.0
st.markdown(
    f'<div class="box danger"><b>Wastage (zero-sales spend)</b><br>{money(wastage)} ({waste_pct:.1f}%)</div>',
    unsafe_allow_html=True,
)

# =============================================================================
# GROUPS
# =============================================================================
st.subheader("🎯 Keyword groups (Scale / Test / Watch / Reduce / Pause)")
groups = keyword_groups(df)
tabs = st.tabs([
    f"🏆 Scale ({len(groups['Scale'])})",
    f"⚡ Test ({len(groups['Test'])})",
    f"👀 Watch ({len(groups['Watch'])})",
    f"⚠️ Reduce ({len(groups['Reduce'])})",
    f"🚨 Pause ({len(groups['Pause'])})",
])

for tab, key in zip(tabs, ["Scale", "Test", "Watch", "Reduce", "Pause"]):
    with tab:
        if len(groups[key]) == 0:
            st.markdown('<div class="box info">No rows here.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(add_sno(groups[key]), use_container_width=True, hide_index=True, height=520)

# =============================================================================
# BID SUGGESTIONS
# =============================================================================
st.subheader("💡 Bid suggestions")
sug = bids(df, target_roas=float(target_roas), target_acos=float(target_acos))
if len(sug) == 0:
    st.markdown('<div class="box info">No bid actions found (need enough spend/clicks or signals).</div>', unsafe_allow_html=True)
else:
    st.dataframe(add_sno(sug), use_container_width=True, hide_index=True, height=520)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as wr:
        sug.to_excel(wr, index=False, sheet_name="Bid Suggestions")
    out.seek(0)

    st.download_button(
        "Download Bid Suggestions (XLSX)",
        data=out,
        file_name=f"Bid_Suggestions_{datetime.now().strftime('%Y%m%d')}.xlsx",
        use_container_width=True,
    )

# =============================================================================
# PLACEMENT RECOMMENDATION (only if placement column exists)
# =============================================================================
st.subheader("📍 Placement recommendations")
pcol = placement_col(cols)
if not pcol:
    st.markdown('<div class="box info">Search Term report usually has no placement column. Upload a Placement report to see placement recommendations.</div>', unsafe_allow_html=True)
else:
    tmp = raw.copy()
    tmp[pcol] = tmp[pcol].astype(str).fillna("UNKNOWN").str.strip()
    tmp_sp = to_num(tmp["Spend"])
    tmp_sales = to_num(tmp[sales_col])
    g = pd.DataFrame({"Placement": tmp[pcol], "Spend": tmp_sp, "Sales": tmp_sales}).groupby("Placement", as_index=False).sum()
    g["ROAS"] = g.apply(lambda r: (r["Sales"] / r["Spend"]) if r["Spend"] > 0 else 0.0, axis=1)
    g["ACOS"] = g.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 0.0, axis=1)

    view = g.copy()
    view["Spend"] = view["Spend"].apply(money)
    view["Sales"] = view["Sales"].apply(money)
    view["ROAS"] = view["ROAS"].apply(lambda x: f"{x:.2f}x")
    view["ACOS"] = view["ACOS"].apply(lambda x: f"{x:.2f}%")

    st.dataframe(add_sno(view.sort_values("Spend", ascending=False)), use_container_width=True, hide_index=True, height=320)
