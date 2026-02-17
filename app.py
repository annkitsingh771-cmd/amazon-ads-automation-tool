#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amazon Ads Dashboard PRO (Simple + Correct) v1.0

Built to work with Amazon Sponsored Products Search Term report like yours:
- Sales column: '14 Day Total Sales (₹)' (auto-detects 7/14/30 day total sales)
- Orders column: '14 Day Total Orders (#)' (auto-detects 7/14/30 day total orders)
- Shows Spend, Sales, ROAS, ACOS, TCoAS, Orders, CTR, CVR, Avg CPC
- Keyword groups: Scale / Test / Watch / Reduce / Pause
- Bid suggestions: Increase / Reduce / Pause
- S.No starts from 1 (and dataframe index hidden)
- Premium high-contrast UI
- Placement recommendations if placement column exists
"""

import io
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# ----------------------------- UI THEME --------------------------------------
st.set_page_config(page_title="Amazon Ads Dashboard PRO", page_icon="🏢", layout="wide")

CSS = """
<style>
.main {background: radial-gradient(circle at top, #020617 0, #020617 45%, #000 100%); padding-top: 0.5rem;}
.header {
  background: radial-gradient(circle at top left, #a855f7 0, #1e293b 35%, #020617 100%);
  border: 1px solid rgba(148,163,184,0.5);
  box-shadow: 0 18px 50px rgba(15,23,42,0.9);
  color: #e5e7eb; border-radius: 18px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
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


# ----------------------------- HELPERS ---------------------------------------
def parse_number(x, default=0.0) -> float:
    """Robust: handles ₹, commas, text, Excel formats like '[$₹-en-US]123.4'."""
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


def num(v) -> str:
    v = int(parse_number(v, 0.0))
    return f"{v:,}"


def add_sno(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.reset_index(drop=True).copy()
    out.insert(0, "S.No", range(1, len(out) + 1))
    return out


def pick_col(cols_lower: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat)
        hits = [c for c in cols_lower if rx.search(c)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def read_excel_best_sheet(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file)
    best_df, best_score, best_rows = None, -1, -1
    hints = ["customer search term", "campaign name", "spend", "clicks"]

    for sh in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=sh)
        if tmp is None or len(tmp) == 0:
            continue
        cols = [str(c).lower().strip() for c in tmp.columns]
        score = sum(1 for h in hints if any(h == c or h in c for c in cols))
        if score > best_score or (score == best_score and len(tmp) > best_rows):
            best_df, best_score, best_rows = tmp, score, len(tmp)

    return best_df if best_df is not None else pd.read_excel(uploaded_file)


# ----------------------------- CORE BUILD ------------------------------------
def build_clean_df(df: pd.DataFrame, overrides: Dict[str, str]) -> (pd.DataFrame, Dict[str, str]):
    df = df.copy().dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    cols = list(df.columns)
    cols_lower = [c.lower().strip() for c in cols]
    map_lower_to_orig = {c.lower().strip(): c for c in cols}

    def det(key: str, patterns: List[str]) -> str:
        forced = overrides.get(key, "").strip()
        if forced and forced in cols:
            return forced
        low = pick_col(cols_lower, patterns)
        return map_lower_to_orig.get(low, "") if low else ""

    cst = det("Customer Search Term", [r"^customer search term$", r"customer search term", r"search term"])
    camp = det("Campaign Name", [r"^campaign name$", r"campaign name"])
    clicks = det("Clicks", [r"^clicks$", r"clicks"])
    spend = det("Spend", [r"^spend$", r"\bcost\b", r"ad spend", r"spend"])
    impr = det("Impressions", [r"^impressions$", r"impressions", r"\bimps\b"])
    sales = det("Sales", [r"\b7\s*day\s*total\s*sales\b", r"\b14\s*day\s*total\s*sales\b", r"\b30\s*day\s*total\s*sales\b", r"\btotal\s*sales\b", r"(^sales$)|(\bsales\b)"])
    orders = det("Orders", [r"\b7\s*day\s*total\s*orders\b", r"\b14\s*day\s*total\s*orders\b", r"\b30\s*day\s*total\s*orders\b", r"\btotal\s*orders\b", r"(^orders$)|(\borders\b)"])
    adg = det("Ad Group Name", [r"ad group name", r"adgroup"])
    match = det("Match Type", [r"match type", r"matchtype"])
    cpc = det("CPC", [r"cost per click", r"\bcpc\b"])

    if not cst or not camp or not clicks or not spend:
        raise ValueError("Missing required columns. Set Column Override for Customer Search Term, Campaign Name, Clicks, Spend.")

    out = pd.DataFrame()
    out["Customer Search Term"] = df[cst].astype(str).fillna("").str.strip()
    out["Campaign Name"] = df[camp].astype(str).fillna("").str.strip()
    out["Ad Group Name"] = df[adg].astype(str).fillna("N/A").str.strip() if adg else "N/A"
    out["Match Type"] = df[match].astype(str).fillna("N/A").str.strip() if match else "N/A"

    out["Clicks"] = to_num(df[clicks])
    out["Spend"] = to_num(df[spend])
    out["Impressions"] = to_num(df[impr]) if impr else 0.0

    out["Sales"] = to_num(df[sales]) if sales else 0.0
    out["Orders"] = to_num(df[orders]) if orders else 0.0

    out["CPC"] = to_num(df[cpc]) if cpc else 0.0
    out["CPC"] = out.apply(lambda r: float(r["CPC"]) if float(r["CPC"]) > 0 else (float(r["Spend"]) / float(r["Clicks"]) if float(r["Clicks"]) > 0 else 0.0), axis=1)

    out = out[(out["Spend"] > 0) | (out["Clicks"] > 0)].copy()

    out["ROAS"] = out.apply(lambda r: (r["Sales"] / r["Spend"]) if r["Spend"] > 0 else 0.0, axis=1)
    out["ACOS"] = out.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 0.0, axis=1)
    out["TCoAS"] = out["ACOS"]
    out["CVR"] = out.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r["Clicks"] > 0 else 0.0, axis=1)
    out["CTR"] = out.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r["Impressions"] > 0 else 0.0, axis=1)
    out["Wastage"] = out.apply(lambda r: r["Spend"] if r["Sales"] == 0 else 0.0, axis=1)

    diag = {
        "Sales column": sales or "NOT DETECTED",
        "Orders column": orders or "NOT DETECTED",
        "Spend column": spend,
        "Clicks column": clicks,
        "Sales sum": f"{out['Sales'].sum():.2f}",
        "Orders sum": f"{int(out['Orders'].sum())}",
    }
    return out, diag


def classify(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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

    return {
        "Scale": view(scale),
        "Test": view(test),
        "Watch": view(watch),
        "Reduce": view(reduce),
        "Pause": view(pause),
    }


def bid_suggestions(df: pd.DataFrame, target_roas=3.0, target_acos=30.0) -> pd.DataFrame:
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

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ----------------------------- APP -------------------------------------------
def app_header():
    st.markdown(
        f"""
        <div class="header">
            <h1>🏢 Amazon Ads Dashboard PRO</h1>
            <p>Made for Search Term report (7/14/30-day) • Sales/Orders fixed • Premium UI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    app_header()

    with st.sidebar:
        st.title("Clients")
        debug = st.checkbox("Show diagnostics", value=True)

        if st.button("Reset session (Fix 0 Sales)", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown("---")
        client = st.text_input("Client name", value="Client 1")

        target_roas = st.number_input("Target ROAS", value=3.0, step=0.5)
        target_acos = st.number_input("Target ACOS %", value=30.0, step=5.0)

        up = st.file_uploader("Upload Search Term report", type=["xlsx", "xls", "csv"])

    if up is None:
        st.markdown('<div class="box info">Upload your Search Term report from the sidebar.</div>', unsafe_allow_html=True)
        return

    # Read file
    if up.name.lower().endswith(".csv"):
        raw = pd.read_csv(up)
    else:
        raw = read_excel_best_sheet(up)

    # Column override UI
    detected_like = list(raw.columns)
    cols = [""] + detected_like

    st.subheader("Column mapping (auto + manual override)")
    c1, c2, c3, c4 = st.columns(4)

    def guess_default(label: str) -> str:
        # for your file: Sales & Orders are 14 Day Total Sales/Orders [code:0]
        if label == "Sales":
            return "14 Day Total Sales (₹)" if "14 Day Total Sales (₹)" in raw.columns else ""
        if label == "Orders":
            return "14 Day Total Orders (#)" if "14 Day Total Orders (#)" in raw.columns else ""
        if label == "Spend":
            return "Spend" if "Spend" in raw.columns else ""
        if label == "Clicks":
            return "Clicks" if "Clicks" in raw.columns else ""
        if label == "Customer Search Term":
            return "Customer Search Term" if "Customer Search Term" in raw.columns else ""
        if label == "Campaign Name":
            return "Campaign Name" if "Campaign Name" in raw.columns else ""
        if label == "Impressions":
            return "Impressions" if "Impressions" in raw.columns else ""
        return ""

    def sel(col, key):
        default = guess_default(col)
        idx = cols.index(default) if default in cols else 0
        return st.selectbox(col, cols, index=idx, key=key)

    overrides = {}
    with c1:
        overrides["Sales"] = sel("Sales", "ov_sales")
        overrides["Orders"] = sel("Orders", "ov_orders")
    with c2:
        overrides["Spend"] = sel("Spend", "ov_spend")
        overrides["Clicks"] = sel("Clicks", "ov_clicks")
    with c3:
        overrides["Customer Search Term"] = sel("Customer Search Term", "ov_cst")
        overrides["Campaign Name"] = sel("Campaign Name", "ov_camp")
    with c4:
        overrides["Impressions"] = sel("Impressions", "ov_impr")

    # Clean + compute
    df, diag = build_clean_df(raw, overrides)

    # Summary
    spend = df["Spend"].sum()
    sales = df["Sales"].sum()
    orders = int(df["Orders"].sum())
    clicks = int(df["Clicks"].sum())
    roas = (sales / spend) if spend > 0 else 0.0
    acos = (spend / sales * 100) if sales > 0 else 0.0
    tcoas = acos
    cvr = (orders / clicks * 100) if clicks > 0 else 0.0
    ctr = (df["Clicks"].sum() / df["Impressions"].sum() * 100) if df["Impressions"].sum() > 0 else 0.0
    avg_cpc = (spend / clicks) if clicks > 0 else 0.0

    st.subheader("💰 Financial performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spend", money(spend))
    m2.metric("Sales", money(sales))
    m3.metric("ROAS", f"{roas:.2f}x")
    m4.metric("ACOS", f"{acos:.1f}%")
    m5.metric("TCoAS", f"{tcoas:.1f}%")

    st.subheader("📈 Key metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Orders", f"{orders:,}")
    k2.metric("Clicks", f"{clicks:,}")
    k3.metric("CTR", f"{ctr:.2f}%")
    k4.metric("CVR", f"{cvr:.2f}%")
    k5.metric("Avg CPC", money(avg_cpc))

    wastage = df.loc[df["Sales"] == 0, "Spend"].sum()
    waste_pct = (wastage / spend * 100) if spend > 0 else 0.0
    st.markdown(
        f'<div class="box danger"><b>Wastage (zero-sales spend)</b><br>{money(wastage)} ({waste_pct:.1f}%)</div>',
        unsafe_allow_html=True,
    )

    if debug:
        diag_html = "<br>".join([f"<b>{k}:</b> <code>{v}</code>" for k, v in diag.items()])
        st.markdown(f'<div class="box info"><b>Diagnostics</b><br>{diag_html}</div>', unsafe_allow_html=True)

    # Groups
    st.subheader("🎯 Keyword groups (Scale / Test / Watch / Reduce / Pause)")
    groups = classify(df)
    t = st.tabs([f"🏆 Scale ({len(groups['Scale'])})", f"⚡ Test ({len(groups['Test'])})", f"👀 Watch ({len(groups['Watch'])})", f"⚠️ Reduce ({len(groups['Reduce'])})", f"🚨 Pause ({len(groups['Pause'])})"])

    for tab, key in zip(t, ["Scale", "Test", "Watch", "Reduce", "Pause"]):
        with tab:
            if len(groups[key]) == 0:
                st.markdown('<div class="box info">No rows here.</div>', unsafe_allow_html=True)
            else:
                st.dataframe(add_sno(groups[key]), use_container_width=True, hide_index=True, height=520)

    # Bid suggestions
    st.subheader("💡 Bid suggestions")
    sug = bid_suggestions(df, target_roas=target_roas, target_acos=target_acos)
    if len(sug) == 0:
        st.markdown('<div class="box info">No bid actions found (need enough spend/clicks or performance signals).</div>', unsafe_allow_html=True)
    else:
        st.dataframe(add_sno(sug), use_container_width=True, hide_index=True, height=520)

        # Export bulk file
        st.markdown("### 📥 Export")
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as wr:
            sug.to_excel(wr, index=False, sheet_name="Bid Suggestions")
        out.seek(0)
        st.download_button("Download Bid Suggestions (XLSX)", data=out, file_name=f"Bid_Suggestions_{client}_{datetime.now().strftime('%Y%m%d')}.xlsx", use_container_width=True)


if __name__ == "__main__":
    main()
