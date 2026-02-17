#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amazon Ads Dashboard Pro v7.2 (Sales Fix + Premium UI)

Fixes:
- Sales/Orders/ROAS/TCoAS/CVR not showing for big-date reports:
  * Auto-selects correct Excel sheet
  * Smart column detection for 7/14/30 day Sales & Orders (currency variations)
  * Robust numeric parsing for formatted currency strings
- S.No starts from 1 and hides default index (no 0 index visible)
- Premium UI with high-contrast tables and clearer metric labels
- Placement-wise recommendations when placement data exists
"""

import io
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
# Premium styling
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
        background: radial-gradient(circle at top left, #a855f7 0, #1e293b 35%, #020617 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.1rem;
        color: #e5e7eb;
        box-shadow: 0 18px 50px rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.5);
    }
    .agency-header h1 { margin: 0; font-size: 1.5rem; }
    .agency-header p { margin: 0.35rem 0 0 0; font-size: 0.9rem; color: #e0e7ff; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 55%, #020617 100%);
        border-radius: 16px;
        padding: 1.1rem 0.9rem;
        border: 1px solid #1f2937;
        box-shadow: 0 16px 40px rgba(15,23,42,0.9);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #e5e7eb !important;
        white-space: normal !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        color: #f9fafb !important;
        white-space: normal !important;
        word-break: break-word !important;
    }

    /* Sidebar inputs */
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
        background: radial-gradient(circle at top left, rgba(59,130,246,0.33), rgba(15,23,42,0.98));
        border-left: 4px solid #3b82f6;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.92rem;
        color: #e5e7eb;
    }
    .success-box {
        background: radial-gradient(circle at top left, rgba(34,197,94,0.35), rgba(6,78,59,0.98));
        border-left: 4px solid #22c55e;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.92rem;
        color: #dcfce7;
    }
    .warning-box {
        background: radial-gradient(circle at top left, rgba(250,204,21,0.33), rgba(77,54,10,0.98));
        border-left: 4px solid #eab308;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.92rem;
        color: #fef9c3;
    }
    .danger-box {
        background: radial-gradient(circle at top left, rgba(248,113,113,0.40), rgba(127,29,29,0.99));
        border-left: 4px solid #ef4444;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.92rem;
        color: #fee2e2;
    }
    .cyan-box {
        background: radial-gradient(circle at top left, rgba(6,182,212,0.35), rgba(8,47,73,0.98));
        border-left: 4px solid #06b6d4;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.92rem;
        color: #cffafe;
    }

    /* Dataframe: high contrast header & cells */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        border: 1px solid rgba(148,163,184,0.35);
        overflow: hidden;
        box-shadow: 0 14px 36px rgba(15,23,42,0.65);
    }
    div[data-testid="stDataFrame"] thead tr th {
        background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(2,6,23,0.95)) !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(148,163,184,0.35) !important;
    }
    div[data-testid="stDataFrame"] tbody tr td {
        color: #e5e7eb !important;
        background: rgba(2,6,23,0.65) !important;
        border-bottom: 1px solid rgba(148,163,184,0.14) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Robust parsing helpers
# -----------------------------------------------------------------------------
def safe_str(value, default: str = "N/A") -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            return default
        return str(value).strip()
    except Exception:
        return default


def parse_number(value, default: float = 0.0) -> float:
    """
    Very robust numeric parser:
    - Works with '₹1,234.50', 'INR 1234', '[$₹-en-US]18.96', '(123.4)' etc.
    - Extracts the first numeric token found.
    """
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            return default
        if isinstance(value, (int, float)):
            return float(value)

        s = str(value).strip()
        s = s.replace(",", "")
        # parenthesis negative
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        # extract first number token
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return default
        return float(m.group(0))
    except Exception:
        return default


def to_numeric_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: parse_number(x, 0.0)).astype(float)


def format_currency(value) -> str:
    v = parse_number(value, 0.0)
    if v >= 10_000_000:
        return f"₹{v/10_000_000:.2f}Cr"
    if v >= 100_000:
        return f"₹{v/100_000:.2f}L"
    return f"₹{v:,.2f}"


def format_number(value) -> str:
    v = int(parse_number(value, 0.0))
    if v >= 10_000_000:
        return f"{v/10_000_000:.2f}Cr"
    if v >= 100_000:
        return f"{v/100_000:.2f}L"
    if v >= 1000:
        return f"{v:,}"
    return str(v)


def add_serial_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.reset_index(drop=True).copy()
    out.insert(0, "S.No", range(1, len(out) + 1))
    return out


def is_asin(value) -> bool:
    val = safe_str(value, "")
    return bool(re.match(r"^[B][0-9A-Z]{9,}$", val.upper()))


def get_negative_type(value) -> str:
    return "PRODUCT" if is_asin(value) else "KEYWORD"


# -----------------------------------------------------------------------------
# File reading (auto-sheet selection)
# -----------------------------------------------------------------------------
REQUIRED_HINTS = [
    "customer search term",
    "campaign name",
    "spend",
    "clicks",
]


def _score_sheet_columns(cols: List[str]) -> int:
    cols_l = [c.lower().strip() for c in cols]
    score = 0
    for h in REQUIRED_HINTS:
        if any(h == c or h in c for c in cols_l):
            score += 1
    return score


def read_uploaded_report(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        # try common CSV reading
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")

    # Excel: choose best sheet automatically
    xls = pd.ExcelFile(uploaded_file)
    best_df = None
    best_score = -1
    best_rows = -1

    for sh in xls.sheet_names:
        try:
            tmp = pd.read_excel(xls, sheet_name=sh)
        except Exception:
            continue

        if tmp is None or len(tmp) == 0:
            continue

        score = _score_sheet_columns(list(tmp.columns))
        rows = int(tmp.shape[0])

        if score > best_score or (score == best_score and rows > best_rows):
            best_score = score
            best_rows = rows
            best_df = tmp

    if best_df is None:
        # fallback to first sheet
        return pd.read_excel(uploaded_file)

    return best_df


# -----------------------------------------------------------------------------
# Analyzer
# -----------------------------------------------------------------------------
def pick_best_column(columns_lower: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat)
        matches = [c for c in columns_lower if rx.search(c)]
        if matches:
            # choose the shortest match (usually the cleanest)
            return sorted(matches, key=len)[0]
    return None


class CompleteAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        client_name: str,
        target_acos: Optional[float] = None,
        target_roas: Optional[float] = None,
        target_cpa: Optional[float] = None,
        target_tcoas: Optional[float] = None,
    ):
        self.client_name = client_name
        self.target_acos = target_acos
        self.target_roas = target_roas
        self.target_cpa = target_cpa
        self.target_tcoas = target_tcoas

        self.df: Optional[pd.DataFrame] = None
        self.sales_source: str = ""
        self.orders_source: str = ""
        self.spend_source: str = ""
        self.clicks_source: str = ""
        self.impr_source: str = ""

        self.df = self._prepare(df)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Empty file / sheet")

        df = df.copy()
        df.columns = [safe_str(c, "").strip() for c in df.columns]

        # drop fully empty rows
        df = df.dropna(how="all")
        if len(df) == 0:
            raise ValueError("No usable rows in the file")

        # Build column lookup in lowercase
        cols_lower = [c.lower().strip() for c in df.columns]
        col_map = {c.lower().strip(): c for c in df.columns}

        # Required base fields (flexible)
        cst = pick_best_column(cols_lower, [r"^customer search term$", r"customer search term", r"search term"])
        camp = pick_best_column(cols_lower, [r"^campaign name$", r"campaign name"])
        clicks = pick_best_column(cols_lower, [r"^clicks$", r"clicks"])
        spend = pick_best_column(cols_lower, [r"^spend$", r"\bcost\b", r"ad spend", r"spend"])
        impr = pick_best_column(cols_lower, [r"^impressions$", r"impressions", r"\bimps\b"])

        if not cst or not camp or not clicks or not spend:
            raise ValueError(
                "Required columns not found. Need: Customer Search Term, Campaign Name, Spend/Cost, Clicks."
            )

        # Sales column detection (7/14/30 day, currency variations)
        sales_col_l = pick_best_column(
            cols_lower,
            [
                r"\b7\s*day\s*total\s*sales\b",
                r"\b14\s*day\s*total\s*sales\b",
                r"\b30\s*day\s*total\s*sales\b",
                r"\btotal\s*sales\b",
                r"(^sales$)|(\bsales\b)",
                r"\brevenue\b",
            ],
        )

        # Orders column detection (7/14/30 day, units/orders variations)
        orders_col_l = pick_best_column(
            cols_lower,
            [
                r"\b7\s*day\s*total\s*orders\b",
                r"\b14\s*day\s*total\s*orders\b",
                r"\b30\s*day\s*total\s*orders\b",
                r"\bordered\s*units\b",
                r"\btotal\s*orders\b",
                r"(^orders$)|(\borders\b)",
                r"\bunits\b",
            ],
        )

        # Extra fallback sales split columns
        adv_sales_l = pick_best_column(cols_lower, [r"advertised\s*sku\s*sales"])
        other_sales_l = pick_best_column(cols_lower, [r"other\s*sku\s*sales"])

        # Optional columns
        adgroup_l = pick_best_column(cols_lower, [r"ad group name", r"adgroup"])
        match_l = pick_best_column(cols_lower, [r"match type", r"matchtype"])
        cpc_l = pick_best_column(cols_lower, [r"cost per click", r"\bcpc\b"])

        # Build normalized dataframe
        out = pd.DataFrame()
        out["Customer Search Term"] = df[col_map[cst]].astype(str).fillna("").str.strip()
        out["Campaign Name"] = df[col_map[camp]].astype(str).fillna("").str.strip()
        out["Ad Group Name"] = df[col_map[adgroup_l]].astype(str).fillna("N/A").str.strip() if adgroup_l else "N/A"
        out["Match Type"] = df[col_map[match_l]].astype(str).fillna("N/A").str.strip() if match_l else "N/A"

        out["Clicks"] = to_numeric_series(df[col_map[clicks]])
        out["Spend"] = to_numeric_series(df[col_map[spend]])
        out["Impressions"] = to_numeric_series(df[col_map[impr]]) if impr else 0.0

        self.clicks_source = col_map[clicks]
        self.spend_source = col_map[spend]
        self.impr_source = col_map[impr] if impr else ""

        # Sales
        if sales_col_l:
            out["Sales"] = to_numeric_series(df[col_map[sales_col_l]])
            self.sales_source = col_map[sales_col_l]
        else:
            out["Sales"] = 0.0
            self.sales_source = ""

        # Orders
        if orders_col_l:
            out["Orders"] = to_numeric_series(df[col_map[orders_col_l]])
            self.orders_source = col_map[orders_col_l]
        else:
            out["Orders"] = 0.0
            self.orders_source = ""

        # Fallback: if Sales is still zero but we have adv/other sales, rebuild
        if out["Sales"].sum() == 0 and (adv_sales_l or other_sales_l):
            adv = to_numeric_series(df[col_map[adv_sales_l]]) if adv_sales_l else 0.0
            oth = to_numeric_series(df[col_map[other_sales_l]]) if other_sales_l else 0.0
            out["Sales"] = adv + oth
            self.sales_source = "Advertised SKU Sales + Other SKU Sales"

        # CPC: use file CPC if present; else calculate
        if cpc_l:
            out["CPC"] = to_numeric_series(df[col_map[cpc_l]])
        else:
            out["CPC"] = 0.0

        out["CPC"] = out.apply(
            lambda r: float(r["CPC"]) if float(r["CPC"]) > 0 else (float(r["Spend"]) / float(r["Clicks"]) if float(r["Clicks"]) > 0 else 0.0),
            axis=1
        )

        # Keep active rows
        out = out[(out["Spend"] > 0) | (out["Clicks"] > 0)].copy()
        if len(out) == 0:
            raise ValueError("No valid rows after filtering (Spend/Clicks are all 0)")

        # Derived metrics
        out["Profit"] = out["Sales"] - out["Spend"]
        out["Wastage"] = out.apply(lambda r: float(r["Spend"]) if float(r["Sales"]) == 0 else 0.0, axis=1)
        out["CVR"] = out.apply(lambda r: (float(r["Orders"]) / float(r["Clicks"]) * 100) if float(r["Clicks"]) > 0 else 0.0, axis=1)
        out["ROAS"] = out.apply(lambda r: (float(r["Sales"]) / float(r["Spend"])) if float(r["Spend"]) > 0 else 0.0, axis=1)
        out["ACOS"] = out.apply(lambda r: (float(r["Spend"]) / float(r["Sales"]) * 100) if float(r["Sales"]) > 0 else 0.0, axis=1)
        out["CTR"] = out.apply(lambda r: (float(r["Clicks"]) / float(r["Impressions"]) * 100) if float(r["Impressions"]) > 0 else 0.0, axis=1)
        out["CPA"] = out.apply(lambda r: (float(r["Spend"]) / float(r["Orders"])) if float(r["Orders"]) > 0 else 0.0, axis=1)
        out["TCoAS"] = out["ACOS"]

        out["Negative_Type"] = out["Customer Search Term"].apply(get_negative_type)
        out["Client"] = self.client_name
        out["Processed_Date"] = datetime.now()

        return out

    def summary(self) -> Dict[str, float]:
        if self.df is None or len(self.df) == 0:
            return {
                "spend": 0, "sales": 0, "orders": 0, "clicks": 0, "impr": 0,
                "wastage": 0, "roas": 0, "acos": 0, "tcoas": 0,
                "avg_cpc": 0, "avg_ctr": 0, "avg_cvr": 0, "avg_cpa": 0,
            }

        ts = float(self.df["Spend"].sum())
        sa = float(self.df["Sales"].sum())
        od = float(self.df["Orders"].sum())
        ck = float(self.df["Clicks"].sum())
        im = float(self.df["Impressions"].sum()) if "Impressions" in self.df.columns else 0.0
        wa = float(self.df["Wastage"].sum())

        roas = sa / ts if ts > 0 else 0.0
        acos = ts / sa * 100 if sa > 0 else 0.0
        avg_cpc = (ts / ck) if ck > 0 else 0.0
        avg_ctr = float(self.df["CTR"].mean()) if len(self.df) else 0.0
        avg_cvr = float(self.df["CVR"].mean()) if len(self.df) else 0.0
        avg_cpa = (ts / od) if od > 0 else 0.0

        return {
            "spend": ts,
            "sales": sa,
            "orders": od,
            "clicks": ck,
            "impr": im,
            "wastage": wa,
            "roas": roas,
            "acos": acos,
            "tcoas": acos,
            "avg_cpc": avg_cpc,
            "avg_ctr": avg_ctr,
            "avg_cvr": avg_cvr,
            "avg_cpa": avg_cpa,
        }

    def classify_keywords(self) -> Dict[str, List[Dict]]:
        cats = {"scale": [], "test": [], "watch": [], "reduce": [], "pause": []}
        if self.df is None or len(self.df) == 0:
            return cats

        for _, r in self.df.iterrows():
            sp = float(r["Spend"])
            sa = float(r["Sales"])
            ro = float(r["ROAS"])
            o = int(r["Orders"])
            c = int(r["Clicks"])
            cv = float(r["CVR"])
            cpc = float(r["CPC"])

            item = {
                "Keyword": safe_str(r["Customer Search Term"]),
                "Campaign": safe_str(r["Campaign Name"]),
                "Ad Group": safe_str(r["Ad Group Name"]),
                "Match Type": safe_str(r["Match Type"]),
                "Spend": format_currency(sp),
                "Sales": format_currency(sa),
                "Orders": o,
                "Clicks": c,
                "ROAS": f"{ro:.2f}x",
                "CVR": f"{cv:.2f}%",
                "CPC": format_currency(cpc),
                "Reason": "",
            }

            if sp >= 20 and o >= 1 and ro >= 2.5:
                item["Reason"] = "Winner (scale +15–25%)."
                cats["scale"].append(item)
            elif sp >= 50 and sa == 0 and c >= 3:
                item["Reason"] = "Spent high, zero sales (pause/negative)."
                cats["pause"].append(item)
            elif sp >= 30 and ro < 1.0 and c >= 5:
                item["Reason"] = "Low ROAS (reduce ~30%)."
                cats["reduce"].append(item)
            elif sp >= 20 and 1.5 <= ro < 2.5 and c >= 3:
                item["Reason"] = "Good potential (test +10–15%)."
                cats["test"].append(item)
            else:
                item["Reason"] = "Collect more data (watch)."
                cats["watch"].append(item)

        return cats

    def bid_suggestions(self) -> List[Dict]:
        if self.df is None or len(self.df) == 0:
            return []

        target_roas = self.target_roas or 3.0
        target_acos = self.target_acos or 30.0

        suggestions = []
        for _, r in self.df.iterrows():
            sp = float(r["Spend"])
            sa = float(r["Sales"])
            ro = float(r["ROAS"])
            o = int(r["Orders"])
            c = int(r["Clicks"])
            cpc = float(r["CPC"])
            cv = float(r["CVR"])

            if sp < 20 or c < 3 or cpc <= 0:
                continue

            acos = (sp / sa * 100) if sa > 0 else 999.0
            action = ""
            change = 0
            new_bid = cpc
            reason = ""

            if sa == 0 and sp >= 50:
                action = "PAUSE"
                change = -100
                new_bid = 0
                reason = "High spend, zero sales."
            elif ro >= target_roas and o >= 1 and cv >= 1.0:
                action = "INCREASE"
                change = 15
                new_bid = cpc * 1.15
                reason = "Above target ROAS."
            elif ro >= 3.0 and o >= 2 and cv >= 2.0:
                action = "INCREASE"
                change = 25
                new_bid = cpc * 1.25
                reason = "Strong performance."
            elif ro < 1.5 and sp >= 30:
                action = "REDUCE"
                change = -30
                new_bid = cpc * 0.70
                reason = "Low ROAS."
            elif acos > target_acos and sp >= 30:
                action = "REDUCE"
                red = min(30, (acos - target_acos) / target_acos * 100)
                change = int(-red)
                new_bid = cpc * (1 - red / 100)
                reason = f"ACOS {acos:.1f}% above target."

            if action:
                suggestions.append(
                    {
                        "Keyword": safe_str(r["Customer Search Term"]),
                        "Campaign": safe_str(r["Campaign Name"]),
                        "Ad Group": safe_str(r["Ad Group Name"]),
                        "Match Type": safe_str(r["Match Type"]),
                        "Spend": format_currency(sp),
                        "Sales": format_currency(sa),
                        "Orders": o,
                        "ROAS": f"{ro:.2f}x",
                        "CVR": f"{cv:.2f}%",
                        "Current CPC": format_currency(cpc),
                        "Action": action,
                        "Change (%)": change,
                        "Suggested Bid": format_currency(new_bid) if new_bid > 0 else "₹0.00",
                        "Reason": reason,
                    }
                )

        return suggestions

    def build_amazon_bulk_from_bids(self, bid_suggestions: List[Dict]) -> pd.DataFrame:
        rows = []
        for s in bid_suggestions:
            campaign = safe_str(s.get("Campaign"), "")
            ad_group = safe_str(s.get("Ad Group"), "")
            keyword = safe_str(s.get("Keyword"), "")
            match_type = safe_str(s.get("Match Type"), "")
            action = safe_str(s.get("Action"), "").upper()

            bid_val = parse_number(s.get("Suggested Bid", 0), 0.0)
            if bid_val <= 0:
                # suggested bid could be '₹0.00' -> becomes 0
                bid_val = 0.0

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

        return pd.DataFrame(rows)

    def placement_recommendations(self) -> Dict:
        res = {"available": False, "table": pd.DataFrame(), "recs": [], "message": ""}
        if self.df is None or len(self.df) == 0:
            res["message"] = "No data."
            return res

        placement_cols = [c for c in self.df.columns if "placement" in c.lower()]
        if not placement_cols:
            res["message"] = "No placement column found in this report."
            return res

        col = placement_cols[0]
        dfp = self.df.copy()
        dfp[col] = dfp[col].fillna("UNKNOWN").astype(str).str.strip()

        g = dfp.groupby(col).agg(
            Spend=("Spend", "sum"),
            Sales=("Sales", "sum"),
            Orders=("Orders", "sum"),
            Clicks=("Clicks", "sum"),
            Impressions=("Impressions", "sum"),
        ).reset_index().rename(columns={col: "Placement"})

        g["ROAS"] = g.apply(lambda r: (r["Sales"] / r["Spend"]) if r["Spend"] > 0 else 0.0, axis=1)
        g["ACOS"] = g.apply(lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 0.0, axis=1)
        g["CVR"] = g.apply(lambda r: (r["Orders"] / r["Clicks"] * 100) if r["Clicks"] > 0 else 0.0, axis=1)
        g["CTR"] = g.apply(lambda r: (r["Clicks"] / r["Impressions"] * 100) if r["Impressions"] > 0 else 0.0, axis=1)

        ta = self.target_acos or 30.0
        tr = self.target_roas or 3.0

        recs = []
        for _, r in g.iterrows():
            sp = float(r["Spend"])
            ro = float(r["ROAS"])
            ac = float(r["ACOS"])
            if sp <= 0:
                continue
            if ro >= tr or ac <= ta:
                recs.append({"Placement": r["Placement"], "Action": "INCREASE", "Recommendation": f"Good placement (ROAS {ro:.2f}x, ACOS {ac:.1f}%). Test +10–20% placement bid."})
            elif ro < 1.0 or ac > ta * 1.5:
                recs.append({"Placement": r["Placement"], "Action": "REDUCE", "Recommendation": f"Weak placement (ROAS {ro:.2f}x, ACOS {ac:.1f}%). Reduce by 15–30% or pause."})
            else:
                recs.append({"Placement": r["Placement"], "Action": "HOLD", "Recommendation": f"Average placement (ROAS {ro:.2f}x, ACOS {ac:.1f}%). Keep stable and monitor."})

        res["available"] = True
        res["table"] = g
        res["recs"] = recs
        return res


# -----------------------------------------------------------------------------
# App state
# -----------------------------------------------------------------------------
class ClientData:
    def __init__(self, name: str):
        self.name = name
        self.analyzer: Optional[CompleteAnalyzer] = None
        self.target_acos: Optional[float] = None
        self.target_roas: Optional[float] = None
        self.target_cpa: Optional[float] = None
        self.target_tcoas: Optional[float] = None


def init_session():
    if "agency_name" not in st.session_state:
        st.session_state.agency_name = "Your Agency"
    if "clients" not in st.session_state:
        st.session_state.clients: Dict[str, ClientData] = {}
    if "active_client" not in st.session_state:
        st.session_state.active_client = None
    if "debug" not in st.session_state:
        st.session_state.debug = False


def header():
    st.markdown(
        f"""
        <div class="agency-header">
            <h1>🏢 {st.session_state.agency_name} – Amazon Ads Dashboard Pro v7.2</h1>
            <p>Sales/Orders fixed • Sheet auto-detect • Premium UI • Placement recommendations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_df(df: pd.DataFrame, height: int = 420):
    if df is None or len(df) == 0:
        st.info("No rows to show.")
        return
    st.dataframe(add_serial_column(df), use_container_width=True, hide_index=True, height=height)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.session_state.agency_name = st.text_input("Agency name", st.session_state.agency_name)
        st.session_state.debug = st.checkbox("Show diagnostics", value=st.session_state.debug)

        st.markdown("---")
        st.markdown("### 👥 Clients")

        if st.session_state.clients:
            sel = st.selectbox("Active client", list(st.session_state.clients.keys()))
            st.session_state.active_client = sel

        st.markdown("---")
        with st.expander("➕ Add client", expanded=True):
            nm = st.text_input("Client name*")
            c1, c2 = st.columns(2)
            with c1:
                t_acos = st.number_input("Target ACOS %", value=0.0, step=5.0, format="%.1f")
                t_roas = st.number_input("Target ROAS", value=0.0, step=0.5, format="%.1f")
            with c2:
                t_cpa = st.number_input("Target CPA ₹", value=0.0, step=50.0, format="%.0f")
                t_tcoas = st.number_input("Target TCoAS %", value=0.0, step=5.0, format="%.1f")

            up = st.file_uploader("Upload report (XLSX/CSV)*", type=["xlsx", "xls", "csv"])

            if st.button("✅ Create / Update client", use_container_width=True):
                if not nm:
                    st.error("Enter client name.")
                    return
                if not up:
                    st.error("Upload a report file.")
                    return

                try:
                    df = read_uploaded_report(up)
                    cl = st.session_state.clients.get(nm) or ClientData(nm)

                    cl.target_acos = t_acos if t_acos > 0 else None
                    cl.target_roas = t_roas if t_roas > 0 else None
                    cl.target_cpa = t_cpa if t_cpa > 0 else None
                    cl.target_tcoas = t_tcoas if t_tcoas > 0 else None

                    cl.analyzer = CompleteAnalyzer(
                        df=df,
                        client_name=nm,
                        target_acos=cl.target_acos,
                        target_roas=cl.target_roas,
                        target_cpa=cl.target_cpa,
                        target_tcoas=cl.target_tcoas,
                    )

                    st.session_state.clients[nm] = cl
                    st.session_state.active_client = nm
                    st.success("Client ready.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload error: {e}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

        if st.session_state.clients:
            st.markdown("---")
            st.markdown("### 📋 All clients")
            for name in list(st.session_state.clients.keys()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📊 {name}")
                with col2:
                    if st.button("❌", key=f"del_{name}"):
                        del st.session_state.clients[name]
                        if st.session_state.active_client == name:
                            st.session_state.active_client = None
                        st.rerun()


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def dashboard_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report to view dashboard.")
        return

    s = an.summary()

    st.subheader("💰 Financial performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spend", format_currency(s["spend"]))
    c2.metric("Sales", format_currency(s["sales"]))
    c3.metric("ROAS", f"{s['roas']:.2f}x")
    c4.metric("ACOS", f"{s['acos']:.1f}%")
    c5.metric("TCoAS", f"{s['tcoas']:.1f}%")

    st.subheader("📈 Key metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders", format_number(s["orders"]))
    c2.metric("Clicks", format_number(s["clicks"]))
    c3.metric("CTR", f"{s['avg_ctr']:.2f}%")
    c4.metric("CVR", f"{s['avg_cvr']:.2f}%")
    c5.metric("Avg CPC", format_currency(s["avg_cpc"]))

    wp = (s["wastage"] / s["spend"] * 100) if s["spend"] > 0 else 0.0
    st.markdown(
        f"""
        <div class="danger-box">
            <strong>Wastage (zero‑sales spend)</strong><br>
            {format_currency(s["wastage"])} ({wp:.1f}%)
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.debug:
        st.markdown(
            f"""
            <div class="info-box">
                <strong>Diagnostics</strong><br>
                Spend column: <code>{an.spend_source}</code><br>
                Clicks column: <code>{an.clicks_source}</code><br>
                Impressions column: <code>{an.impr_source or "Not found"}</code><br>
                Sales source: <code>{an.sales_source or "Not detected"}</code><br>
                Orders source: <code>{an.orders_source or "Not detected"}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📍 Placement recommendations")
    plac = an.placement_recommendations()
    if plac["available"]:
        dfp = plac["table"].copy()
        # format placement table
        dfp["Spend"] = dfp["Spend"].apply(format_currency)
        dfp["Sales"] = dfp["Sales"].apply(format_currency)
        dfp["ROAS"] = dfp["ROAS"].apply(lambda x: f"{float(x):.2f}x")
        dfp["ACOS"] = dfp["ACOS"].apply(lambda x: f"{float(x):.1f}%")
        dfp["CVR"] = dfp["CVR"].apply(lambda x: f"{float(x):.2f}%")
        dfp["CTR"] = dfp["CTR"].apply(lambda x: f"{float(x):.2f}%")
        show_df(dfp, height=280)

        for r in plac["recs"]:
            box = "success-box" if r["Action"] == "INCREASE" else ("danger-box" if r["Action"] == "REDUCE" else "info-box")
            st.markdown(
                f"""
                <div class="{box}">
                    <strong>{r["Placement"]}</strong> – {r["Action"]}<br>
                    {r["Recommendation"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(f'<div class="info-box">{plac["message"]}</div>', unsafe_allow_html=True)


def keywords_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    cats = an.classify_keywords()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏆 Scale", len(cats["scale"]))
    c2.metric("⚡ Test", len(cats["test"]))
    c3.metric("👀 Watch", len(cats["watch"]))
    c4.metric("⚠️ Reduce", len(cats["reduce"]))
    c5.metric("🚨 Pause", len(cats["pause"]))

    tabs = st.tabs(
        [
            f"🏆 Scale ({len(cats['scale'])})",
            f"⚡ Test ({len(cats['test'])})",
            f"👀 Watch ({len(cats['watch'])})",
            f"⚠️ Reduce ({len(cats['reduce'])})",
            f"🚨 Pause ({len(cats['pause'])})",
        ]
    )

    with tabs[0]:
        show_df(pd.DataFrame(cats["scale"]))
    with tabs[1]:
        show_df(pd.DataFrame(cats["test"]))
    with tabs[2]:
        show_df(pd.DataFrame(cats["watch"]))
    with tabs[3]:
        show_df(pd.DataFrame(cats["reduce"]))
    with tabs[4]:
        show_df(pd.DataFrame(cats["pause"]))


def bids_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    sug = an.bid_suggestions()
    if not sug:
        st.info("No bid suggestions yet (need enough clicks/spend).")
        return

    inc = sum(1 for x in sug if x["Action"] == "INCREASE")
    red = sum(1 for x in sug if x["Action"] == "REDUCE")
    pau = sum(1 for x in sug if x["Action"] == "PAUSE")

    c1, c2, c3 = st.columns(3)
    c1.metric("⬆️ Increase", inc)
    c2.metric("⬇️ Reduce", red)
    c3.metric("⏸️ Pause", pau)

    flt = st.selectbox("Filter", ["All", "INCREASE", "REDUCE", "PAUSE"], index=0)
    view = sug if flt == "All" else [x for x in sug if x["Action"] == flt]
    show_df(pd.DataFrame(view), height=520)


def exports_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    st.subheader("📥 Exports")

    sug = an.bid_suggestions()
    if sug:
        bulk = an.build_amazon_bulk_from_bids(sug)
        if len(bulk) > 0:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as wr:
                bulk.to_excel(wr, index=False, sheet_name="Sponsored Products")
            out.seek(0)
            st.download_button(
                f"Download Amazon BULK bid file ({len(bulk)} rows)",
                data=out,
                file_name=f"Bulk_Bids_{cl.name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    else:
        st.info("No bid suggestions to export.")

    csv = an.df.to_csv(index=False)
    st.download_button(
        f"Download cleaned dataset CSV ({len(an.df)} rows)",
        data=csv,
        file_name=f"Cleaned_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    load_custom_css()
    init_session()
    sidebar()
    header()

    if not st.session_state.clients:
        st.markdown('<div class="info-box">Add a client and upload a report from the left sidebar.</div>', unsafe_allow_html=True)
        return

    if not st.session_state.active_client:
        st.markdown('<div class="warning-box">Select an active client from the sidebar.</div>', unsafe_allow_html=True)
        return

    cl = st.session_state.clients[st.session_state.active_client]
    tabs = st.tabs(["📊 Dashboard", "🎯 Keywords", "💡 Bids", "📥 Exports"])
    with tabs[0]:
        dashboard_page(cl)
    with tabs[1]:
        keywords_page(cl)
    with tabs[2]:
        bids_page(cl)
    with tabs[3]:
        exports_page(cl)

    st.markdown(
        """
        <hr>
        <div style="text-align:center;color:#64748b;font-size:0.8rem;padding:0.6rem 0;">
        Amazon Ads Dashboard Pro v7.2
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
