#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amazon Ads Dashboard Pro – Full v13

Features

- Multi-client sidebar (agency name, clients, upload file per client)
- Auto-detect columns from Amazon Search Term report (CSV/XLSX)
- Keyword metrics: ROAS, ACOS, CTR, CVR, TACoS (approx from ad sales)
- TACoS-aware pause guidance + 20-click rule
- High-performing keyword harvest (promote to exact/SK)
- Automated negative harvest (0-order terms above click/spend thresholds)
- Negative keyword suggestions (campaign vs ad-group level, generic bad terms)
- Bid suggestions (increase / reduce / pause / salvage-low-bid)
- Placement optimization (Top of Search vs Product pages etc.)
- Bulk export files:
    * Bid updates
    * Negative keywords (human readable)
    * Negative keywords (BULK file format)
    * Negative keywords (API-style payload)
    * High-performing harvest CSV
    * Non-converting harvest CSV

Visual fix

- Uses custom HTML metric cards instead of st.metric for headline KPIs so
  large INR values are never truncated or shortened. [web:138]
"""

import io
import math
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Amazon Ads Dashboard Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_CVR = 4.0
DEFAULT_BUFFER = 1.5
DEFAULT_PRICE = 500.0

LOW_VOL_MIN_CLICKS_SMALL = 10
LOW_VOL_MIN_CLICKS_MEDIUM = 15
LOW_VOL_MIN_CLICKS_LARGE = 20

PRICE_SPEND_MULTIPLIER = 2.0
SALVAGE_BID_FACTOR = 0.3

HARVEST_MIN_CLICKS = 10
HARVEST_MIN_ORDERS = 2
HARVEST_MIN_ROAS = 2.0

DEFAULT_TACOS_TARGET = 12.0  # % account-level TACoS target band. [web:130]


# ---------------------------------------------------------------------------
# STYLING + METRIC CARDS
# ---------------------------------------------------------------------------

def load_custom_css():
    css = """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #020617 0, #020617 40%, #000000 100%);
    }
    .main {
        padding-top: 0.5rem;
    }
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

    .info-box, .success-box, .warning-box, .danger-box {
        padding: 0.95rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.9rem;
        font-size: 0.92rem;
    }
    .info-box {
        background: radial-gradient(circle at top left, rgba(59,130,246,0.33), rgba(15,23,42,0.98));
        border-left: 4px solid #3b82f6;
        color: #e5e7eb;
    }
    .success-box {
        background: radial-gradient(circle at top left, rgba(34,197,94,0.35), rgba(6,78,59,0.98));
        border-left: 4px solid #22c55e;
        color: #dcfce7;
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
    .metric-card {
        background: radial-gradient(circle at top left, #020617, #020617);
        border-radius: 14px;
        padding: 0.9rem 1.0rem;
        border: 1px solid rgba(148,163,184,0.5);
        box-shadow: 0 10px 26px rgba(15,23,42,0.85);
        color: #e5e7eb;
        white-space: nowrap;
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


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def safe_str(value, default: str = "N/A") -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            return default
        return str(value).strip()
    except Exception:
        return default


def parse_number(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            return default
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().replace(",", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
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


def get_excel_writer_engine(preferred: str = "openpyxl") -> str:
    try:
        if preferred == "xlsxwriter":
            import xlsxwriter  # noqa: F401
            return "xlsxwriter"
    except Exception:
        pass
    return "openpyxl"


# ---------------------------------------------------------------------------
# FILE IO + COLUMN DETECTION
# ---------------------------------------------------------------------------

REQUIRED_HINTS = ["customer search term", "campaign name", "spend", "clicks"]


def _score_sheet_columns(cols):
    cols_l = [str(c).lower().strip() for c in cols]
    score = 0
    for h in REQUIRED_HINTS:
        if any(h == c or h in c for c in cols_l):
            score += 1
    return score


def read_uploaded_report(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")

    xls = pd.ExcelFile(uploaded_file)
    best_df, best_score, best_rows = None, -1, -1
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
            best_df, best_score, best_rows = tmp, score, rows
    if best_df is None:
        return pd.read_excel(uploaded_file)
    return best_df


def pick_best_column(columns_lower, patterns):
    for pat in patterns:
        rx = re.compile(pat)
        matches = [c for c in columns_lower if rx.search(c)]
        if matches:
            return sorted(matches, key=len)[0]
    return None


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = [safe_str(c, "").strip() for c in df.columns]
    cols_lower = [c.lower().strip() for c in cols]
    col_map = {c.lower().strip(): c for c in cols}

    cst = pick_best_column(cols_lower, [r"^customer search term$", r"customer search term", r"search term"])
    camp = pick_best_column(cols_lower, [r"^campaign name$", r"campaign name"])
    clicks = pick_best_column(cols_lower, [r"^clicks$", r"clicks"])
    spend = pick_best_column(cols_lower, [r"^spend$", r"\bcost\b", r"ad spend", r"spend"])
    impr = pick_best_column(cols_lower, [r"^impressions$", r"impressions", r"\bimps\b"])
    match = pick_best_column(cols_lower, [r"match type", r"matchtype"])
    adg = pick_best_column(cols_lower, [r"ad group name", r"adgroup"])
    cpc = pick_best_column(cols_lower, [r"cost per click", r"\bcpc\b"])
    sales = pick_best_column(
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
    orders = pick_best_column(
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

    def m(x):
        return col_map.get(x, "") if x else ""

    return {
        "Customer Search Term": m(cst),
        "Campaign Name": m(camp),
        "Ad Group Name": m(adg),
        "Match Type": m(match),
        "Clicks": m(clicks),
        "Spend": m(spend),
        "Impressions": m(impr),
        "Sales": m(sales),
        "Orders": m(orders),
        "CPC": m(cpc),
    }


# ---------------------------------------------------------------------------
# ANALYZER
# ---------------------------------------------------------------------------

class CompleteAnalyzer:
    def __init__(
        self,
        raw_df: pd.DataFrame,
        client_name: str,
        column_overrides: Optional[Dict[str, str]] = None,
        target_acos: Optional[float] = None,
        target_roas: Optional[float] = None,
        product_price: Optional[float] = None,
        product_cvr: Optional[float] = None,
        pause_buffer: float = DEFAULT_BUFFER,
        low_vol_bucket: str = "Medium",
        tacos_target: float = DEFAULT_TACOS_TARGET,
    ):
        self.client_name = client_name
        self.target_acos = target_acos
        self.target_roas = target_roas
        self.column_overrides = column_overrides or {}
        self.product_price = product_price or DEFAULT_PRICE
        self.product_cvr = product_cvr or DEFAULT_CVR
        self.pause_buffer = pause_buffer
        self.low_vol_bucket = low_vol_bucket
        self.tacos_target = tacos_target
        self.pause_clicks = self._compute_pause_clicks()
        self.low_vol_min_clicks = self._compute_low_vol_min_clicks()
        self.df: Optional[pd.DataFrame] = None
        self.diag: Dict[str, str] = {}
        self.df = self._prepare(raw_df)

    # ---------- thresholds ----------

    def _compute_pause_clicks(self) -> int:
        cvr = max(self.product_cvr, 0.1)
        base = 100.0 / cvr
        pause_clicks = max(20.0, base * self.pause_buffer)  # 20-click minimum data rule.
        return int(math.ceil(pause_clicks))

    def _compute_low_vol_min_clicks(self) -> int:
        if self.low_vol_bucket == "Small":
            return LOW_VOL_MIN_CLICKS_SMALL
        if self.low_vol_bucket == "Large":
            return LOW_VOL_MIN_CLICKS_LARGE
        return LOW_VOL_MIN_CLICKS_MEDIUM

    # ---------- data prep ----------

    def _col(self, detected: Dict[str, str], key: str) -> str:
        ov = safe_str(self.column_overrides.get(key, ""), "").strip()
        if ov:
            return ov
        return detected.get(key, "")

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Empty file / sheet")

        df = df.copy().dropna(how="all")
        df.columns = [safe_str(c, "").strip() for c in df.columns]
        detected = auto_detect_columns(df)

        cst = self._col(detected, "Customer Search Term")
        camp = self._col(detected, "Campaign Name")
        clicks = self._col(detected, "Clicks")
        spend = self._col(detected, "Spend")
        impr = self._col(detected, "Impressions")
        sales = self._col(detected, "Sales")
        orders = self._col(detected, "Orders")
        adg = self._col(detected, "Ad Group Name")
        match = self._col(detected, "Match Type")
        cpc = self._col(detected, "CPC")

        # required
        required_missing = [
            k
            for k, v in {
                "Customer Search Term": cst,
                "Campaign Name": camp,
                "Clicks": clicks,
                "Spend": spend,
            }.items()
            if not v
        ]
        if required_missing:
            raise ValueError(
                f"Missing required mapping: {required_missing}. Use Column Override to fix."
            )

        out = pd.DataFrame()
        out["Customer Search Term"] = df[cst].astype(str).fillna("").str.strip()
        out["Campaign Name"] = df[camp].astype(str).fillna("").str.strip()
        out["Ad Group Name"] = df[adg].astype(str).fillna("N/A").str.strip() if adg else "N/A"
        out["Match Type"] = df[match].astype(str).fillna("N/A").str.strip() if match else "N/A"
        out["Clicks"] = to_numeric_series(df[clicks])
        out["Spend"] = to_numeric_series(df[spend])
        out["Impressions"] = to_numeric_series(df[impr]) if impr else 0.0
        out["Sales"] = to_numeric_series(df[sales]) if sales else 0.0
        out["Orders"] = to_numeric_series(df[orders]) if orders else 0.0
        out["CPC"] = to_numeric_series(df[cpc]) if cpc else 0.0

        out["CPC"] = out.apply(
            lambda r: float(r["CPC"])
            if float(r["CPC"]) > 0
            else (float(r["Spend"]) / float(r["Clicks"]) if float(r["Clicks"]) > 0 else 0.0),
            axis=1,
        )

        out = out[(out["Spend"] > 0) | (out["Clicks"] > 0)].copy()
        if len(out) == 0:
            raise ValueError("No valid rows after filtering (Spend/Clicks all 0).")

        out["Profit"] = out["Sales"] - out["Spend"]
        out["Wastage"] = out.apply(
            lambda r: float(r["Spend"]) if float(r["Orders"]) == 0 else 0.0,
            axis=1,
        )
        out["CVR"] = out.apply(
            lambda r: (float(r["Orders"]) / float(r["Clicks"]) * 100) if float(r["Clicks"]) > 0 else 0.0,
            axis=1,
        )
        out["ROAS"] = out.apply(
            lambda r: (float(r["Sales"]) / float(r["Spend"])) if float(r["Spend"]) > 0 else 0.0,
            axis=1,
        )
        out["ACOS"] = out.apply(
            lambda r: (float(r["Spend"]) / float(r["Sales"]) * 100) if float(r["Sales"]) > 0 else 0.0,
            axis=1,
        )
        out["CTR"] = out.apply(
            lambda r: (float(r["Clicks"]) / float(r["Impressions"]) * 100)
            if float(r["Impressions"]) > 0
            else 0.0,
            axis=1,
        )

        # TACoS at row level, using ad sales as proxy for total until organic is merged. [web:130]
        out["Total_Sales_for_TACOS"] = out["Sales"]
        out["TACOS"] = out.apply(
            lambda r: (float(r["Spend"]) / float(r["Total_Sales_for_TACOS"]) * 100)
            if float(r["Total_Sales_for_TACOS"]) > 0
            else 0.0,
            axis=1,
        )

        out["Negative_Type"] = out["Customer Search Term"].apply(
            lambda x: "PRODUCT" if is_asin(x) else "KEYWORD"
        )
        out["Client"] = self.client_name
        out["Processed_Date"] = datetime.now()

        self.diag = {
            "Sales column": sales or "NOT SET",
            "Orders column": orders or "NOT SET",
            "Spend column": spend or "NOT SET",
            "Clicks column": clicks or "NOT SET",
            "Pause clicks (20-click rule floor)": str(self.pause_clicks),
            "Low-vol min clicks": str(self.low_vol_min_clicks),
        }
        return out

    # ---------- summary ----------

    def summary(self) -> Dict[str, float]:
        if self.df is None or len(self.df) == 0:
            return {
                "spend": 0,
                "sales": 0,
                "orders": 0,
                "clicks": 0,
                "impr": 0,
                "wastage": 0,
                "roas": 0,
                "acos": 0,
                "avg_cpc": 0,
                "avg_ctr": 0,
                "avg_cvr": 0,
                "tacos": 0,
            }

        ts = float(self.df["Spend"].sum())
        sa = float(self.df["Sales"].sum())
        od = float(self.df["Orders"].sum())
        ck = float(self.df["Clicks"].sum())
        im = float(self.df["Impressions"].sum())
        wa = float(self.df["Wastage"].sum())
        roas = sa / ts if ts > 0 else 0.0
        acos = ts / sa * 100 if sa > 0 else 0.0
        tacos = ts / sa * 100 if sa > 0 else 0.0

        return {
            "spend": ts,
            "sales": sa,
            "orders": od,
            "clicks": ck,
            "impr": im,
            "wastage": wa,
            "roas": roas,
            "acos": acos,
            "avg_cpc": (ts / ck) if ck > 0 else 0.0,
            "avg_ctr": float(self.df["CTR"].mean()),
            "avg_cvr": float(self.df["CVR"].mean()),
            "tacos": tacos,
        }

    # ---------- pause / negation helpers ----------

    def _is_price_guardrail_fail(self, r) -> bool:
        spend = float(r["Spend"])
        orders = float(r["Orders"])
        return orders == 0 and spend >= PRICE_SPEND_MULTIPLIER * self.product_price

    def _is_high_click_zero_order(self, r) -> bool:
        clicks = float(r["Clicks"])
        orders = float(r["Orders"])
        return orders == 0 and clicks >= max(self.pause_clicks, self.low_vol_min_clicks)

    def _is_irrelevant_negative(self, r) -> bool:
        return self._is_high_click_zero_order(r) or self._is_price_guardrail_fail(r)

    def _is_relevant_unprofitable(self, r) -> bool:
        orders = float(r["Orders"])
        if orders <= 0:
            return False
        acos = float(r["ACOS"])
        roas = float(r["ROAS"])
        ta = self.target_acos or 30.0
        tr = self.target_roas or 3.0
        return (acos > ta * 1.5) or (roas < tr * 0.5)

    # ---------- TACoS pause band ----------

    def tacos_pause_band(self, tacos_current: float) -> str:
        tgt = self.tacos_target
        t = tacos_current
        if t <= tgt * 0.9:
            return "Aggressive: TACoS healthy; safe to cut 0-order and very high ACOS keywords."
        if tgt * 0.9 < t <= tgt * 1.1:
            return "Neutral: TACoS on target; pause only clear losers, keep ranking/discovery terms."
        return "Defensive: TACoS high; first lower bids & placements, then prune worst 0-order terms."

    # ---------- generic negative pattern ----------

    def _generic_negative_pattern(self, term: str) -> bool:
        patt = [
            r"\bfree\b",
            r"\bjobs?\b",
            r"\bcareer\b",
            r"\bwholesale\b",
            r"\bbulk\b",
            r"\bcheap\b",
            r"\bused\b",
            r"\bsecond hand\b",
            r"\bmanual\b",
            r"\breview\b",
        ]
        t = term.lower()
        return any(re.search(p, t) for p in patt)

    # ---------- harvesting high performers ----------

    def harvest_high_performers(self) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()
        df = self.df.copy()
        df = df[
            (df["Clicks"] >= HARVEST_MIN_CLICKS)
            & (df["Orders"] >= HARVEST_MIN_ORDERS)
            & (df["ROAS"] >= HARVEST_MIN_ROAS)
        ]
        if df.empty:
            return df
        out = df[
            [
                "Customer Search Term",
                "Campaign Name",
                "Ad Group Name",
                "Match Type",
                "Clicks",
                "Orders",
                "Spend",
                "Sales",
                "ROAS",
                "ACOS",
                "TACOS",
            ]
        ].copy()
        out["Clicks"] = out["Clicks"].astype(int)
        out["Orders"] = out["Orders"].astype(int)
        out["Spend"] = out["Spend"].apply(format_currency)
        out["Sales"] = out["Sales"].apply(format_currency)
        out["ROAS"] = out["ROAS"].apply(lambda x: f"{float(x):.2f}x")
        out["ACOS"] = out["ACOS"].apply(lambda x: f"{float(x):.1f}%")
        out["TACOS"] = out["TACOS"].apply(lambda x: f"{float(x):.1f}%")
        out["Harvest Type"] = "Promote to Exact (SK campaign)"
        return out

    # ---------- automated negative harvesting ----------

    def harvest_negatives(self) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()
        df = self.df.copy()
        df = df[(df["Orders"] == 0) & (df["Clicks"] >= max(self.pause_clicks, self.low_vol_min_clicks))]
        if df.empty:
            return df
        df = df.sort_values("Spend", ascending=False)
        out = df[
            [
                "Customer Search Term",
                "Campaign Name",
                "Ad Group Name",
                "Match Type",
                "Clicks",
                "Spend",
                "Impressions",
                "ACOS",
                "ROAS",
                "TACOS",
            ]
        ].copy()
        out["Spend"] = out["Spend"].apply(format_currency)
        out["ACOS"] = out["ACOS"].apply(lambda x: f"{float(x):.1f}%")
        out["ROAS"] = out["ROAS"].apply(lambda x: f"{float(x):.2f}x")
        out["TACOS"] = out["TACOS"].apply(lambda x: f"{float(x):.1f}%")
        out["Recommended Level"] = "Campaign"
        out["Recommended Match"] = "Exact"
        out["Reason"] = (
            "0 orders after ≥20 clicks (20-click rule) and sufficient spend – add as campaign-level negative exact."
        )
        return out

    # ---------- classification ----------

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
            tacos = float(r["TACOS"])
            term = safe_str(r["Customer Search Term"])

            item = {
                "Keyword": term,
                "Campaign": safe_str(r["Campaign Name"]),
                "Ad Group": safe_str(r["Ad Group Name"]),
                "Match Type": safe_str(r["Match Type"]),
                "Spend": format_currency(sp),
                "Sales": format_currency(sa),
                "Orders": o,
                "Clicks": c,
                "ROAS": f"{ro:.2f}x",
                "ACOS": f"{float(r['ACOS']):.1f}%",
                "TACOS": f"{tacos:.1f}%",
                "CVR": f"{cv:.2f}%",
                "CPC": format_currency(cpc),
                "Generic Negative Pattern": "Yes" if self._generic_negative_pattern(term) else "",
                "Reason": "",
            }

            if sp >= 20 and o >= 1 and ro >= 2.5:
                item["Reason"] = "Winner (scale +15–25%; keep if TACoS heading down)."
                cats["scale"].append(item)
            elif self._is_irrelevant_negative(r):
                item["Reason"] = (
                    f"Irrelevant? {c} clicks (20-click rule hit), 0 orders and/or spend ≥ {PRICE_SPEND_MULTIPLIER}× price – "
                    "candidate for campaign-level negative exact."
                )
                cats["pause"].append(item)
            elif sp >= 30 and ro < 1.0 and c >= 5:
                item["Reason"] = "Low ROAS (reduce ~30% or salvage with low bid)."
                cats["reduce"].append(item)
            elif sp >= 20 and 1.5 <= ro < 2.5 and c >= 3:
                item["Reason"] = "Good potential (test +10–15%)."
                cats["test"].append(item)
            else:
                item["Reason"] = "Collect more data (watch)."
                cats["watch"].append(item)

        return cats

    # ---------- bid suggestions ----------

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
            action, change, new_bid, reason = "", 0, cpc, ""

            if self._is_irrelevant_negative(r):
                action, change, new_bid, reason = (
                    "PAUSE",
                    -100,
                    0.0,
                    "Irrelevant/non-buying term – 0 orders after ≥20 clicks and high spend. "
                    "Pause + campaign-level negative exact.",
                )
            elif self._is_relevant_unprofitable(r):
                salvage_bid = cpc * SALVAGE_BID_FACTOR
                action, change, new_bid, reason = (
                    "SALVAGE_LOW_BID",
                    int(-100 * (1 - SALVAGE_BID_FACTOR)),
                    salvage_bid,
                    "Relevant but unprofitable – move to salvage bid instead of pausing, to protect rank and TACoS.",
                )
            elif ro >= target_roas and o >= 1 and cv >= 1.0:
                action, change, new_bid, reason = "INCREASE", 15, cpc * 1.15, "Above target ROAS."
            elif ro < 1.5 and sp >= 30:
                action, change, new_bid, reason = "REDUCE", -30, cpc * 0.70, "Low ROAS."
            elif acos > target_acos and sp >= 30:
                red = min(30, (acos - target_acos) / target_acos * 100)
                action, change, new_bid, reason = (
                    "REDUCE",
                    int(-red),
                    cpc * (1 - red / 100),
                    f"ACOS {acos:.1f}% above target.",
                )

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
                        "ACOS": f"{float(r['ACOS']):.1f}%",
                        "TACOS": f"{float(r['TACOS']):.1f}%",
                        "CVR": f"{cv:.2f}%",
                        "Current CPC": format_currency(cpc),
                        "Action": action,
                        "Change (%)": change,
                        "Suggested Bid": format_currency(new_bid) if new_bid > 0 else "₹0.00",
                        "Reason": reason,
                    }
                )

        return suggestions

    # ---------- negatives ----------

    def negative_keywords(self) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()

        rows = []
        for _, r in self.df.iterrows():
            kw = safe_str(r["Customer Search Term"])
            if not kw:
                continue
            campaign = safe_str(r["Campaign Name"])
            adgroup = safe_str(r["Ad Group Name"])
            sp = float(r["Spend"])
            sa = float(r["Sales"])
            o = int(r["Orders"])
            acos = float(r["ACOS"])
            roas = float(r["ROAS"])
            match_type = "Exact"

            if self._is_irrelevant_negative(r) or self._generic_negative_pattern(kw):
                reason = (
                    "Campaign-level negative exact: irrelevant/generic term "
                    "or 0 orders after ≥20 clicks and significant spend."
                )
                rows.append(
                    {
                        "Level": "Campaign",
                        "Campaign Name": campaign,
                        "Ad Group Name": "",
                        "Negative Keyword": kw,
                        "Match Type": match_type,
                        "Type": "Irrelevant / Generic",
                        "Spend": format_currency(sp),
                        "Sales": format_currency(sa),
                        "Orders": o,
                        "ACOS": f"{acos:.1f}%",
                        "ROAS": f"{roas:.2f}x",
                        "Recommendation": reason,
                    }
                )
            elif self._is_relevant_unprofitable(r):
                reason = (
                    "Ad-group-level negative exact: relevant but unprofitable here. "
                    "Move to dedicated exact campaign at low bid; block in this ad group."
                )
                rows.append(
                    {
                        "Level": "Ad Group",
                        "Campaign Name": campaign,
                        "Ad Group Name": adgroup,
                        "Negative Keyword": kw,
                        "Match Type": match_type,
                        "Type": "Relevant but unprofitable",
                        "Spend": format_currency(sp),
                        "Sales": format_currency(sa),
                        "Orders": o,
                        "ACOS": f"{acos:.1f}%",
                        "ROAS": f"{roas:.2f}x",
                        "Recommendation": reason,
                    }
                )

        return pd.DataFrame(rows)

    # ---------- API & BULK negatives ----------

    def api_negative_payload(self) -> pd.DataFrame:
        neg = self.negative_keywords()
        if neg.empty:
            return neg
        api_rows = []
        for _, r in neg.iterrows():
            api_rows.append(
                {
                    "campaignId": "",
                    "adGroupId": "" if r["Level"] == "Campaign" else "",
                    "keywordText": r["Negative Keyword"],
                    "matchType": r["Match Type"].upper(),
                    "state": "ENABLED",
                    "level": r["Level"],
                    "note": r["Recommendation"],
                }
            )
        return pd.DataFrame(api_rows)

    def bulk_negative_file(self) -> pd.DataFrame:
        neg = self.negative_keywords()
        if neg.empty:
            return neg
        rows = []
        for _, r in neg.iterrows():
            level = r["Level"]
            rows.append(
                {
                    "Record Type": "Negative Keyword",
                    "Campaign Name": r["Campaign Name"],
                    "Ad Group Name": r["Ad Group Name"] if level == "Ad Group" else "",
                    "Keyword or Product Targeting": r["Negative Keyword"],
                    "Match Type": r["Match Type"],
                    "State": "enabled",
                    "Operation": "create",
                }
            )
        return pd.DataFrame(rows)

    # ---------- placements ----------

    def placement_recommendations(self) -> Dict:
        res = {"available": False, "table": pd.DataFrame(), "recs": [], "message": ""}

        if self.df is None or len(self.df) == 0:
            res["message"] = "No data."
            return res

        placement_cols = [c for c in self.df.columns if "placement" in c.lower()]
        if not placement_cols:
            res["message"] = "No placement column found."
            return res

        col = placement_cols[0]
        dfp = self.df.copy()
        dfp[col] = dfp[col].fillna("UNKNOWN").astype(str).str.strip()

        g = (
            dfp.groupby(col)
            .agg(
                Spend=("Spend", "sum"),
                Sales=("Sales", "sum"),
                Orders=("Orders", "sum"),
                Clicks=("Clicks", "sum"),
                Impressions=("Impressions", "sum"),
            )
            .reset_index()
            .rename(columns={col: "Placement"})
        )

        g["ROAS"] = g.apply(lambda r: (r["Sales"] / r["Spend"]) if r["Spend"] > 0 else 0.0, axis=1)
        g["ACOS"] = g.apply(
            lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 0.0,
            axis=1,
        )

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
                recs.append(
                    {
                        "Placement": r["Placement"],
                        "Action": "INCREASE",
                        "Recommendation": "High-performing placement – increase placement bid by 10–20% instead of pausing keywords.",
                    }
                )
            elif ro < 1.0 or ac > ta * 1.5:
                recs.append(
                    {
                        "Placement": r["Placement"],
                        "Action": "REDUCE",
                        "Recommendation": "Weak placement – lower placement multiplier 15–30% or to 0%, keep keywords live.",
                    }
                )
            else:
                recs.append(
                    {
                        "Placement": r["Placement"],
                        "Action": "HOLD",
                        "Recommendation": "Average placement – minor bid tweaks only; don't pause solely due to placement.",
                    }
                )

        res["available"] = True
        res["table"] = g
        res["recs"] = recs
        return res

    # ---------- wastage ----------

    def top_wastage(self, n: Optional[int] = 50, min_spend: float = 0.0) -> pd.DataFrame:
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()
        df = self.df.copy()
        df = df[df["Wastage"] > 0]
        if min_spend and min_spend > 0:
            df = df[df["Spend"] >= float(min_spend)]
        df = df.sort_values("Wastage", ascending=False)
        if n and n > 0:
            df = df.head(n)
        return df


# ---------------------------------------------------------------------------
# SESSION / STATE
# ---------------------------------------------------------------------------

class ClientData:
    def __init__(self, name: str):
        self.name = name
        self.analyzer: Optional[CompleteAnalyzer] = None
        self.target_acos: Optional[float] = None
        self.target_roas: Optional[float] = None
        self.tacos_target: float = DEFAULT_TACOS_TARGET
        self.column_overrides: Dict[str, str] = {}
        self.product_price: float = DEFAULT_PRICE
        self.product_cvr: float = DEFAULT_CVR
        self.pause_buffer: float = DEFAULT_BUFFER
        self.low_vol_bucket: str = "Medium"


def init_session():
    if "agency_name" not in st.session_state:
        st.session_state.agency_name = "Your Agency"
    if "clients" not in st.session_state:
        st.session_state.clients: Dict[str, ClientData] = {}
    if "active_client" not in st.session_state:
        st.session_state.active_client = None
    if "debug" not in st.session_state:
        st.session_state.debug = True


def header():
    st.markdown(
        f"""
        <div class="agency-header">
            <h1>🏢 {st.session_state.agency_name} – Amazon Ads Dashboard Pro v13</h1>
            <p>Full metrics (no truncation) • TACoS-aware rules • placements • negatives • bulk files</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_df(df: pd.DataFrame, height: int = 450):
    if df is None or len(df) == 0:
        st.info("No rows to show.")
        return
    st.dataframe(add_serial_column(df), use_container_width=True, hide_index=True, height=height)


def reset_app():
    st.session_state.clients = {}
    st.session_state.active_client = None


def sidebar():
    with st.sidebar:
        st.session_state.agency_name = st.text_input("Agency name", st.session_state.agency_name)

        c1, c2 = st.columns(2)
        with c1:
            st.session_state.debug = st.checkbox("Show diagnostics", value=st.session_state.debug)
        with c2:
            if st.button("Reset app", use_container_width=True):
                reset_app()
                st.rerun()

        st.markdown("---")
        st.markdown("### 👥 Clients")

        existing_names = list(st.session_state.clients.keys())
        existing_client = None
        if existing_names:
            sel = st.selectbox("Active client", existing_names)
            st.session_state.active_client = sel
            existing_client = st.session_state.clients.get(sel)

        st.markdown("---")
        with st.expander("➕ Add / Update client", expanded=True):
            nm = st.text_input("Client name*", value=existing_client.name if existing_client else "")

            t1, t2 = st.columns(2)
            with t1:
                t_acos = st.number_input(
                    "Target ACOS % (optional)",
                    value=float(existing_client.target_acos if existing_client and existing_client.target_acos else 30.0),
                    step=5.0,
                    format="%.1f",
                )
            with t2:
                t_roas = st.number_input(
                    "Target ROAS (optional)",
                    value=float(existing_client.target_roas if existing_client and existing_client.target_roas else 3.0),
                    step=0.5,
                    format="%.1f",
                )

            t3, t4 = st.columns(2)
            with t3:
                tacos_target = st.number_input(
                    "TACoS target % (account)",
                    value=float(existing_client.tacos_target if existing_client else DEFAULT_TACOS_TARGET),
                    step=1.0,
                    format="%.1f",
                )
            with t4:
                st.caption("TACoS = Ad spend ÷ Total sales × 100. Lower TACoS with stable sales is ideal. [web:130]")

            p1, p2 = st.columns(2)
            with p1:
                prod_price = st.number_input(
                    "Typical product price (₹)",
                    value=float(existing_client.product_price if existing_client else DEFAULT_PRICE),
                    step=50.0,
                    format="%.0f",
                )
            with p2:
                prod_cvr = st.number_input(
                    "Avg product CVR % (orders/clicks)",
                    value=float(existing_client.product_cvr if existing_client else DEFAULT_CVR),
                    step=0.5,
                    format="%.1f",
                )

            b1, b2 = st.columns(2)
            with b1:
                pause_buf = st.slider(
                    "Pause buffer (× breakeven clicks)",
                    min_value=1.0,
                    max_value=3.0,
                    value=float(existing_client.pause_buffer if existing_client else DEFAULT_BUFFER),
                    step=0.1,
                )
            with b2:
                low_vol_bucket = st.selectbox(
                    "Budget level (click threshold)",
                    ["Small", "Medium", "Large"],
                    index=["Small", "Medium", "Large"].index(
                        existing_client.low_vol_bucket if existing_client else "Medium"
                    ),
                )

            up = st.file_uploader(
                "Upload Search Term report (XLSX/CSV)* (required only for new client)",
                type=["xlsx", "xls", "csv"],
            )

            preview_df, detected = None, {}
            if up is not None:
                try:
                    preview_df = read_uploaded_report(up)
                    detected = auto_detect_columns(preview_df)
                except Exception as e:
                    st.error(f"Preview read error: {e}")
                    preview_df = None

            overrides = {}
            if preview_df is not None:
                st.markdown("#### Column override (only if needed)")
                cols = [""] + list(preview_df.columns)

                def selbox(label, key, default):
                    idx = cols.index(default) if default in cols else 0
                    return st.selectbox(label, cols, index=idx, key=key)

                overrides["Sales"] = selbox("Sales column", "ov_sales", detected.get("Sales", ""))
                overrides["Orders"] = selbox("Orders column", "ov_orders", detected.get("Orders", ""))
                overrides["Spend"] = selbox("Spend column", "ov_spend", detected.get("Spend", ""))
                overrides["Clicks"] = selbox("Clicks column", "ov_clicks", detected.get("Clicks", ""))
                overrides["Impressions"] = selbox(
                    "Impressions column", "ov_impr", detected.get("Impressions", "")
                )
                overrides["Customer Search Term"] = selbox(
                    "Customer Search Term", "ov_cst", detected.get("Customer Search Term", "")
                )
                overrides["Campaign Name"] = selbox(
                    "Campaign Name", "ov_camp", detected.get("Campaign Name", "")
                )

            if st.button("✅ Create / Update", use_container_width=True):
                if not nm:
                    st.error("Enter client name.")
                    return

                current_client = st.session_state.clients.get(nm)
                if current_client is None and up is None:
                    st.error("Upload a Search Term report for a new client.")
                    return

                try:
                    if up is not None:
                        df = read_uploaded_report(up)
                        cl = current_client or ClientData(nm)
                        cl.target_acos = float(t_acos) if t_acos > 0 else None
                        cl.target_roas = float(t_roas) if t_roas > 0 else None
                        cl.tacos_target = float(tacos_target)
                        cl.product_price = float(prod_price) if prod_price > 0 else DEFAULT_PRICE
                        cl.product_cvr = float(prod_cvr) if prod_cvr > 0 else DEFAULT_CVR
                        cl.pause_buffer = float(pause_buf)
                        cl.low_vol_bucket = low_vol_bucket
                        cl.column_overrides = {
                            k: v for k, v in (overrides or {}).items() if safe_str(v, "").strip()
                        }
                        cl.analyzer = CompleteAnalyzer(
                            raw_df=df,
                            client_name=nm,
                            column_overrides=cl.column_overrides,
                            target_acos=cl.target_acos,
                            target_roas=cl.target_roas,
                            product_price=cl.product_price,
                            product_cvr=cl.product_cvr,
                            pause_buffer=cl.pause_buffer,
                            low_vol_bucket=cl.low_vol_bucket,
                            tacos_target=cl.tacos_target,
                        )
                        st.session_state.clients[nm] = cl
                        st.session_state.active_client = nm
                        st.success("Client created/updated from uploaded report.")
                    else:
                        if current_client is None:
                            st.error("No existing client found with this name.")
                            return
                        current_client.target_acos = float(t_acos) if t_acos > 0 else None
                        current_client.target_roas = float(t_roas) if t_roas > 0 else None
                        current_client.tacos_target = float(tacos_target)
                        current_client.product_price = float(prod_price) if prod_price > 0 else DEFAULT_PRICE
                        current_client.product_cvr = float(prod_cvr) if prod_cvr > 0 else DEFAULT_CVR
                        current_client.pause_buffer = float(pause_buf)
                        current_client.low_vol_bucket = low_vol_bucket
                        if current_client.analyzer:
                            a = current_client.analyzer
                            a.target_acos = current_client.target_acos
                            a.target_roas = current_client.target_roas
                            a.tacos_target = current_client.tacos_target
                            a.product_price = current_client.product_price
                            a.product_cvr = current_client.product_cvr
                            a.pause_buffer = current_client.pause_buffer
                            a.low_vol_bucket = current_client.low_vol_bucket
                            a.pause_clicks = a._compute_pause_clicks()
                            a.low_vol_min_clicks = a._compute_low_vol_min_clicks()
                        st.session_state.clients[nm] = current_client
                        st.session_state.active_client = nm
                        st.success("Targets/statistics updated. Open Dashboard.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload/update error: {e}")
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


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

def dashboard_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report to view dashboard.")
        return

    s = an.summary()
    st.subheader("💰 Financial performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Spend", format_currency(s["spend"]))
    with c2:
        metric_card("Sales", format_currency(s["sales"]))
    with c3:
        metric_card("ROAS", f"{s['roas']:.2f}x")
    with c4:
        metric_card("ACOS", f"{s['acos']:.1f}%")
    with c5:
        metric_card("TACoS (approx.)", f"{s['tacos']:.1f}%")

    st.subheader("📈 Key metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Orders", format_number(s["orders"]))
    with c2:
        metric_card("Clicks", format_number(s["clicks"]))
    with c3:
        metric_card("Impressions", format_number(s["impr"]))
    with c4:
        metric_card("CTR", f"{s['avg_ctr']:.2f}%")
    with c5:
        metric_card("Avg CPC", format_currency(s["avg_cpc"]))

    wp = (s["wastage"] / s["spend"] * 100) if s["spend"] > 0 else 0.0
    st.markdown(
        f"""
        <div class="danger-box">
            <strong>Wastage (zero‑order spend)</strong><br>
            {format_currency(s["wastage"])} ({wp:.1f}% of spend)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="warning-box">20-click rule: only pause a keyword once it reaches at least 20 clicks with 0 sales, '
        'and preferably more for low CVR products. This protects you from killing winners too early.</div>',
        unsafe_allow_html=True,
    )

    tacos_band = an.tacos_pause_band(tacos_current=s["tacos"])
    st.markdown(
        f'<div class="info-box"><strong>TACoS guidance:</strong> {tacos_band}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.debug:
        lines = "<br>".join(
            [f"<strong>{k}:</strong> <code>{safe_str(v,'')}</code>" for k, v in an.diag.items()]
        )
        st.markdown(
            f'<div class="info-box"><strong>Diagnostics</strong><br>{lines}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("🔥 Wastage hotspots (zero‑order spend terms)")
    min_spend = st.slider(
        "Minimum spend to include (₹)", min_value=0, max_value=1000, value=50, step=10
    )
    top_n = st.selectbox("Max rows", [20, 50, 100], index=1)
    wdf = an.top_wastage(n=top_n, min_spend=min_spend)
    if wdf.empty:
        st.info("No wastage rows with the selected filters.")
    else:
        view = wdf[
            [
                "Customer Search Term",
                "Campaign Name",
                "Match Type",
                "Spend",
                "Wastage",
                "Clicks",
                "Impressions",
                "Sales",
                "Orders",
                "ACOS",
                "ROAS",
                "TACOS",
            ]
        ].copy()
        view["Spend"] = view["Spend"].apply(format_currency)
        view["Wastage"] = view["Wastage"].apply(format_currency)
        view["Sales"] = view["Sales"].apply(format_currency)
        view["ACOS"] = view["ACOS"].apply(lambda x: f"{float(x):.1f}%")
        view["ROAS"] = view["ROAS"].apply(lambda x: f"{float(x):.2f}x")
        view["TACOS"] = view["TACOS"].apply(lambda x: f"{float(x):.1f}%")
        show_df(view, height=320)


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
    c5.metric("🚨 Pause / Negatives", len(cats["pause"]))

    tabs = st.tabs(
        [
            f"🏆 Scale ({len(cats['scale'])})",
            f"⚡ Test ({len(cats['test'])})",
            f"👀 Watch ({len(cats['watch'])})",
            f"⚠️ Reduce ({len(cats['reduce'])})",
            f"🚨 Pause / Negatives ({len(cats['pause'])})",
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
        st.info("No bid suggestions yet.")
        return

    inc = sum(1 for x in sug if x["Action"] == "INCREASE")
    red = sum(1 for x in sug if x["Action"] == "REDUCE")
    pau = sum(1 for x in sug if x["Action"] == "PAUSE")
    salv = sum(1 for x in sug if x["Action"] == "SALVAGE_LOW_BID")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⬆️ Increase", inc)
    c2.metric("⬇️ Reduce", red)
    c3.metric("⏸️ Pause", pau)
    c4.metric("🛟 Salvage low-bid", salv)

    flt = st.selectbox(
        "Filter", ["All", "INCREASE", "REDUCE", "PAUSE", "SALVAGE_LOW_BID"], index=0
    )
    view = sug if flt == "All" else [x for x in sug if x["Action"] == flt]
    show_df(pd.DataFrame(view), height=520)


def placement_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    st.subheader("📍 Placement optimization (no pause)")
    st.caption(
        "Use placement reports to adjust bid multipliers (Top of Search, Product Pages, Rest of Search) "
        "instead of pausing keywords."
    )

    res = an.placement_recommendations()
    if not res["available"]:
        st.info(res.get("message", "No placement data."))
        return

    dfp = res["table"].copy()
    dfp["Spend"] = dfp["Spend"].apply(format_currency)
    dfp["Sales"] = dfp["Sales"].apply(format_currency)
    dfp["ROAS"] = dfp["ROAS"].apply(lambda x: f"{float(x):.2f}x")
    dfp["ACOS"] = dfp["ACOS"].apply(lambda x: f"{float(x):.1f}%")
    show_df(dfp, height=360)

    st.markdown("#### Recommendations")
    show_df(pd.DataFrame(res["recs"]), height=260)


def harvest_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    st.subheader("🌱 High-performing keyword harvest")
    hdf = an.harvest_high_performers()
    if hdf.empty:
        st.info("No high-performing search terms found with current thresholds.")
        return

    show_df(hdf, height=520)


def negative_harvest_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    st.subheader("🧹 Automated negative harvesting (0-order terms)")
    neg_h = an.harvest_negatives()
    if neg_h.empty:
        st.info("No non-converting terms above thresholds.")
        return

    show_df(neg_h, height=520)


def negatives_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    st.subheader("🚫 Negative keyword suggestions")
    neg_df = an.negative_keywords()
    if neg_df.empty:
        st.info("No negative keyword suggestions yet.")
        return

    show_df(neg_df, height=520)


def exports_page(cl: ClientData):
    an = cl.analyzer
    if not an or an.df is None:
        st.info("Upload a report first.")
        return

    st.subheader("📥 Exports")

    # BULK bid file
    sug = an.bid_suggestions()
    if sug:
        bulk_rows = []
        for s in sug:
            campaign = safe_str(s.get("Campaign"), "")
            ad_group = safe_str(s.get("Ad Group"), "")
            keyword = safe_str(s.get("Keyword"), "")
            match_type = safe_str(s.get("Match Type"), "")
            action = safe_str(s.get("Action"), "")
            bid_val = parse_number(s.get("Suggested Bid", 0), 0.0)
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
            bulk_rows.append(row)

        bulk = pd.DataFrame(bulk_rows)
        if len(bulk) > 0:
            out = io.BytesIO()
            engine = get_excel_writer_engine()
            with pd.ExcelWriter(out, engine=engine) as wr:
                bulk.to_excel(wr, index=False, sheet_name="Sponsored Products")
            out.seek(0)
            st.download_button(
                f"Download Amazon BULK bid file ({len(bulk)} rows)",
                data=out,
                file_name=f"Bulk_Bids_{cl.name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # Clean dataset
    csv = an.df.to_csv(index=False)
    st.download_button(
        f"Download cleaned dataset CSV ({len(an.df)} rows)",
        data=csv,
        file_name=f"Cleaned_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Wastage-only
    wdf = an.top_wastage(n=None, min_spend=0.0)
    if not wdf.empty:
        wcsv = wdf.to_csv(index=False)
        st.download_button(
            f"Download wastage-only CSV ({len(wdf)} rows)",
            data=wcsv,
            file_name=f"Wastage_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Negative keyword CSV (human-readable)
    neg_df = an.negative_keywords()
    if not neg_df.empty:
        ncsv = neg_df.to_csv(index=False)
        st.download_button(
            f"Download negative keyword CSV ({len(neg_df)} rows)",
            data=ncsv,
            file_name=f"Negatives_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # BULK negative file (Campaign Manager upload)
    bulk_neg = an.bulk_negative_file()
    if not bulk_neg.empty:
        bncsv = bulk_neg.to_csv(index=False)
        st.download_button(
            f"Download BULK negative keywords file ({len(bulk_neg)} rows)",
            data=bncsv,
            file_name=f"Bulk_Negatives_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # API-negative CSV
    api_neg = an.api_negative_payload()
    if not api_neg.empty:
        acsv = api_neg.to_csv(index=False)
        st.download_button(
            f"Download API-ready negative payload ({len(api_neg)} rows)",
            data=acsv,
            file_name=f"API_Negatives_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # High-performer harvest CSV
    hdf = an.harvest_high_performers()
    if not hdf.empty:
        hcsv = hdf.to_csv(index=False)
        st.download_button(
            f"Download high-performing harvest CSV ({len(hdf)} rows)",
            data=hcsv,
            file_name=f"Harvest_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Non-converting harvest CSV
    nh = an.harvest_negatives()
    if not nh.empty:
        nhcsv = nh.to_csv(index=False)
        st.download_button(
            f"Download non-converting (negative harvest) CSV ({len(nh)} rows)",
            data=nhcsv,
            file_name=f"Negative_Harvest_{cl.name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    load_custom_css()
    init_session()
    sidebar()
    header()

    if not st.session_state.clients:
        st.markdown(
            '<div class="info-box">Add a client and upload your Search Term report from the left sidebar.</div>',
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.active_client:
        st.markdown(
            '<div class="warning-box">Select an active client from the sidebar.</div>',
            unsafe_allow_html=True,
        )
        return

    cl = st.session_state.clients[st.session_state.active_client]

    # Ensure analyzer exists
    if not cl.analyzer or cl.analyzer.df is None:
        st.markdown(
            '<div class="warning-box">This client has no analyzer data yet. Upload a Search Term report in the sidebar.</div>',
            unsafe_allow_html=True,
        )
        return

    tabs = st.tabs(
        [
            "📊 Dashboard",
            "🎯 Keywords",
            "💡 Bids",
            "📍 Placements",
            "🌱 Harvest",
            "🧹 Neg Harvest",
            "🚫 Negatives",
            "📥 Exports",
        ]
    )
    with tabs[0]:
        dashboard_page(cl)
    with tabs[1]:
        keywords_page(cl)
    with tabs[2]:
        bids_page(cl)
    with tabs[3]:
        placement_page(cl)
    with tabs[4]:
        harvest_page(cl)
    with tabs[5]:
        negative_harvest_page(cl)
    with tabs[6]:
        negatives_page(cl)
    with tabs[7]:
        exports_page(cl)


if __name__ == "__main__":
    main()
