#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amazon Ads Dashboard Pro v12.2

- Full v12 feature set:
  * Multi-client management
  * TACoS + 20-click rule logic
  * High-performing keyword harvest
  * Automated negative harvest
  * Placement optimization
  * Bulk bid & negative files

- Visual fix:
  * Replaces st.metric for headline KPIs with custom HTML metric cards
    so large INR values never get truncated or shortened. [web:138]
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

DEFAULT_TACOS_TARGET = 12.0  # % account-level TACoS target band.

# ---------------------------------------------------------------------------
# HELPERS / STYLING
# ---------------------------------------------------------------------------

def get_excel_writer_engine(preferred: str = "openpyxl") -> str:
    try:
        if preferred == "xlsxwriter":
            import xlsxwriter  # noqa: F401
            return "xlsxwriter"
    except Exception:
        pass
    return "openpyxl"


def load_custom_css():
    theme_type = "dark"
    try:
        theme_type = getattr(st.context.theme, "type", "dark")
    except Exception:
        pass

    if str(theme_type).lower() == "light":
        css = """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #e5e7eb 0, #e5e7eb 40%, #ffffff 100%);
        }
        .main { padding-top: 0.5rem; }
        .agency-header {
            background: linear-gradient(135deg, #4f46e5 0, #6366f1 35%, #111827 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.1rem;
            color: #f9fafb;
            box-shadow: 0 18px 40px rgba(15,23,42,0.35);
            border: 1px solid rgba(148,163,184,0.6);
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
            background: linear-gradient(135deg, rgba(59,130,246,0.10), #f9fafb);
            border-left: 4px solid #3b82f6;
            color: #111827;
        }
        .success-box {
            background: linear-gradient(135deg, rgba(34,197,94,0.1), #ecfdf3);
            border-left: 4px solid #22c55e;
            color: #064e3b;
        }
        .warning-box {
            background: linear-gradient(135deg, rgba(250,204,21,0.12), #fefce8);
            border-left: 4px solid #eab308;
            color: #78350f;
        }
        .danger-box {
            background: linear-gradient(135deg, rgba(248,113,113,0.16), #fef2f2);
            border-left: 4px solid #ef4444;
            color: #7f1d1d;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.7);
            overflow: hidden;
            box-shadow: 0 12px 30px rgba(148,163,184,0.6);
        }
        .metric-card {
            background: radial-gradient(circle at top left, #111827, #020617);
            border-radius: 14px;
            padding: 0.9rem 1.0rem;
            border: 1px solid rgba(148,163,184,0.5);
            box-shadow: 0 10px 26px rgba(15,23,42,0.65);
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
    else:
        css = """
        <style>
        .main {
            padding-top: 0.5rem;
            background: radial-gradient(circle at top, #020617 0, #020617 40%, #000000 100%);
        }
        .stApp {
            background: radial-gradient(circle at top, #020617 0, #020617 40%, #000000 100%);
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


# ---------------------------------------------------------------------------
# FILE IO + COLUMN DETECTION (same as before)
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
# ANALYZER (this is the same as your v12 logic)
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
        pause_clicks = max(20.0, base * self.pause_buffer)
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

    # summary, pause helpers, TACoS bands, patterns, harvest, negatives, bids,
    # placements, exports – exactly as in your v12 code
    # (you can keep that part unchanged from the v12 app I sent earlier).

    # -------------------------------------------------------------------
    # Due to space, I won't re-paste every helper again – keep your v12
    # CompleteAnalyzer methods exactly as they are below this comment.
    # -------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SESSION / STATE & UI
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
            <h1>🏢 {st.session_state.agency_name} – Amazon Ads Dashboard Pro v12.2</h1>
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
                st.caption("Use TACoS to decide how aggressive you can be with pausing.")

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
# PAGES – only the metric display changed in dashboard_page
# the rest should be copied from your existing v12 file
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


# keywords_page, bids_page, placement_page, harvest_page, negative_harvest_page,
# negatives_page, exports_page should be exactly the same as in v12.
# You can paste them directly under dashboard_page.

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
