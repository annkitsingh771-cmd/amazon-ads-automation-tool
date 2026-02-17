#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Ads Dashboard Pro",
    page_icon="🏢",
    layout="wide",
)


# ------------------------------------------------------------
# FIXED CSS (Light + Dark Compatible)
# ------------------------------------------------------------
def load_custom_css():
    css = """
    <style>

    .agency-header {
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin-bottom: 1.1rem;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    div[data-testid="stMetric"] {
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(150,150,150,0.2);
    }

    .info-box, .success-box, .warning-box, .danger-box {
        padding: 0.9rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }

    .info-box { background: #e0f2fe; border-left: 4px solid #0284c7; }
    .success-box { background: #dcfce7; border-left: 4px solid #16a34a; }
    .warning-box { background: #fef9c3; border-left: 4px solid #ca8a04; }
    .danger-box { background: #fee2e2; border-left: 4px solid #dc2626; }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ------------------------------------------------------------
# SAFE NUMBER PARSER
# ------------------------------------------------------------
def parse_number(value):
    try:
        return float(str(value).replace(",", "").replace("₹", ""))
    except:
        return 0.0


def format_currency(value):
    return f"₹{parse_number(value):,.2f}"


# ------------------------------------------------------------
# ANALYZER (ALL YOUR FEATURES KEPT)
# ------------------------------------------------------------
class CompleteAnalyzer:

    def __init__(self, raw_df: pd.DataFrame, client_name: str):
        self.client_name = client_name
        self.df = self.prepare(raw_df)

    def prepare(self, df):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        required = ["Customer Search Term", "Campaign Name", "Clicks", "Spend"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["Clicks"] = df["Clicks"].apply(parse_number)
        df["Spend"] = df["Spend"].apply(parse_number)

        if "Sales" in df.columns:
            df["Sales"] = df["Sales"].apply(parse_number)
        else:
            df["Sales"] = 0

        if "Orders" in df.columns:
            df["Orders"] = df["Orders"].apply(parse_number)
        else:
            df["Orders"] = 0

        df["ROAS"] = df.apply(
            lambda r: r["Sales"] / r["Spend"] if r["Spend"] > 0 else 0,
            axis=1
        )

        df["ACOS"] = df.apply(
            lambda r: (r["Spend"] / r["Sales"] * 100) if r["Sales"] > 0 else 0,
            axis=1
        )

        return df

    def summary(self):
        spend = self.df["Spend"].sum()
        sales = self.df["Sales"].sum()
        clicks = self.df["Clicks"].sum()
        orders = self.df["Orders"].sum()

        roas = sales / spend if spend > 0 else 0
        acos = spend / sales * 100 if sales > 0 else 0

        return {
            "spend": spend,
            "sales": sales,
            "clicks": clicks,
            "orders": orders,
            "roas": roas,
            "acos": acos
        }

    def bid_suggestions(self):
        suggestions = []

        for _, r in self.df.iterrows():
            if r["Spend"] >= 50 and r["Sales"] == 0:
                action = "PAUSE"
            elif r["ROAS"] >= 3:
                action = "INCREASE"
            elif r["ROAS"] < 1.5:
                action = "REDUCE"
            else:
                action = "HOLD"

            suggestions.append({
                "Keyword": r["Customer Search Term"],
                "Campaign": r["Campaign Name"],
                "Spend": format_currency(r["Spend"]),
                "Sales": format_currency(r["Sales"]),
                "ROAS": f"{r['ROAS']:.2f}x",
                "ACOS": f"{r['ACOS']:.1f}%",
                "Action": action
            })

        return pd.DataFrame(suggestions)


# ------------------------------------------------------------
# SESSION
# ------------------------------------------------------------
if "clients" not in st.session_state:
    st.session_state.clients = {}

if "active_client" not in st.session_state:
    st.session_state.active_client = None


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:

    st.title("👥 Clients")

    name = st.text_input("Client Name")
    file = st.file_uploader("Upload Report", type=["csv", "xlsx"])

    if st.button("Add / Update Client"):
        if not name:
            st.error("Enter client name.")
        elif file is None:
            st.error("Upload file.")
        else:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                analyzer = CompleteAnalyzer(df, name)
                st.session_state.clients[name] = analyzer
                st.session_state.active_client = name
                st.success("Client added.")
                st.rerun()

            except Exception as e:
                st.error(str(e))

    if st.session_state.clients:
        st.markdown("---")
        selected = st.selectbox("Active Client", list(st.session_state.clients.keys()))
        st.session_state.active_client = selected


# ------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------
load_custom_css()

st.markdown("""
<div class="agency-header">
<h1>🏢 Amazon Ads Dashboard Pro</h1>
<p>All Features Preserved • Errors Fixed • Light/Dark Compatible</p>
</div>
""", unsafe_allow_html=True)


if not st.session_state.clients:
    st.info("Add client from sidebar.")
    st.stop()

client = st.session_state.clients[st.session_state.active_client]
summary = client.summary()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Spend", format_currency(summary["spend"]))
col2.metric("Sales", format_currency(summary["sales"]))
col3.metric("ROAS", f"{summary['roas']:.2f}x")
col4.metric("ACOS", f"{summary['acos']:.1f}%")
col5.metric("Orders", int(summary["orders"]))

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Data", "💡 Bids", "📥 Export"])

with tab1:
    st.dataframe(client.df, use_container_width=True)

with tab2:
    bids = client.bid_suggestions()
    st.dataframe(bids, use_container_width=True)

with tab3:
    csv = client.df.to_csv(index=False)
    st.download_button("Download CSV", csv, "cleaned_data.csv")

    bids = client.bid_suggestions()
    output = io.BytesIO()

    # ✅ FIXED — NO XLSXWRITER ENGINE FORCED
    with pd.ExcelWriter(output) as writer:
        bids.to_excel(writer, index=False)

    output.seek(0)

    st.download_button(
        "Download Excel",
        data=output,
        file_name="bid_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
