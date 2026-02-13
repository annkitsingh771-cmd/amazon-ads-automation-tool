#!/usr/bin/env python3
import io
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Amazon Ads Agency Dashboard Pro",
    page_icon="🏢",
    layout="wide"
)

# ==========================
# SAFE DATA ANALYZER
# ==========================

class AgencyAnalyzer:

    def __init__(self, df: pd.DataFrame, client_name: str):
        self.client_name = client_name
        self.df = df.copy()
        self._normalize_columns()
        self._clean_data()
        self._enrich_data()

    # --------------------------
    # COLUMN NORMALIZATION
    # --------------------------

    def _normalize_columns(self):
        self.df.columns = self.df.columns.str.strip()

        mapping = {}

        for col in self.df.columns:
            c = col.lower()

            if "sales" in c:
                mapping[col] = "Sales"
            elif "spend" in c:
                mapping[col] = "Spend"
            elif "click" in c:
                mapping[col] = "Clicks"
            elif "impression" in c:
                mapping[col] = "Impressions"
            elif "order" in c:
                mapping[col] = "Orders"
            elif "roas" in c:
                mapping[col] = "ROAS"
            elif "acos" in c:
                mapping[col] = "ACOS"
            elif "ctr" in c:
                mapping[col] = "CTR"
            elif "cost per click" in c or "cpc" in c:
                mapping[col] = "CPC"
            elif "customer search term" in c:
                mapping[col] = "Keyword"
            elif "campaign" in c:
                mapping[col] = "Campaign Name"
            elif "match" in c:
                mapping[col] = "Match Type"

        self.df.rename(columns=mapping, inplace=True)

    # --------------------------
    # SAFE CLEANING
    # --------------------------

    def _clean_data(self):

        numeric_cols = [
            "Sales", "Spend", "Orders",
            "Clicks", "Impressions", "CPC",
            "ROAS", "ACOS"
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col], errors="coerce"
                ).fillna(0)
            else:
                self.df[col] = 0

        if "CTR" in self.df.columns:
            self.df["CTR"] = (
                self.df["CTR"]
                .astype(str)
                .str.replace("%", "", regex=False)
            )
            self.df["CTR"] = pd.to_numeric(
                self.df["CTR"], errors="coerce"
            ).fillna(0) / 100
        else:
            self.df["CTR"] = 0

    # --------------------------
    # ENRICH DATA
    # --------------------------

    def _enrich_data(self):

        self.df["Profit"] = self.df["Sales"] - self.df["Spend"]

        self.df["Wastage"] = np.where(
            self.df["Sales"] == 0,
            self.df["Spend"],
            0
        )

        self.df["ROAS_Calc"] = np.where(
            self.df["Spend"] > 0,
            self.df["Sales"] / self.df["Spend"],
            0
        )

        self.df["ACOS_Calc"] = np.where(
            self.df["Sales"] > 0,
            (self.df["Spend"] / self.df["Sales"]) * 100,
            0
        )

    # --------------------------
    # SUMMARY
    # --------------------------

    def get_summary(self):

        spend = float(self.df["Spend"].sum())
        sales = float(self.df["Sales"].sum())

        return {
            "Spend": spend,
            "Sales": sales,
            "Profit": float(self.df["Profit"].sum()),
            "Orders": int(self.df["Orders"].sum()),
            "Clicks": int(self.df["Clicks"].sum()),
            "Impressions": int(self.df["Impressions"].sum()),
            "ROAS": sales / spend if spend > 0 else 0,
            "ACOS": (spend / sales * 100) if sales > 0 else 0,
            "Wastage": float(self.df["Wastage"].sum()),
            "Keywords": len(self.df),
            "Campaigns": int(self.df["Campaign Name"].nunique())
        }

    # --------------------------
    # HEALTH SCORE
    # --------------------------

    def get_health_score(self):

        s = self.get_summary()
        score = 0

        if s["ROAS"] >= 3:
            score += 40
        elif s["ROAS"] >= 2:
            score += 30
        elif s["ROAS"] >= 1:
            score += 15

        wastage_pct = (
            s["Wastage"] / s["Spend"] * 100
            if s["Spend"] > 0 else 0
        )

        if wastage_pct <= 10:
            score += 30
        elif wastage_pct <= 20:
            score += 20
        elif wastage_pct <= 30:
            score += 10

        ctr_avg = float(self.df["CTR"].mean()) * 100

        if ctr_avg >= 5:
            score += 30
        elif ctr_avg >= 3:
            score += 20
        elif ctr_avg >= 1:
            score += 10

        return min(score, 100)

    # --------------------------
    # CLASSIFICATION
    # --------------------------

    def classify_keywords(self):

        categories = defaultdict(list)

        for _, row in self.df.iterrows():

            spend = row["Spend"]
            sales = row["Sales"]
            roas = row["ROAS_Calc"]

            kw = row.get("Keyword", "Unknown")

            data = {
                "Keyword": kw,
                "Spend": spend,
                "Sales": sales,
                "ROAS": roas
            }

            if roas >= 3 and spend > 10:
                categories["Champions"].append(data)

            elif spend > 25 and sales == 0:
                categories["Pause Now"].append(data)

            elif roas < 1 and spend > 10:
                categories["Needs Optimization"].append(data)

            else:
                categories["Monitor"].append(data)

        return categories


# ==========================
# SESSION STATE
# ==========================

if "clients" not in st.session_state:
    st.session_state.clients = {}

if "active_client" not in st.session_state:
    st.session_state.active_client = None


# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.header("👥 Clients")

    client_name = st.text_input("New Client Name")

    uploaded = st.file_uploader(
        "Upload Search Term Report",
        type=["xlsx"]
    )

    if st.button("Add Client"):

        if client_name and uploaded:

            df = pd.read_excel(uploaded, engine="openpyxl")
            df = df.dropna(how="all")

            analyzer = AgencyAnalyzer(df, client_name)

            st.session_state.clients[client_name] = analyzer
            st.session_state.active_client = client_name
            st.success("Client Added")

    if st.session_state.clients:

        selected = st.selectbox(
            "Select Client",
            list(st.session_state.clients.keys())
        )

        st.session_state.active_client = selected


# ==========================
# MAIN DASHBOARD
# ==========================

st.title("🏢 Amazon Ads Agency Dashboard Pro")

if not st.session_state.active_client:
    st.info("Add a client to start.")
    st.stop()

analyzer = st.session_state.clients[
    st.session_state.active_client
]

summary = analyzer.get_summary()
health = analyzer.get_health_score()

# ==========================
# METRICS
# ==========================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Spend", f"₹{summary['Spend']:,.0f}")
col2.metric("Sales", f"₹{summary['Sales']:,.0f}")
col3.metric("Profit", f"₹{summary['Profit']:,.0f}")
col4.metric("ROAS", f"{summary['ROAS']:.2f}x")
col5.metric("Health Score", f"{health}/100")

st.markdown("---")

# ==========================
# TABS
# ==========================

tab1, tab2, tab3 = st.tabs(
    ["📊 Overview", "🎯 Keywords", "📥 Export"]
)

# --------------------------
# OVERVIEW
# --------------------------

with tab1:

    st.subheader("Campaign Stats")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Orders", summary["Orders"])
    col2.metric("Clicks", summary["Clicks"])
    col3.metric("Impressions", summary["Impressions"])
    col4.metric("Wastage", f"₹{summary['Wastage']:,.0f}")

# --------------------------
# KEYWORDS
# --------------------------

with tab2:

    categories = analyzer.classify_keywords()

    for name, data in categories.items():
        st.subheader(name)
        if data:
            st.dataframe(pd.DataFrame(data),
                         use_container_width=True)
        else:
            st.write("None")

# --------------------------
# EXPORT
# --------------------------

with tab3:

    categories = analyzer.classify_keywords()

    if categories["Pause Now"]:

        export_data = []

        for kw in categories["Pause Now"]:
            export_data.append({
                "Keyword": kw["Keyword"],
                "Match Type": "Negative Exact",
                "Status": "Enabled"
            })

        output = io.BytesIO()

        with pd.ExcelWriter(output,
                            engine="xlsxwriter") as writer:
            pd.DataFrame(export_data).to_excel(
                writer,
                index=False
            )

        output.seek(0)

        st.download_button(
            "Download Negative Keywords",
            data=output,
            file_name="negative_keywords.xlsx"
        )
