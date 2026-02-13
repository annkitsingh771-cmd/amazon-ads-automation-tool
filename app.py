#!/usr/bin/env python3

import io
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Amazon Ads Agency Dashboard Pro",
    page_icon="🏢",
    layout="wide"
)

# =====================================================
# SAFE HELPERS
# =====================================================

def safe_div(a, b):
    try:
        return a / b if b != 0 else 0
    except:
        return 0

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)

# =====================================================
# ANALYZER CLASS
# =====================================================

class AgencyAnalyzer:

    def __init__(self, df):
        self.df = df.copy()
        self.clean_columns()
        self.clean_data()
        self.calculate_metrics()

    def clean_columns(self):
        self.df.columns = self.df.columns.str.strip()

        mapping = {}
        for col in self.df.columns:
            low = col.lower()

            if "search term" in low:
                mapping[col] = "SearchTerm"
            elif "campaign" in low:
                mapping[col] = "Campaign"
            elif "ad group" in low:
                mapping[col] = "AdGroup"
            elif "spend" in low:
                mapping[col] = "Spend"
            elif "sales" in low:
                mapping[col] = "Sales"
            elif "orders" in low:
                mapping[col] = "Orders"
            elif "click" in low and "through" not in low:
                mapping[col] = "Clicks"
            elif "impression" in low:
                mapping[col] = "Impressions"
            elif "ctr" in low:
                mapping[col] = "CTR"
            elif "cpc" in low:
                mapping[col] = "CPC"
            elif "roas" in low:
                mapping[col] = "ROAS"

        self.df.rename(columns=mapping, inplace=True)

    def clean_data(self):

        numeric_cols = ["Spend", "Sales", "Orders", "Clicks", "Impressions", "CPC"]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = safe_numeric(self.df[col])

        if "CTR" in self.df.columns:
            self.df["CTR"] = (
                self.df["CTR"]
                .astype(str)
                .str.replace("%", "")
            )
            self.df["CTR"] = safe_numeric(self.df["CTR"]) / 100

    def calculate_metrics(self):

        if "Sales" not in self.df.columns:
            self.df["Sales"] = 0

        if "Spend" not in self.df.columns:
            self.df["Spend"] = 0

        if "Orders" not in self.df.columns:
            self.df["Orders"] = 0

        if "Clicks" not in self.df.columns:
            self.df["Clicks"] = 0

        if "Impressions" not in self.df.columns:
            self.df["Impressions"] = 0

        self.df["Profit"] = self.df["Sales"] - self.df["Spend"]
        self.df["ROAS_Calc"] = self.df.apply(lambda r: safe_div(r["Sales"], r["Spend"]), axis=1)
        self.df["ACOS_Calc"] = self.df.apply(lambda r: safe_div(r["Spend"], r["Sales"]) * 100, axis=1)
        self.df["Wastage"] = np.where(self.df["Sales"] == 0, self.df["Spend"], 0)

    # =============================================

    def summary(self):

        total_spend = self.df["Spend"].sum()
        total_sales = self.df["Sales"].sum()

        return {
            "spend": total_spend,
            "sales": total_sales,
            "profit": total_sales - total_spend,
            "orders": self.df["Orders"].sum(),
            "clicks": self.df["Clicks"].sum(),
            "impressions": self.df["Impressions"].sum(),
            "roas": safe_div(total_sales, total_spend),
            "acos": safe_div(total_spend, total_sales) * 100,
            "wastage": self.df["Wastage"].sum(),
            "keywords": len(self.df),
            "campaigns": self.df["Campaign"].nunique() if "Campaign" in self.df.columns else 0
        }

    # =============================================

    def classify(self):

        champions = []
        pause = []
        optimize = []

        for _, r in self.df.iterrows():

            spend = r["Spend"]
            sales = r["Sales"]
            roas = r["ROAS_Calc"]

            item = {
                "Keyword": r.get("SearchTerm", ""),
                "Campaign": r.get("Campaign", ""),
                "Spend": spend,
                "Sales": sales,
                "ROAS": roas
            }

            if roas >= 3 and sales > 0:
                champions.append(item)

            elif spend > 30 and sales == 0:
                pause.append(item)

            elif roas < 1.5 and spend > 10:
                optimize.append(item)

        return champions, pause, optimize

# =====================================================
# UI
# =====================================================

st.title("🏢 Amazon Ads Agency Dashboard Pro")

uploaded_file = st.file_uploader(
    "Upload Amazon Search Term Report",
    type=["xlsx", "xls"]
)

if not uploaded_file:
    st.info("Upload file to begin analysis")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
    analyzer = AgencyAnalyzer(df)
    summary = analyzer.summary()

except Exception as e:
    st.error(f"Error reading file: {str(e)}")
    st.stop()

# =====================================================
# DASHBOARD METRICS
# =====================================================

st.subheader("📊 Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Spend", f"₹{summary['spend']:,.0f}")
c2.metric("Sales", f"₹{summary['sales']:,.0f}")
c3.metric("Profit", f"₹{summary['profit']:,.0f}")
c4.metric("ROAS", f"{summary['roas']:.2f}x")
c5.metric("ACOS", f"{summary['acos']:.1f}%")

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Orders", summary["orders"])
c2.metric("Clicks", summary["clicks"])
c3.metric("Campaigns", summary["campaigns"])
c4.metric("Wastage", f"₹{summary['wastage']:,.0f}")

# =====================================================
# CLASSIFICATION
# =====================================================

champions, pause, optimize = analyzer.classify()

tab1, tab2, tab3 = st.tabs([
    f"🏆 Champions ({len(champions)})",
    f"🚨 Pause Now ({len(pause)})",
    f"⚠️ Optimize ({len(optimize)})"
])

with tab1:
    if champions:
        st.dataframe(pd.DataFrame(champions), use_container_width=True)
    else:
        st.success("No champions yet")

with tab2:
    if pause:
        st.dataframe(pd.DataFrame(pause), use_container_width=True)
    else:
        st.success("No urgent pause keywords")

with tab3:
    if optimize:
        st.dataframe(pd.DataFrame(optimize), use_container_width=True)
    else:
        st.success("No optimization required")

# =====================================================
# EXPORTS
# =====================================================

st.markdown("---")
st.subheader("📥 Export Negative Keywords")

if pause:
    export_df = pd.DataFrame([
        {
            "Campaign": p["Campaign"],
            "Keyword": p["Keyword"],
            "Match Type": "Negative Exact"
        } for p in pause
    ])

    csv_data = export_df.to_csv(index=False)

    st.download_button(
        "Download Negative Keywords CSV",
        csv_data,
        file_name=f"negative_keywords_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("No negative keywords to export")
