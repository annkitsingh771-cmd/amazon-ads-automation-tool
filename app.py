import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Amazon Ads Agency Dashboard Pro",
    layout="wide"
)

# ============================================================
# SAFE HELPERS
# ============================================================

def safe_numeric(series):
    try:
        return pd.to_numeric(series, errors="coerce").fillna(0)
    except:
        return pd.Series([0]*len(series))

def find_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

# ============================================================
# ANALYZER CLASS
# ============================================================

class AgencyAnalyzer:

    def __init__(self, df, client_name):
        self.client_name = client_name
        self.df = df.copy()
        self.clean_columns()
        self.clean_data()
        self.calculate_metrics()

    def clean_columns(self):
        self.df.columns = self.df.columns.str.strip()

    def clean_data(self):

        spend_col = find_column(self.df, ["spend"])
        sales_col = find_column(self.df, ["sales"])
        clicks_col = find_column(self.df, ["click"])
        orders_col = find_column(self.df, ["order"])
        campaign_col = find_column(self.df, ["campaign"])

        self.df["Spend"] = safe_numeric(self.df[spend_col]) if spend_col else 0
        self.df["Sales"] = safe_numeric(self.df[sales_col]) if sales_col else 0
        self.df["Clicks"] = safe_numeric(self.df[clicks_col]) if clicks_col else 0
        self.df["Orders"] = safe_numeric(self.df[orders_col]) if orders_col else 0
        self.df["Campaign"] = self.df[campaign_col] if campaign_col else "Unknown"

    def calculate_metrics(self):

        self.df["ROAS"] = np.where(
            self.df["Spend"] > 0,
            self.df["Sales"] / self.df["Spend"],
            0
        )

        self.df["ACOS"] = np.where(
            self.df["Sales"] > 0,
            (self.df["Spend"] / self.df["Sales"]) * 100,
            0
        )

        self.df["Profit"] = self.df["Sales"] - self.df["Spend"]
        self.df["Wastage"] = np.where(self.df["Sales"] == 0, self.df["Spend"], 0)

    def summary(self):

        total_spend = self.df["Spend"].sum()
        total_sales = self.df["Sales"].sum()
        total_profit = self.df["Profit"].sum()
        total_orders = self.df["Orders"].sum()
        total_wastage = self.df["Wastage"].sum()

        roas = total_sales / total_spend if total_spend > 0 else 0
        acos = (total_spend / total_sales * 100) if total_sales > 0 else 0

        return {
            "Spend": total_spend,
            "Sales": total_sales,
            "Profit": total_profit,
            "Orders": total_orders,
            "ROAS": roas,
            "ACOS": acos,
            "Wastage": total_wastage
        }

# ============================================================
# SESSION STATE (MULTI ACCOUNT)
# ============================================================

if "accounts" not in st.session_state:
    st.session_state.accounts = {}

if "active_account" not in st.session_state:
    st.session_state.active_account = None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👥 Account Manager")

client_name = st.sidebar.text_input("Client Name")
uploaded_file = st.sidebar.file_uploader("Upload Search Term Report", type=["xlsx","xls","csv"])

if st.sidebar.button("Add Account"):

    if client_name and uploaded_file:

        try:
            if uploaded_file.name.endswith("csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            analyzer = AgencyAnalyzer(df, client_name)
            st.session_state.accounts[client_name] = analyzer
            st.session_state.active_account = client_name

            st.sidebar.success("Account Added Successfully")

        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

if st.session_state.accounts:

    selected = st.sidebar.selectbox(
        "Select Account",
        list(st.session_state.accounts.keys())
    )

    st.session_state.active_account = selected

# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("🏢 Amazon Ads Agency Dashboard Pro")

if not st.session_state.active_account:
    st.info("Add a client from sidebar to begin.")
    st.stop()

analyzer = st.session_state.accounts[st.session_state.active_account]
summary = analyzer.summary()

# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Spend", f"₹{summary['Spend']:,.0f}")
col2.metric("Sales", f"₹{summary['Sales']:,.0f}")
col3.metric("Profit", f"₹{summary['Profit']:,.0f}")
col4.metric("ROAS", f"{summary['ROAS']:.2f}x")
col5.metric("ACOS", f"{summary['ACOS']:.1f}%")
col6.metric("Wastage", f"₹{summary['Wastage']:,.0f}")

st.markdown("---")

# ============================================================
# DATA TABLE
# ============================================================

st.subheader("📊 Keyword Level Data")
st.dataframe(analyzer.df, use_container_width=True)

# ============================================================
# EXPORT
# ============================================================

csv_data = analyzer.df.to_csv(index=False)

st.download_button(
    "Download Full Data",
    csv_data,
    file_name=f"{st.session_state.active_account}_analysis.csv",
    mime="text/csv"
)
