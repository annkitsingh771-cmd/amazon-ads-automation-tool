import io
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Amazon Ads Automation Tool",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Amazon Ads Automation Tool")
st.caption("Multi-Account PPC Optimization Platform")
st.markdown("---")

# ---------------- FILE UPLOAD ---------------- #

with st.sidebar:
    st.header("📁 Upload Reports")
    uploaded_files = st.file_uploader(
        "Upload Search Term or Targeting Reports",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )

if not uploaded_files:
    st.info("👈 Upload Amazon Ads reports to begin")
    st.stop()

# ---------------- HELPER FUNCTIONS ---------------- #

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def find_column(df, keyword):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)

# ---------------- LOAD FILES ---------------- #

all_dfs = []

for file in uploaded_files:
    df = pd.read_excel(file)
    df = clean_columns(df)
    df["Account Name"] = file.name
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

st.subheader("📋 Columns Detected")
st.write(df.columns.tolist())

# ---------------- DETECT PERFORMANCE COLUMNS ---------------- #

spend_col = find_column(df, "spend")
clicks_col = find_column(df, "click")
impression_col = find_column(df, "impression")
sales_col = find_column(df, "sales")

# ---------------- STRICT VALIDATION ---------------- #

if not spend_col or not clicks_col:
    st.error("""
❌ This report does NOT contain Spend or Clicks column.

You likely downloaded wrong report type.

Please download:

Amazon Ads → Reports → Sponsored Products → Search Term Report
""")
    st.stop()

# ---------------- RENAME SAFELY ---------------- #

df.rename(columns={
    spend_col: "Spend",
    clicks_col: "Clicks"
}, inplace=True)

df["Spend"] = to_numeric(df["Spend"])
df["Clicks"] = to_numeric(df["Clicks"])
df["Impressions"] = to_numeric(df[impression_col]) if impression_col else 0
df["Sales"] = to_numeric(df[sales_col]) if sales_col else 0

# ---------------- ACCOUNT FILTER ---------------- #

account_filter = st.selectbox(
    "Select Account",
    ["All Accounts"] + df["Account Name"].unique().tolist()
)

if account_filter != "All Accounts":
    df = df[df["Account Name"] == account_filter]

st.success(f"✅ Loaded {len(df)} rows")

# ---------------- METRICS ---------------- #

total_spend = df["Spend"].sum()
total_sales = df["Sales"].sum()
total_clicks = df["Clicks"].sum()
total_impressions = df["Impressions"].sum()

roas = total_sales / total_spend if total_spend > 0 else 0
acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0

st.header("📊 Performance Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Spend", f"₹{total_spend:,.0f}")
c2.metric("Total Sales", f"₹{total_sales:,.0f}")
c3.metric("ROAS", f"{roas:.2f}x")
c4.metric("ACOS", f"{acos:.1f}%")

# ---------------- BASIC ANALYSIS ---------------- #

df["ROAS"] = df.apply(
    lambda x: x["Sales"] / x["Spend"] if x["Spend"] > 0 else 0,
    axis=1
)

high = df[df["ROAS"] >= 3]
low = df[df["ROAS"] < 1.5]
waste = df[(df["Sales"] == 0) & (df["Spend"] > 20)]

tab1, tab2, tab3 = st.tabs([
    f"🌟 High ROAS ({len(high)})",
    f"⚠ Low ROAS ({len(low)})",
    f"❌ Wastage ({len(waste)})"
])

with tab1:
    st.dataframe(high)

with tab2:
    st.dataframe(low)

with tab3:
    st.dataframe(waste)
