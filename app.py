import io
from datetime import datetime
import pandas as pd
import streamlit as st

# ---------------- PAGE SETUP ---------------- #

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
        "Upload Search Term OR Targeting Reports",
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

def find_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

def to_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)

# ---------------- LOAD MULTIPLE FILES ---------------- #

all_dfs = []

for file in uploaded_files:
    try:
        df = pd.read_excel(file)
        df = clean_columns(df)
        df["Account Name"] = file.name
        all_dfs.append(df)
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

if not all_dfs:
    st.stop()

df = pd.concat(all_dfs, ignore_index=True)

# ---------------- AUTO COLUMN DETECTION ---------------- #

search_term_col = find_column(df, ["customer search", "search term", "targeting"])
spend_col = find_column(df, ["spend"])
clicks_col = find_column(df, ["click"])
impression_col = find_column(df, ["impression"])
sales_col = find_column(df, ["sales"])
orders_col = find_column(df, ["order"])
cpc_col = find_column(df, ["cost per click"])
campaign_col = find_column(df, ["campaign"])
adgroup_col = find_column(df, ["ad group"])

# ---------------- CHECK ESSENTIAL COLUMNS ---------------- #

if not spend_col or not clicks_col:
    st.error("❌ Spend or Clicks column not found in report.")
    st.write("Columns detected:")
    st.write(df.columns.tolist())
    st.stop()

# If no search term column found, fallback to Targeting
if not search_term_col:
    st.warning("⚠ Search Term column not found. Using Targeting column instead.")
    search_term_col = find_column(df, ["targeting"])

# Rename safely
df.rename(columns={
    search_term_col: "Keyword" if search_term_col else None,
    spend_col: "Spend",
    clicks_col: "Clicks"
}, inplace=True)

# Create fallback if keyword still missing
if "Keyword" not in df.columns:
    df["Keyword"] = "Unknown"

# ---------------- NUMERIC CLEANING ---------------- #

df["Spend"] = to_numeric(df["Spend"])
df["Clicks"] = to_numeric(df["Clicks"])

df["Impressions"] = to_numeric(df[impression_col]) if impression_col else 0
df["Sales"] = to_numeric(df[sales_col]) if sales_col else 0
df["Orders"] = to_numeric(df[orders_col]) if orders_col else 0
df["CPC"] = to_numeric(df[cpc_col]) if cpc_col else 0

df["Campaign"] = df[campaign_col] if campaign_col else ""
df["Ad Group"] = df[adgroup_col] if adgroup_col else ""

# ---------------- ACCOUNT FILTER ---------------- #

account_filter = st.selectbox(
    "Select Account",
    ["All Accounts"] + df["Account Name"].unique().tolist()
)

if account_filter != "All Accounts":
    df = df[df["Account Name"] == account_filter]

st.success(f"✅ Loaded {len(df)} rows successfully")

# ---------------- METRICS ---------------- #

total_spend = df["Spend"].sum()
total_sales = df["Sales"].sum()
total_clicks = df["Clicks"].sum()
total_impressions = df["Impressions"].sum()
total_orders = df["Orders"].sum()

roas = total_sales / total_spend if total_spend > 0 else 0
acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
conversion = (total_orders / total_clicks * 100) if total_clicks > 0 else 0

st.header("📊 Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Spend", f"₹{total_spend:,.0f}")
c2.metric("Total Sales", f"₹{total_sales:,.0f}")
c3.metric("ROAS", f"{roas:.2f}x")
c4.metric("ACOS", f"{acos:.1f}%")

st.markdown("---")

# ---------------- KEYWORD ANALYSIS ---------------- #

df["ROAS"] = df.apply(
    lambda x: x["Sales"] / x["Spend"] if x["Spend"] > 0 else 0,
    axis=1
)

high = df[(df["ROAS"] >= 3) & (df["Spend"] > 10)]
low = df[(df["ROAS"] < 1.5) & (df["Spend"] > 10)]
waste = df[(df["Sales"] == 0) & (df["Spend"] > 20)]

tab1, tab2, tab3 = st.tabs([
    f"🌟 High ROAS ({len(high)})",
    f"⚠ Low ROAS ({len(low)})",
    f"❌ Wastage ({len(waste)})"
])

with tab1:
    st.dataframe(high, use_container_width=True)

with tab2:
    st.dataframe(low, use_container_width=True)

with tab3:
    st.dataframe(waste, use_container_width=True)

# ---------------- EXPORT NEGATIVES ---------------- #

if len(waste) > 0:
    output = io.BytesIO()

    neg_df = waste[["Campaign", "Keyword"]].copy()
    neg_df["Match Type"] = "Negative Exact"

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        neg_df.to_excel(writer, index=False, sheet_name="Negative Keywords")

    output.seek(0)

    st.download_button(
        "📥 Download Negative Keywords Sheet",
        data=output,
        file_name=f"negative_keywords_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
