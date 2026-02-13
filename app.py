import io
from datetime import datetime
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Amazon Ads Automation Tool",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Amazon Ads Automation Tool")
st.caption("AI-Powered PPC Optimization Platform")
st.markdown("---")


# ---------------- HELPER FUNCTIONS ---------------- #

def normalize_columns(df):
    df.columns = df.columns.str.strip()
    return df


def find_column(df, keyword):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ---------------- FILE UPLOAD ---------------- #

with st.sidebar:
    st.header("📁 Upload Reports")
    uploaded_files = st.file_uploader(
        "Upload one or more Search Term Reports",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )

if not uploaded_files:
    st.info("👈 Upload one or more Amazon Search Term Reports to get started")
    st.stop()

# ---------------- LOAD MULTIPLE ACCOUNTS ---------------- #

all_dfs = []

for file in uploaded_files:
    try:
        df = pd.read_excel(file)
        df = normalize_columns(df)
        df["Account Name"] = file.name
        all_dfs.append(df)
    except Exception as e:
        st.error(f"Error reading file {file.name}: {e}")

if not all_dfs:
    st.stop()

df = pd.concat(all_dfs, ignore_index=True)

# ---------------- FLEXIBLE COLUMN DETECTION ---------------- #

customer_col = find_column(df, "customer search")
spend_col = find_column(df, "spend")
clicks_col = find_column(df, "click")
impression_col = find_column(df, "impression")
sales_col = find_column(df, "sales")
orders_col = find_column(df, "order")
cpc_col = find_column(df, "cost per click")
ctr_col = find_column(df, "ctr")
campaign_col = find_column(df, "campaign")
adgroup_col = find_column(df, "ad group")
matchtype_col = find_column(df, "match")

required_cols = {
    "Customer Search Term": customer_col,
    "Spend": spend_col,
    "Clicks": clicks_col,
}

missing = [k for k, v in required_cols.items() if v is None]

if missing:
    st.error(f"❌ Missing required columns: {', '.join(missing)}")
    st.write("Available columns in file:")
    st.write(df.columns.tolist())
    st.stop()

# Rename dynamically
df.rename(columns={
    customer_col: "Customer Search Term",
    spend_col: "Spend",
    clicks_col: "Clicks"
}, inplace=True)

# Convert numeric safely
df["Spend"] = safe_numeric(df["Spend"])
df["Clicks"] = safe_numeric(df["Clicks"])

if impression_col:
    df["Impressions"] = safe_numeric(df[impression_col])
else:
    df["Impressions"] = 0

if sales_col:
    df["Sales"] = safe_numeric(df[sales_col])
else:
    df["Sales"] = 0

if orders_col:
    df["Orders"] = safe_numeric(df[orders_col])
else:
    df["Orders"] = 0

if cpc_col:
    df["CPC"] = safe_numeric(df[cpc_col])
else:
    df["CPC"] = 0

if ctr_col:
    df["CTR"] = df[ctr_col].astype(str).str.replace("%", "")
    df["CTR"] = safe_numeric(df["CTR"]) / 100
else:
    df["CTR"] = 0

if campaign_col:
    df["Campaign"] = df[campaign_col]
else:
    df["Campaign"] = ""

if adgroup_col:
    df["Ad Group"] = df[adgroup_col]
else:
    df["Ad Group"] = ""

if matchtype_col:
    df["Match Type"] = df[matchtype_col]
else:
    df["Match Type"] = ""

# ---------------- ACCOUNT FILTER ---------------- #

account_filter = st.selectbox(
    "Select Account",
    ["All Accounts"] + df["Account Name"].unique().tolist()
)

if account_filter != "All Accounts":
    df = df[df["Account Name"] == account_filter]

st.success(f"✅ Successfully analyzed {len(df)} keywords")

# ---------------- OVERVIEW METRICS ---------------- #

total_spend = df["Spend"].sum()
total_sales = df["Sales"].sum()
total_clicks = df["Clicks"].sum()
total_impressions = df["Impressions"].sum()
total_orders = df["Orders"].sum()

avg_cpc = df["CPC"].mean() if total_clicks > 0 else 0
avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
overall_roas = total_sales / total_spend if total_spend > 0 else 0
overall_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
conversion_rate = (total_orders / total_clicks * 100) if total_clicks > 0 else 0

st.header("📊 Overview Metrics")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Spend", f"₹{total_spend:,.0f}")
    st.metric("Avg CPC", f"₹{avg_cpc:.2f}")

with c2:
    st.metric("Total Sales", f"₹{total_sales:,.0f}")
    st.metric("Total Orders", f"{total_orders:,}")

with c3:
    st.metric("Overall ROAS", f"{overall_roas:.2f}x")
    st.metric("Avg CTR", f"{avg_ctr:.2f}%")

with c4:
    st.metric("Overall ACOS", f"{overall_acos:.1f}%")
    st.metric("Conversion Rate", f"{conversion_rate:.2f}%")

st.markdown("---")

# ---------------- KEYWORD CLASSIFICATION ---------------- #

HIGH_ROAS = 3
LOW_ROAS = 1.5

df["ROAS"] = df.apply(
    lambda x: x["Sales"] / x["Spend"] if x["Spend"] > 0 else 0,
    axis=1
)

high_potential = df[(df["ROAS"] >= HIGH_ROAS) & (df["Spend"] > 10)]
low_potential = df[(df["ROAS"] < LOW_ROAS) & (df["Spend"] > 10)]
wastage = df[(df["Sales"] == 0) & (df["Spend"] > 20)]

tab1, tab2, tab3 = st.tabs([
    f"🌟 High Potential ({len(high_potential)})",
    f"⚠️ Low Potential ({len(low_potential)})",
    f"❌ Wastage ({len(wastage)})"
])

with tab1:
    st.dataframe(high_potential, use_container_width=True)

with tab2:
    st.dataframe(low_potential, use_container_width=True)

with tab3:
    st.dataframe(wastage, use_container_width=True)

st.markdown("---")

# ---------------- EXPORT NEGATIVE KEYWORDS ---------------- #

if len(wastage) > 0:
    output = io.BytesIO()

    neg_df = wastage[["Campaign", "Customer Search Term"]].copy()
    neg_df["Match Type"] = "Negative Exact"

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        neg_df.to_excel(writer, index=False, sheet_name="Negative Keywords")

    output.seek(0)

    st.download_button(
        "📥 Download Negative Keywords",
        data=output,
        file_name=f"negative_keywords_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
