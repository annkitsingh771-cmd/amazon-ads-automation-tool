import io
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st


# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Amazon Ads Dashboard Pro",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def safe_float(x):
    try:
        return float(str(x).replace(",", "").replace("₹", ""))
    except:
        return 0.0


def format_currency(x):
    return f"₹{safe_float(x):,.2f}"


def add_serial(df):
    df = df.reset_index(drop=True)
    df.insert(0, "S.No", range(1, len(df) + 1))
    return df


# ------------------------------------------------
# Analyzer
# ------------------------------------------------
class Analyzer:

    def __init__(self, df: pd.DataFrame, client: str):
        self.client = client
        self.df = self.prepare(df)

    def prepare(self, df):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        required = ["Customer Search Term", "Campaign Name", "Clicks", "Spend"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["Clicks"] = df["Clicks"].apply(safe_float)
        df["Spend"] = df["Spend"].apply(safe_float)

        if "Sales" in df.columns:
            df["Sales"] = df["Sales"].apply(safe_float)
        else:
            df["Sales"] = 0

        if "Orders" in df.columns:
            df["Orders"] = df["Orders"].apply(safe_float)
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
        df = self.df
        spend = df["Spend"].sum()
        sales = df["Sales"].sum()
        clicks = df["Clicks"].sum()
        orders = df["Orders"].sum()

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
            if r["Spend"] >= 30 and r["Sales"] == 0:
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


# ------------------------------------------------
# Session Init
# ------------------------------------------------
if "clients" not in st.session_state:
    st.session_state.clients: Dict[str, Analyzer] = {}

if "active_client" not in st.session_state:
    st.session_state.active_client = None


# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:
    st.title("👥 Clients")

    name = st.text_input("Client Name")

    file = st.file_uploader("Upload Search Term Report", type=["csv", "xlsx"])

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

                analyzer = Analyzer(df, name)
                st.session_state.clients[name] = analyzer
                st.session_state.active_client = name
                st.success("Client added successfully.")
                st.rerun()

            except Exception as e:
                st.error(str(e))

    if st.session_state.clients:
        st.markdown("---")
        selected = st.selectbox(
            "Active Client",
            list(st.session_state.clients.keys())
        )
        st.session_state.active_client = selected


# ------------------------------------------------
# Main App
# ------------------------------------------------
st.title("📊 Amazon Ads Dashboard Pro")

if not st.session_state.clients:
    st.info("Add a client from sidebar.")
    st.stop()

if not st.session_state.active_client:
    st.warning("Select active client.")
    st.stop()

client = st.session_state.clients[st.session_state.active_client]
summary = client.summary()

# ------------------------------------------------
# Metrics
# ------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Spend", format_currency(summary["spend"]))
col2.metric("Sales", format_currency(summary["sales"]))
col3.metric("ROAS", f"{summary['roas']:.2f}x")
col4.metric("ACOS", f"{summary['acos']:.1f}%")
col5.metric("Orders", int(summary["orders"]))

st.markdown("---")

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data", "💡 Bid Suggestions", "📥 Export"])

with tab1:
    st.dataframe(add_serial(client.df), use_container_width=True)

with tab2:
    bids = client.bid_suggestions()
    st.dataframe(add_serial(bids), use_container_width=True)

with tab3:
    csv = client.df.to_csv(index=False)
    st.download_button(
        "Download Cleaned CSV",
        data=csv,
        file_name=f"{client.client}_cleaned_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    bids = client.bid_suggestions()
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        bids.to_excel(writer, index=False)

    output.seek(0)

    st.download_button(
        "Download Bid Excel",
        data=output,
        file_name=f"{client.client}_bids_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
