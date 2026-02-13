import io
from datetime import datetime

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="Amazon Ads Automation Tool",
    page_icon="🚀",
    layout="wide",
)

# Custom CSS focused on readability in dark and light themes
st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        min-height: 92px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 600;
        font-size: 0.85rem;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700;
        font-size: 1.35rem;
    }

    div[data-testid="stMetricDelta"] {
        color: #22c55e !important;
        font-weight: 600;
    }

    /* Alert boxes */
    .success-box,
    .warning-box,
    .danger-box {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        font-weight: 500;
    }

    .success-box {
        background: rgba(22, 163, 74, 0.15);
        border-left: 5px solid #22c55e;
        color: #dcfce7;
    }

    .warning-box {
        background: rgba(234, 179, 8, 0.15);
        border-left: 5px solid #facc15;
        color: #fef9c3;
    }

    .danger-box {
        background: rgba(220, 38, 38, 0.16);
        border-left: 5px solid #ef4444;
        color: #fee2e2;
    }

    /* Tables readability */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Section spacing */
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


class AmazonAdsAnalyzer:
    """Core analysis engine for Amazon search term reports"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._normalize_column_names()
        self._clean_data()

    def _normalize_column_names(self):
        """Normalize column names to handle variations"""
        self.df.columns = self.df.columns.str.strip()

        column_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower().strip()

            if "acos" in col_lower and "total" in col_lower:
                column_mapping[col] = "Total Advertising Cost of Sales (ACOS)"
            elif "roas" in col_lower and "total" in col_lower:
                column_mapping[col] = "Total Return on Advertising Spend (ROAS)"
            elif "tacos" in col_lower:
                column_mapping[col] = "TACOS"
            elif "7 day total sales" in col_lower or "total sales" in col_lower:
                column_mapping[col] = "7 Day Total Sales"
            elif "7 day total orders" in col_lower:
                column_mapping[col] = "7 Day Total Orders"
            elif "7 day conversion" in col_lower:
                column_mapping[col] = "7 Day Conversion Rate"

        if column_mapping:
            self.df.rename(columns=column_mapping, inplace=True)

    def _clean_data(self):
        """Clean and prepare data"""
        for col in self.df.columns:
            if "Rate" in col or "CTR" in col:
                if self.df[col].dtype == "object":
                    try:
                        self.df[col] = self.df[col].str.rstrip("%").astype("float") / 100.0
                    except Exception:
                        pass

        sales_col = self._find_column(["7 Day Total Sales", "Total Sales"])
        acos_col = self._find_column(["Total Advertising Cost of Sales (ACOS)", "ACOS"])
        roas_col = self._find_column(["Total Return on Advertising Spend (ROAS)", "ROAS"])

        if sales_col:
            self.df[sales_col] = self.df[sales_col].fillna(0)
        if acos_col:
            self.df[acos_col] = self.df[acos_col].fillna(0)
        if roas_col:
            self.df[roas_col] = self.df[roas_col].fillna(0)

    def _find_column(self, possible_names):
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None

    def get_overview_metrics(self):
        total_spend = self.df["Spend"].sum()

        sales_col = self._find_column(["7 Day Total Sales", "Total Sales"])
        total_sales = self.df[sales_col].sum() if sales_col else 0

        total_clicks = self.df["Clicks"].sum()
        total_impressions = self.df["Impressions"].sum()

        orders_col = self._find_column(["7 Day Total Orders", "7 Day Total Orders (#)"])
        total_orders = self.df[orders_col].sum() if orders_col else 0

        avg_cpc = self.df["Cost Per Click (CPC)"].mean()
        avg_ctr = self.df["Click-Through Rate (CTR)"].mean() * 100

        overall_roas = total_sales / total_spend if total_spend > 0 else 0
        overall_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        conversion_rate = (total_orders / total_clicks * 100) if total_clicks > 0 else 0

        return {
            "total_spend": total_spend,
            "total_sales": total_sales,
            "total_clicks": total_clicks,
            "total_impressions": total_impressions,
            "total_orders": total_orders,
            "avg_cpc": avg_cpc,
            "avg_ctr": avg_ctr,
            "overall_roas": overall_roas,
            "overall_acos": overall_acos,
            "conversion_rate": conversion_rate,
        }

    def identify_keyword_types(self):
        sales_col = self._find_column(["7 Day Total Sales", "Total Sales"])
        roas_col = self._find_column(["Total Return on Advertising Spend (ROAS)", "ROAS"])

        HIGH_ROAS_THRESHOLD = 3.0
        LOW_ROAS_THRESHOLD = 1.5
        MIN_SPEND_FOR_EVALUATION = 10
        WASTAGE_SPEND_THRESHOLD = 20

        high_potential, low_potential, wastage, future_potential = [], [], [], []

        for _, row in self.df.iterrows():
            keyword = row["Customer Search Term"]
            spend = row["Spend"]
            sales = row[sales_col] if sales_col else 0
            roas = row[roas_col] if roas_col else 0
            clicks = row["Clicks"]
            ctr = row["Click-Through Rate (CTR)"]

            keyword_data = {
                "Keyword": keyword,
                "Spend": spend,
                "Sales": sales,
                "ROAS": roas,
                "Clicks": clicks,
                "CTR (%)": ctr * 100,
                "Campaign": row["Campaign Name"],
                "Match Type": row["Match Type"],
                "Reason": "",
            }

            if spend >= WASTAGE_SPEND_THRESHOLD and sales == 0:
                keyword_data["Reason"] = f"₹{spend:.0f} spent with zero sales"
                wastage.append(keyword_data)
            elif roas >= HIGH_ROAS_THRESHOLD and spend >= MIN_SPEND_FOR_EVALUATION:
                keyword_data["Reason"] = f"Strong ROAS ({roas:.2f}x) with proven sales"
                high_potential.append(keyword_data)
            elif spend >= MIN_SPEND_FOR_EVALUATION and (roas < LOW_ROAS_THRESHOLD or sales == 0):
                keyword_data["Reason"] = f"Low ROAS ({roas:.2f}x) despite ₹{spend:.0f} spend"
                low_potential.append(keyword_data)
            elif spend < MIN_SPEND_FOR_EVALUATION and ctr > 0.02 and clicks > 5:
                keyword_data["Reason"] = f"High CTR ({ctr*100:.2f}%) with limited data"
                future_potential.append(keyword_data)

        return {
            "high_potential": sorted(high_potential, key=lambda x: x["ROAS"], reverse=True),
            "low_potential": sorted(low_potential, key=lambda x: x["Spend"], reverse=True),
            "wastage": sorted(wastage, key=lambda x: x["Spend"], reverse=True),
            "future_potential": sorted(future_potential, key=lambda x: x["CTR (%)"], reverse=True),
        }

    def get_bid_suggestions(self):
        suggestions = []

        sales_col = self._find_column(["7 Day Total Sales", "Total Sales"])
        roas_col = self._find_column(["Total Return on Advertising Spend (ROAS)", "ROAS"])

        for _, row in self.df.iterrows():
            current_cpc = row["Cost Per Click (CPC)"]
            roas = row[roas_col] if roas_col else 0
            spend = row["Spend"]
            sales = row[sales_col] if sales_col else 0

            if spend < 5:
                continue

            suggestion = {
                "Keyword": row["Customer Search Term"],
                "Campaign": row["Campaign Name"],
                "Ad Group": row["Ad Group Name"],
                "Current CPC": current_cpc,
                "Spend": spend,
                "ROAS": roas,
                "Action": "",
                "Suggested Bid": 0,
                "Change (%)": 0,
                "Reason": "",
            }

            if roas >= 3.5:
                suggestion.update(
                    {
                        "Action": "INCREASE",
                        "Suggested Bid": current_cpc * 1.3,
                        "Change (%)": 30,
                        "Reason": f"High ROAS ({roas:.2f}x) - scale winning keyword",
                    }
                )
            elif roas >= 2.5:
                suggestion.update(
                    {
                        "Action": "INCREASE",
                        "Suggested Bid": current_cpc * 1.15,
                        "Change (%)": 15,
                        "Reason": f"Good ROAS ({roas:.2f}x) - moderate increase",
                    }
                )
            elif sales == 0 and spend > 20:
                suggestion.update(
                    {
                        "Action": "PAUSE",
                        "Suggested Bid": 0,
                        "Change (%)": -100,
                        "Reason": f"₹{spend:.0f} spent with no sales - pause keyword",
                    }
                )
            elif roas < 1.0 and spend > 10:
                suggestion.update(
                    {
                        "Action": "DECREASE",
                        "Suggested Bid": current_cpc * 0.7,
                        "Change (%)": -30,
                        "Reason": f"Poor ROAS ({roas:.2f}x) - reduce spend",
                    }
                )
            else:
                continue

            suggestions.append(suggestion)

        return sorted(suggestions, key=lambda x: x["Spend"], reverse=True)


def main():
    st.title("🚀 Amazon Ads Automation Tool")
    st.caption("AI-Powered PPC Optimization Platform")
    st.markdown("---")

    with st.sidebar:
        st.header("📁 Upload Report")
        st.markdown("Upload your Amazon Sponsored Products Search Term Report")

        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=["xlsx", "xls"],
            help="Download from Amazon Ads Console → Reports → Search Term Report",
        )

    if uploaded_file is None:
        st.info("👈 Upload your Amazon search term report to get started")
        st.stop()

    try:
        df = pd.read_excel(uploaded_file)

        required_cols = ["Customer Search Term", "Spend", "Clicks"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()

        analyzer = AmazonAdsAnalyzer(df)
        st.success(f"✅ Successfully analyzed {len(df)} keywords from your report!")

        st.header("📊 Overview Metrics")
        metrics = analyzer.get_overview_metrics()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Spend", f"₹{metrics['total_spend']:,.0f}")
            st.metric("Avg CPC", f"₹{metrics['avg_cpc']:.2f}")
        with c2:
            st.metric("Total Sales", f"₹{metrics['total_sales']:,.0f}")
            st.metric("Total Orders", f"{metrics['total_orders']:,}")
        with c3:
            st.metric("Overall ROAS", f"{metrics['overall_roas']:.2f}x")
            st.metric("Avg CTR", f"{metrics['avg_ctr']:.2f}%")
        with c4:
            st.metric("Overall ACOS", f"{metrics['overall_acos']:.1f}%")
            st.metric("Conversion Rate", f"{metrics['conversion_rate']:.2f}%")

        st.markdown("---")
        st.header("🎯 Keyword Classification")
        classification = analyzer.identify_keyword_types()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                f"🌟 High Potential ({len(classification['high_potential'])})",
                f"🔮 Future Potential ({len(classification['future_potential'])})",
                f"⚠️ Low Potential ({len(classification['low_potential'])})",
                f"❌ Wastage ({len(classification['wastage'])})",
            ]
        )

        with tab1:
            if classification["high_potential"]:
                st.markdown(
                    '<div class="success-box">These keywords are performing well! Consider increasing bids.</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(classification["high_potential"]), use_container_width=True, hide_index=True)
            else:
                st.info("No high potential keywords found")

        with tab2:
            if classification["future_potential"]:
                st.markdown(
                    '<div class="success-box">These keywords show promise but need more data.</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(classification["future_potential"]), use_container_width=True, hide_index=True)
            else:
                st.info("No future potential keywords found")

        with tab3:
            if classification["low_potential"]:
                st.markdown(
                    '<div class="warning-box">These keywords are underperforming. Consider reducing bids or pausing.</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(classification["low_potential"]), use_container_width=True, hide_index=True)
            else:
                st.info("No low potential keywords found")

        with tab4:
            if classification["wastage"]:
                st.markdown(
                    '<div class="danger-box">These keywords are wasting budget with zero sales!</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(pd.DataFrame(classification["wastage"]), use_container_width=True, hide_index=True)
            else:
                st.success("No wastage keywords - great job!")

        st.markdown("---")
        st.header("💡 Bid Optimization Suggestions")
        bid_suggestions = analyzer.get_bid_suggestions()

        if bid_suggestions:
            st.info(f"Found {len(bid_suggestions)} bid adjustment recommendations")
            st.dataframe(pd.DataFrame(bid_suggestions), use_container_width=True, hide_index=True)
        else:
            st.success("No bid adjustments needed at this time!")

        st.markdown("---")
        st.header("📥 Export Bulk Upload Sheets")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Negative Keywords")
            negative_keywords = classification["wastage"] + classification["low_potential"]
            if negative_keywords:
                df_negatives = pd.DataFrame(
                    [
                        {
                            "Campaign": kw["Campaign"],
                            "Ad Group": "",
                            "Keyword": kw["Keyword"],
                            "Match Type": "Negative Exact",
                            "Status": "Enabled",
                        }
                        for kw in negative_keywords
                    ]
                )

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df_negatives.to_excel(writer, index=False, sheet_name="Negative Keywords")
                output.seek(0)

                st.download_button(
                    "📥 Download Negative Keywords Sheet",
                    data=output,
                    file_name=f"negative_keywords_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                st.info("No negative keywords to export")

        with col2:
            st.subheader("Bid Adjustments")
            if bid_suggestions:
                bulk_bids = [
                    {
                        "Campaign": s["Campaign"],
                        "Ad Group": s["Ad Group"],
                        "Keyword": s["Keyword"],
                        "Current Bid": s["Current CPC"],
                        "Suggested Bid": s["Suggested Bid"],
                        "Change %": s["Change (%)"],
                        "Reason": s["Reason"],
                    }
                    for s in bid_suggestions
                    if s["Action"] != "PAUSE"
                ]
                if bulk_bids:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        pd.DataFrame(bulk_bids).to_excel(writer, index=False, sheet_name="Bid Adjustments")
                    output.seek(0)
                    st.download_button(
                        "📥 Download Bid Adjustments Sheet",
                        data=output,
                        file_name=f"bid_adjustments_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                else:
                    st.info("No bid adjustments to export")
            else:
                st.info("No bid adjustments to export")

    except Exception as e:
        st.error(f"❌ Error analyzing file: {e}")


if __name__ == "__main__":
    main()
