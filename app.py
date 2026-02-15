#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Ads Agency Dashboard Pro v2.0 - Complete Edition
Fixed version with no syntax errors
"""

import io
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Amazon Ads Agency Dashboard Pro v2.0",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════

def load_custom_css():
    st.markdown("""
        <style>
        .main { padding-top: 1rem; }

        .agency-header {
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            padding: 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
        }

        .client-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }

        .client-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 1.3rem 1.2rem;
            min-height: 105px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
        }

        div[data-testid="stMetricLabel"] {
            color: #cbd5e1 !important;
            font-weight: 600;
            font-size: 0.9rem;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 700;
            font-size: 1.7rem;
        }

        .success-box, .warning-box, .danger-box, .info-box {
            padding: 1.3rem 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .success-box {
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.25) 0%, rgba(22, 163, 74, 0.1) 100%);
            border-left: 5px solid #22c55e;
            color: #dcfce7;
        }

        .warning-box {
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.25) 0%, rgba(234, 179, 8, 0.1) 100%);
            border-left: 5px solid #facc15;
            color: #fef9c3;
        }

        .danger-box {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(220, 38, 38, 0.1) 100%);
            border-left: 5px solid #ef4444;
            color: #fee2e2;
        }

        .info-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(59, 130, 246, 0.1) 100%);
            border-left: 5px solid #3b82f6;
            color: #dbeafe;
        }
        </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class ClientData:
    def __init__(self, name: str, industry: str = "E-commerce", monthly_budget: float = 50000):
        self.name = name
        self.industry = industry
        self.monthly_budget = monthly_budget
        self.analyzer = None
        self.notes = []
        self.added_date = datetime.now()
        self.monthly_fee = 0
        self.contact_email = ""
        self.contact_phone = ""
        self.status = "Active"
        self.target_acos = 30.0
        self.target_roas = 3.0

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'industry': self.industry,
            'monthly_budget': self.monthly_budget,
            'monthly_fee': self.monthly_fee,
            'target_acos': self.target_acos,
            'target_roas': self.target_roas,
            'added_date': self.added_date.isoformat()
        }

# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class CompleteAnalyzer:
    MIN_SPEND_FOR_LOW_POTENTIAL = 50
    MIN_CLICKS_FOR_LOW_POTENTIAL = 10
    MIN_SPEND_FOR_WASTAGE = 100
    MIN_CLICKS_FOR_WASTAGE = 5
    MIN_SPEND_FOR_HIGH_POTENTIAL = 30
    MIN_ORDERS_FOR_HIGH_POTENTIAL = 2
    MIN_CVR_FOR_CHAMPION = 2.0

    def __init__(self, df: pd.DataFrame, client_name: str, target_acos: float = 30.0, target_roas: float = 3.0):
        self.df = df.copy()
        self.client_name = client_name
        self.target_acos = target_acos
        self.target_roas = target_roas
        self._normalize_columns()
        self._clean_data()
        self._enrich_data()

    def _normalize_columns(self):
        self.df.columns = self.df.columns.str.strip()

        column_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower().strip()
            if "acos" in col_lower:
                column_mapping[col] = "ACOS"
            elif "roas" in col_lower:
                column_mapping[col] = "ROAS"
            elif "7 day total sales" in col_lower or ("sales" in col_lower and "7" in col_lower):
                column_mapping[col] = "Sales"
            elif "7 day total orders" in col_lower or "7 day orders" in col_lower:
                column_mapping[col] = "Orders"

        if column_mapping:
            self.df.rename(columns=column_mapping, inplace=True)

    def _clean_data(self):
        percentage_cols = ["CTR", "Conversion_Rate", "ACOS"]
        for col in percentage_cols:
            if col in self.df.columns and self.df[col].dtype == "object":
                try:
                    self.df[col] = self.df[col].astype(str).str.rstrip("%").astype("float") / 100.0
                except:
                    pass

        numeric_cols = ["Sales", "ACOS", "ROAS", "Orders", "Spend", "Clicks", "Impressions", "CPC"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def _enrich_data(self):
        self.df['Client'] = self.client_name
        self.df['Profit'] = self.df['Sales'] - self.df['Spend']
        self.df['Wastage'] = np.where(self.df['Sales'] == 0, self.df['Spend'], 0)

        self.df['CVR'] = np.where(
            self.df['Clicks'] > 0,
            (self.df['Orders'] / self.df['Clicks']) * 100,
            0
        )

        self.df['ACOS_Calc'] = np.where(
            self.df['Sales'] > 0,
            (self.df['Spend'] / self.df['Sales']) * 100,
            999
        )

        self.df['Processed_Date'] = datetime.now()

    def get_client_summary(self) -> Dict:
        total_spend = float(self.df['Spend'].sum())
        total_sales = float(self.df['Sales'].sum())

        return {
            'total_spend': total_spend,
            'total_sales': total_sales,
            'total_profit': float(self.df['Profit'].sum()),
            'total_orders': int(self.df['Orders'].sum()),
            'total_clicks': int(self.df['Clicks'].sum()),
            'total_impressions': int(self.df['Impressions'].sum()),
            'total_wastage': float(self.df['Wastage'].sum()),
            'roas': total_sales / total_spend if total_spend > 0 else 0,
            'acos': (total_spend / total_sales * 100) if total_sales > 0 else 0,
            'avg_cpc': float(self.df['CPC'].mean()),
            'avg_ctr': float(self.df['CTR'].mean() * 100) if 'CTR' in self.df.columns else 0,
            'avg_cvr': float(self.df['CVR'].mean()),
            'conversion_rate': (self.df['Orders'].sum() / self.df['Clicks'].sum() * 100) if self.df['Clicks'].sum() > 0 else 0,
            'keywords_count': len(self.df),
            'campaigns_count': int(self.df['Campaign Name'].nunique())
        }

    def get_health_score(self) -> int:
        summary = self.get_client_summary()
        score = 0

        roas = summary['roas']
        if roas >= 3.5:
            score += 50
        elif roas >= 2.5:
            score += 40
        elif roas >= 1.5:
            score += 25
        elif roas > 0:
            score += 10

        wastage_pct = (summary['total_wastage'] / summary['total_spend'] * 100) if summary['total_spend'] > 0 else 0
        if wastage_pct <= 10:
            score += 30
        elif wastage_pct <= 20:
            score += 20
        elif wastage_pct <= 30:
            score += 10

        avg_ctr = summary['avg_ctr']
        if avg_ctr >= 5.0:
            score += 20
        elif avg_ctr >= 3.0:
            score += 15
        elif avg_ctr >= 1.0:
            score += 10

        return min(score, 100)

    def classify_keywords_improved(self) -> Dict[str, List[Dict]]:
        categories = {
            'high_potential': [],
            'low_potential': [],
            'wastage': [],
            'opportunities': []
        }

        for _, row in self.df.iterrows():
            try:
                spend = float(row.get('Spend', 0))
                sales = float(row.get('Sales', 0))
                roas = float(row.get('ROAS', 0))
                orders = int(row.get('Orders', 0))
                clicks = int(row.get('Clicks', 0))
                cvr = float(row.get('CVR', 0))

                kw_data = {
                    'Keyword': str(row['Customer Search Term']),
                    'Spend': f"₹{spend:.2f}",
                    'Sales': f"₹{sales:.2f}",
                    'ROAS': f"{roas:.2f}x",
                    'Orders': orders,
                    'Clicks': clicks,
                    'CVR': f"{cvr:.2f}%",
                    'Campaign': row['Campaign Name'],
                    'Match Type': row.get('Match Type', 'N/A'),
                    'Reason': ''
                }

                if (roas >= 3.0 and orders >= self.MIN_ORDERS_FOR_HIGH_POTENTIAL and 
                    spend >= self.MIN_SPEND_FOR_HIGH_POTENTIAL and cvr > 0):
                    kw_data['Reason'] = f"High ROAS ({roas:.2f}x), {orders} orders, CVR {cvr:.2f}%"
                    categories['high_potential'].append(kw_data)

                elif (spend >= self.MIN_SPEND_FOR_WASTAGE and sales == 0 and 
                      clicks >= self.MIN_CLICKS_FOR_WASTAGE):
                    kw_data['Reason'] = f"Rs{spend:.0f} spent, {clicks} clicks, ZERO sales"
                    categories['wastage'].append(kw_data)

                elif (spend >= self.MIN_SPEND_FOR_LOW_POTENTIAL and 
                      clicks >= self.MIN_CLICKS_FOR_LOW_POTENTIAL and roas < 1.5):
                    kw_data['Reason'] = f"Poor ROAS ({roas:.2f}x), CVR {cvr:.2f}%"
                    categories['low_potential'].append(kw_data)

                elif spend >= 20 and roas >= 1.5 and roas < 3.0 and clicks >= 5:
                    kw_data['Reason'] = f"Decent ROAS ({roas:.2f}x), optimize"
                    categories['opportunities'].append(kw_data)

            except Exception:
                continue

        return categories

    def get_bid_suggestions_improved(self) -> List[Dict]:
        suggestions = []

        for _, row in self.df.iterrows():
            try:
                spend = float(row.get('Spend', 0))
                sales = float(row.get('Sales', 0))
                roas = float(row.get('ROAS', 0))
                orders = int(row.get('Orders', 0))
                clicks = int(row.get('Clicks', 0))
                cvr = float(row.get('CVR', 0))
                current_cpc = float(row.get('CPC', 0))
                match_type = str(row.get('Match Type', 'N/A'))

                if spend < 30 or clicks < 5:
                    continue

                suggestion = {
                    'Keyword': str(row['Customer Search Term']),
                    'Campaign': row['Campaign Name'],
                    'Ad Group': row.get('Ad Group Name', 'N/A'),
                    'Match Type': match_type,
                    'Current CPC': f"₹{current_cpc:.2f}",
                    'Spend': f"₹{spend:.2f}",
                    'ROAS': f"{roas:.2f}x",
                    'CVR': f"{cvr:.2f}%",
                    'Orders': orders,
                    'Action': '',
                    'Suggested Bid': '',
                    'Change (%)': 0,
                    'Reason': ''
                }

                acos_current = (spend / sales * 100) if sales > 0 else 999

                if roas >= 3.5 and cvr >= self.MIN_CVR_FOR_CHAMPION and orders >= 2:
                    new_bid = current_cpc * 1.25
                    suggestion.update({
                        'Action': 'INCREASE',
                        'Suggested Bid': f"₹{new_bid:.2f}",
                        'Change (%)': 25,
                        'Reason': f"Champion! ROAS {roas:.2f}x, CVR {cvr:.2f}%"
                    })

                elif roas >= self.target_roas and cvr >= 1.0 and orders >= 1:
                    new_bid = current_cpc * 1.15
                    suggestion.update({
                        'Action': 'INCREASE',
                        'Suggested Bid': f"₹{new_bid:.2f}",
                        'Change (%)': 15,
                        'Reason': f"Above target ROAS"
                    })

                elif sales == 0 and spend >= self.MIN_SPEND_FOR_WASTAGE:
                    suggestion.update({
                        'Action': 'PAUSE',
                        'Suggested Bid': '₹0.00',
                        'Change (%)': -100,
                        'Reason': f"Rs{spend:.0f} wasted, ZERO sales"
                    })

                elif roas < 1.5 and spend >= 50:
                    new_bid = current_cpc * 0.7
                    suggestion.update({
                        'Action': 'REDUCE',
                        'Suggested Bid': f"₹{new_bid:.2f}",
                        'Change (%)': -30,
                        'Reason': f"Poor ROAS ({roas:.2f}x)"
                    })

                elif acos_current > self.target_acos and spend >= 50:
                    reduction = min(30, (acos_current - self.target_acos) / self.target_acos * 100)
                    new_bid = current_cpc * (1 - reduction/100)
                    suggestion.update({
                        'Action': 'REDUCE',
                        'Suggested Bid': f"₹{new_bid:.2f}",
                        'Change (%)': -int(reduction),
                        'Reason': f"ACOS {acos_current:.1f}% > Target {self.target_acos:.1f}%"
                    })

                else:
                    continue

                suggestions.append(suggestion)

            except Exception:
                continue

        return sorted(suggestions, key=lambda x: float(x['Spend'].replace('₹','')), reverse=True)

    def get_match_type_performance(self) -> pd.DataFrame:
        if 'Match Type' not in self.df.columns:
            return pd.DataFrame()

        match_perf = self.df.groupby('Match Type').agg({
            'Spend': 'sum',
            'Sales': 'sum',
            'Orders': 'sum',
            'Clicks': 'sum',
            'Impressions': 'sum'
        })

        match_perf['ROAS'] = match_perf['Sales'] / match_perf['Spend']
        match_perf['ACOS'] = (match_perf['Spend'] / match_perf['Sales'] * 100)
        match_perf['CVR'] = (match_perf['Orders'] / match_perf['Clicks'] * 100)
        match_perf['CTR'] = (match_perf['Clicks'] / match_perf['Impressions'] * 100)

        return match_perf

    def generate_client_report(self) -> str:
        summary = self.get_client_summary()
        health = self.get_health_score()
        classification = self.classify_keywords_improved()

        health_status = "EXCELLENT" if health >= 70 else "GOOD" if health >= 50 else "NEEDS ATTENTION"

        report = f"""
================================================================================
                    AMAZON PPC PERFORMANCE REPORT                            
                        Client: {self.client_name}
                        Date: {datetime.now().strftime('%B %d, %Y')}
================================================================================

EXECUTIVE SUMMARY
================================================================================

Campaign Health Score: {health}/100 - {health_status}
Target ACOS: {self.target_acos:.1f}% | Target ROAS: {self.target_roas:.1f}x

FINANCIAL PERFORMANCE
--------------------------------------------------------------------------------
Total Ad Spend:              Rs {summary['total_spend']:>15,.2f}
Total Sales Generated:       Rs {summary['total_sales']:>15,.2f}
Net Profit:                  Rs {summary['total_profit']:>15,.2f}
ROAS:                        {summary['roas']:>16.2f}x
ACOS:                        {summary['acos']:>15.1f}%

ENGAGEMENT METRICS
--------------------------------------------------------------------------------
Total Orders:                {summary['total_orders']:>16,}
Total Clicks:                {summary['total_clicks']:>16,}
Conversion Rate (CVR):       {summary['avg_cvr']:>15.2f}%
Click-Through Rate:          {summary['avg_ctr']:>15.2f}%

KEYWORD PERFORMANCE
--------------------------------------------------------------------------------
High Potential:              {len(classification['high_potential']):>16}
Opportunities:               {len(classification['opportunities']):>16}
Low Potential:               {len(classification['low_potential']):>16}
Wastage:                     {len(classification['wastage']):>16}

RECOMMENDATIONS
--------------------------------------------------------------------------------
"""

        if health >= 70:
            report += "EXCELLENT performance! Scale winning campaigns.\n"
        elif health >= 50:
            report += "GOOD performance. Optimize for improvement.\n"
        else:
            report += "IMMEDIATE optimization required.\n"

        report += f"""

ACTION ITEMS
--------------------------------------------------------------------------------
- Pause {len(classification['wastage'])} wastage keywords
- Scale {len(classification['high_potential'])} high potential keywords
- Review match type performance
- Implement bid adjustments
- Follow-up review in 7 days

================================================================================
Report Generated By: Amazon Ads Agency Dashboard Pro v2.0
================================================================================
"""

        return report

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    if 'clients' not in st.session_state:
        st.session_state.clients = {}
    if 'active_client' not in st.session_state:
        st.session_state.active_client = None
    if 'agency_name' not in st.session_state:
        st.session_state.agency_name = "Your Agency Name"

# ═══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def render_agency_header():
    st.markdown(f"""
    <div class="agency-header">
        <h1>🏢 {st.session_state.agency_name}</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">Amazon Ads Agency Dashboard Pro v2.0</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=False):
            new_name = st.text_input("Agency Name", value=st.session_state.agency_name)
            if new_name != st.session_state.agency_name:
                st.session_state.agency_name = new_name
                st.rerun()

        st.markdown("---")
        st.markdown("### 👥 Clients")

        if st.session_state.clients:
            selected = st.selectbox("Active Client", list(st.session_state.clients.keys()))
            st.session_state.active_client = selected

            if selected:
                client = st.session_state.clients[selected]
                if client.analyzer:
                    health = client.analyzer.get_health_score()
                    emoji = "🟢" if health >= 70 else "🟡" if health >= 50 else "🔴"
                    st.info(f"{emoji} Health: {health}/100")

        st.markdown("---")

        with st.expander("➕ Add Client", expanded=False):
            name = st.text_input("Client Name*")
            industry = st.selectbox("Industry", ["E-commerce", "Electronics", "Fashion", "Beauty", "Other"])
            budget = st.number_input("Monthly Budget (₹)", value=50000, step=5000)

            col1, col2 = st.columns(2)
            with col1:
                target_acos = st.number_input("Target ACOS (%)", value=30.0, step=5.0)
            with col2:
                target_roas = st.number_input("Target ROAS (x)", value=3.0, step=0.5)

            fee = st.number_input("Your Fee (₹)", value=10000, step=1000)
            email = st.text_input("Email")
            file = st.file_uploader("Upload Report*", type=["xlsx", "xls"])

            if st.button("✅ Add", type="primary", use_container_width=True):
                if name and file:
                    try:
                        with st.spinner(f"Analyzing {name}'s data..."):
                            df = pd.read_excel(file)
                            client = ClientData(name, industry, budget)
                            client.monthly_fee = fee
                            client.contact_email = email
                            client.target_acos = target_acos
                            client.target_roas = target_roas
                            client.analyzer = CompleteAnalyzer(df, name, target_acos, target_roas)
                            st.session_state.clients[name] = client
                            st.session_state.active_client = name
                            st.success(f"✅ Added {name}!")
                            st.balloons()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please provide name and file")

def render_dashboard_tab(client, analyzer):
    st.header(f"📊 {client.name} - Dashboard")

    summary = analyzer.get_client_summary()
    health = analyzer.get_health_score()

    st.markdown(f"""
    <div class="info-box">
        <h2 style="margin:0;">Health Score: {health}/100</h2>
        <p style="margin:0.5rem 0 0 0;">Target ACOS: {client.target_acos:.1f}% | Target ROAS: {client.target_roas:.1f}x</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Spend", f"₹{summary['total_spend']:,.0f}")
    with col2:
        st.metric("Sales", f"₹{summary['total_sales']:,.0f}")
    with col3:
        st.metric("ROAS", f"{summary['roas']:.2f}x")
    with col4:
        st.metric("Orders", f"{summary['total_orders']:,}")
    with col5:
        st.metric("CVR", f"{summary['avg_cvr']:.2f}%")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CPC", f"₹{summary['avg_cpc']:.2f}")
    with col2:
        st.metric("ACOS", f"{summary['acos']:.1f}%")
    with col3:
        st.metric("Clicks", f"{summary['total_clicks']:,}")
    with col4:
        st.metric("Wastage", f"₹{summary['total_wastage']:,.0f}")

def render_keywords_tab(analyzer):
    st.header("🎯 Keywords Analysis")

    classification = analyzer.classify_keywords_improved()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 High Potential", len(classification['high_potential']))
    with col2:
        st.metric("⚡ Opportunities", len(classification['opportunities']))
    with col3:
        st.metric("⚠️ Low Potential", len(classification['low_potential']))
    with col4:
        st.metric("🚨 Wastage", len(classification['wastage']))

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["High Potential", "Opportunities", "Low Potential", "Wastage"])

    with tab1:
        if classification['high_potential']:
            st.success("Scale these keywords!")
            st.dataframe(pd.DataFrame(classification['high_potential']), use_container_width=True, hide_index=True)
        else:
            st.info("No high potential keywords yet")

    with tab2:
        if classification['opportunities']:
            st.info("Test bid increases on these")
            st.dataframe(pd.DataFrame(classification['opportunities']), use_container_width=True, hide_index=True)
        else:
            st.info("No opportunities identified")

    with tab3:
        if classification['low_potential']:
            st.warning("Reduce bids or pause")
            st.dataframe(pd.DataFrame(classification['low_potential']), use_container_width=True, hide_index=True)
        else:
            st.success("No low potential keywords")

    with tab4:
        if classification['wastage']:
            total_wasted = sum(float(k['Spend'].replace('₹','')) for k in classification['wastage'])
            st.error(f"URGENT: Rs{total_wasted:,.2f} wasted - Pause immediately!")
            st.dataframe(pd.DataFrame(classification['wastage']), use_container_width=True, hide_index=True)
        else:
            st.success("No wastage!")

def render_bid_optimization_tab(analyzer):
    st.header("💡 Bid Optimization")

    suggestions = analyzer.get_bid_suggestions_improved()

    if suggestions:
        st.success(f"Found {len(suggestions)} recommendations")

        col1, col2 = st.columns(2)
        with col1:
            action_filter = st.selectbox("Filter by Action", ["All", "INCREASE", "REDUCE", "PAUSE"])

        filtered = suggestions if action_filter == "All" else [s for s in suggestions if action_filter in s['Action']]

        st.markdown(f"**Showing {len(filtered)} of {len(suggestions)} suggestions**")
        st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)
    else:
        st.info("No suggestions at this time")

def render_match_type_tab(analyzer):
    st.header("📊 Match Type Analysis")

    match_perf = analyzer.get_match_type_performance()

    if not match_perf.empty:
        st.subheader("Performance by Match Type")
        st.dataframe(match_perf.style.format({
            'Spend': '₹{:,.2f}',
            'Sales': '₹{:,.2f}',
            'ROAS': '{:.2f}x',
            'ACOS': '{:.1f}%',
            'CVR': '{:.2f}%',
            'CTR': '{:.2f}%'
        }), use_container_width=True)

        best_roas = match_perf['ROAS'].idxmax()
        st.success(f"Best performing: {best_roas} with {match_perf.loc[best_roas, 'ROAS']:.2f}x ROAS")
    else:
        st.warning("Match type data not available")

def render_exports_tab(analyzer, client_name):
    st.header("📥 Export Files")

    classification = analyzer.classify_keywords_improved()
    suggestions = analyzer.get_bid_suggestions_improved()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚫 Negative Keywords")
        wastage = classification['wastage']

        if wastage:
            neg_data = [{
                'Campaign': kw['Campaign'],
                'Ad Group': '',
                'Keyword': kw['Keyword'],
                'Match Type': 'Negative Exact',
                'Status': 'Enabled'
            } for kw in wastage]

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(neg_data).to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                f"Download ({len(neg_data)} keywords)",
                data=output,
                file_name=f"Negatives_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.success("No negative keywords needed")

    with col2:
        st.subheader("💰 Bid Adjustments")

        if suggestions:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(suggestions).to_excel(writer, index=False)
            output.seek(0)

            st.download_button(
                f"Download ({len(suggestions)} bids)",
                data=output,
                file_name=f"Bids_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.info("No bid adjustments")

def render_report_tab(client, analyzer):
    st.header("📝 Client Report")

    report = analyzer.generate_client_report()
    st.text_area("Report", report, height=600)

    st.download_button(
        "📄 Download Report (TXT)",
        data=report,
        file_name=f"Report_{client.name}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )

def render_all_clients_tab():
    st.header("👥 All Clients")

    if not st.session_state.clients:
        st.info("No clients added yet")
        return

    data = []
    for name, client in st.session_state.clients.items():
        if client.analyzer:
            summary = client.analyzer.get_client_summary()
            health = client.analyzer.get_health_score()
            data.append({
                'Client': name,
                'Health': f"{health}/100",
                'Spend': f"₹{summary['total_spend']:,.0f}",
                'Sales': f"₹{summary['total_sales']:,.0f}",
                'ROAS': f"{summary['roas']:.2f}x",
                'Fee': f"₹{client.monthly_fee:,.0f}"
            })

    if data:
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        total_revenue = sum(c.monthly_fee for c in st.session_state.clients.values())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clients", len(st.session_state.clients))
        with col2:
            st.metric("Monthly Revenue", f"₹{total_revenue:,.0f}")
        with col3:
            st.metric("Annual Revenue", f"₹{total_revenue * 12:,.0f}")

def render_dashboard():
    render_agency_header()

    if not st.session_state.clients:
        st.info("👈 Add your first client from the sidebar")
        return

    if not st.session_state.active_client:
        st.warning("⚠️ Please select a client")
        return

    client = st.session_state.clients[st.session_state.active_client]

    if not client.analyzer:
        st.error("❌ No data loaded")
        return

    analyzer = client.analyzer

    tabs = st.tabs([
        "📊 Dashboard",
        "🎯 Keywords",
        "💡 Bids",
        "📊 Match Types",
        "📝 Report",
        "👥 All Clients",
        "📥 Exports"
    ])

    with tabs[0]:
        render_dashboard_tab(client, analyzer)
    with tabs[1]:
        render_keywords_tab(analyzer)
    with tabs[2]:
        render_bid_optimization_tab(analyzer)
    with tabs[3]:
        render_match_type_tab(analyzer)
    with tabs[4]:
        render_report_tab(client, analyzer)
    with tabs[5]:
        render_all_clients_tab()
    with tabs[6]:
        render_exports_tab(analyzer, client.name)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_custom_css()
    init_session_state()
    render_sidebar()
    render_dashboard()

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 1rem;">
        <strong>{st.session_state.agency_name}</strong><br>
        Amazon Ads Agency Dashboard Pro v2.0<br>
        <small>FIXED - Ready for Production</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
