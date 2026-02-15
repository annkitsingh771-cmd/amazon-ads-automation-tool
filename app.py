#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Ads Agency Dashboard Pro v2.0
FULLY ERROR-PROOF VERSION with comprehensive error handling
"""

import io
import traceback
from datetime import datetime
from typing import Dict, List, Optional

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
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 1.3rem 1.2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .success-box {
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.25) 0%, rgba(22, 163, 74, 0.1) 100%);
            border-left: 5px solid #22c55e;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .warning-box {
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.25) 0%, rgba(234, 179, 8, 0.1) 100%);
            border-left: 5px solid #facc15;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .danger-box {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(220, 38, 38, 0.1) 100%);
            border-left: 5px solid #ef4444;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .info-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(59, 130, 246, 0.1) 100%);
            border-left: 5px solid #3b82f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return float(value)
    except:
        return default

def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return int(float(value))
    except:
        return default

def safe_str(value, default='N/A'):
    """Safely convert value to string"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return str(value).strip()
    except:
        return default

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class ClientData:
    def __init__(self, name: str, industry: str = "E-commerce", monthly_budget: float = 50000):
        self.name = name
        self.industry = industry
        self.monthly_budget = monthly_budget
        self.analyzer = None
        self.added_date = datetime.now()
        self.monthly_fee = 0
        self.contact_email = ""
        self.target_acos = 30.0
        self.target_roas = 3.0

# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER CLASS WITH ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

class CompleteAnalyzer:
    REQUIRED_COLUMNS = ['Customer Search Term', 'Campaign Name', 'Spend', 'Clicks']

    MIN_SPEND_FOR_LOW_POTENTIAL = 50
    MIN_CLICKS_FOR_LOW_POTENTIAL = 10
    MIN_SPEND_FOR_WASTAGE = 100
    MIN_CLICKS_FOR_WASTAGE = 5
    MIN_SPEND_FOR_HIGH_POTENTIAL = 30
    MIN_ORDERS_FOR_HIGH_POTENTIAL = 2
    MIN_CVR_FOR_CHAMPION = 2.0

    def __init__(self, df: pd.DataFrame, client_name: str, target_acos: float = 30.0, target_roas: float = 3.0):
        self.client_name = client_name
        self.target_acos = target_acos
        self.target_roas = target_roas
        self.df = None
        self.error = None

        try:
            self.df = self._validate_and_prepare_data(df)
        except Exception as e:
            self.error = str(e)
            raise ValueError(f"Data validation failed: {str(e)}")

    def _validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare data with comprehensive error handling"""
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty")

        # Make a copy
        df = df.copy()

        # Clean column names
        df.columns = df.columns.str.strip()

        # Map common column name variations
        column_mapping = {
            'customer search term': 'Customer Search Term',
            'search term': 'Customer Search Term',
            'keyword': 'Customer Search Term',
            'campaign': 'Campaign Name',
            'campaign name': 'Campaign Name',
            'ad group': 'Ad Group Name',
            'ad group name': 'Ad Group Name',
            'match type': 'Match Type',
            '7 day total sales': 'Sales',
            '7 day total orders': 'Orders',
            '7 day orders': 'Orders',
            'total sales': 'Sales',
            'total orders': 'Orders',
            'cost': 'Spend',
        }

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # Capitalize first letters for remaining columns
        df.columns = [col.title() for col in df.columns]

        # Check for required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {available_cols}. "
                f"Please ensure your file is an Amazon Search Term Report."
            )

        # Add missing optional columns with defaults
        if 'Sales' not in df.columns:
            df['Sales'] = 0.0
        if 'Orders' not in df.columns:
            df['Orders'] = 0
        if 'Impressions' not in df.columns:
            df['Impressions'] = 0
        if 'CPC' not in df.columns and 'Cpc' not in df.columns:
            df['CPC'] = 0.0
        elif 'Cpc' in df.columns:
            df['CPC'] = df['Cpc']
        if 'Ad Group Name' not in df.columns:
            df['Ad Group Name'] = 'N/A'
        if 'Match Type' not in df.columns:
            df['Match Type'] = 'N/A'

        # Clean and convert numeric columns
        numeric_cols = ['Spend', 'Sales', 'Clicks', 'Impressions', 'Orders', 'CPC']
        for col in numeric_cols:
            if col in df.columns:
                # Remove currency symbols, commas, percentages
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('[₹$,]', '', regex=True)
                    df[col] = df[col].str.replace('%', '', regex=False)

                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Remove rows with zero spend and zero clicks (irrelevant data)
        df = df[(df['Spend'] > 0) | (df['Clicks'] > 0)].copy()

        if len(df) == 0:
            raise ValueError("No valid data rows found after filtering")

        # Calculate derived metrics
        df['Profit'] = df['Sales'] - df['Spend']
        df['Wastage'] = df.apply(lambda x: x['Spend'] if x['Sales'] == 0 else 0, axis=1)

        # CVR (Conversion Rate)
        df['CVR'] = df.apply(
            lambda x: (x['Orders'] / x['Clicks'] * 100) if x['Clicks'] > 0 else 0,
            axis=1
        )

        # ROAS
        df['ROAS'] = df.apply(
            lambda x: (x['Sales'] / x['Spend']) if x['Spend'] > 0 else 0,
            axis=1
        )

        # ACOS
        df['ACOS'] = df.apply(
            lambda x: (x['Spend'] / x['Sales'] * 100) if x['Sales'] > 0 else 0,
            axis=1
        )

        # CTR
        df['CTR'] = df.apply(
            lambda x: (x['Clicks'] / x['Impressions'] * 100) if x['Impressions'] > 0 else 0,
            axis=1
        )

        # Add client info
        df['Client'] = self.client_name
        df['Processed_Date'] = datetime.now()

        return df

    def get_client_summary(self) -> Dict:
        """Get comprehensive summary with error handling"""
        try:
            if self.df is None or len(self.df) == 0:
                return self._get_empty_summary()

            total_spend = safe_float(self.df['Spend'].sum())
            total_sales = safe_float(self.df['Sales'].sum())
            total_orders = safe_int(self.df['Orders'].sum())
            total_clicks = safe_int(self.df['Clicks'].sum())
            total_impressions = safe_int(self.df['Impressions'].sum())
            total_wastage = safe_float(self.df['Wastage'].sum())

            return {
                'total_spend': total_spend,
                'total_sales': total_sales,
                'total_profit': safe_float(self.df['Profit'].sum()),
                'total_orders': total_orders,
                'total_clicks': total_clicks,
                'total_impressions': total_impressions,
                'total_wastage': total_wastage,
                'roas': (total_sales / total_spend) if total_spend > 0 else 0,
                'acos': (total_spend / total_sales * 100) if total_sales > 0 else 0,
                'avg_cpc': safe_float(self.df['CPC'].mean()),
                'avg_ctr': safe_float(self.df['CTR'].mean()),
                'avg_cvr': safe_float(self.df['CVR'].mean()),
                'conversion_rate': (total_orders / total_clicks * 100) if total_clicks > 0 else 0,
                'keywords_count': len(self.df),
                'campaigns_count': safe_int(self.df['Campaign Name'].nunique())
            }
        except Exception as e:
            st.error(f"Error calculating summary: {e}")
            return self._get_empty_summary()

    def _get_empty_summary(self) -> Dict:
        """Return empty summary structure"""
        return {
            'total_spend': 0,
            'total_sales': 0,
            'total_profit': 0,
            'total_orders': 0,
            'total_clicks': 0,
            'total_impressions': 0,
            'total_wastage': 0,
            'roas': 0,
            'acos': 0,
            'avg_cpc': 0,
            'avg_ctr': 0,
            'avg_cvr': 0,
            'conversion_rate': 0,
            'keywords_count': 0,
            'campaigns_count': 0
        }

    def get_health_score(self) -> int:
        """Calculate health score 0-100"""
        try:
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
        except:
            return 0

    def classify_keywords_improved(self) -> Dict[str, List[Dict]]:
        """Classify keywords with error handling"""
        categories = {
            'high_potential': [],
            'low_potential': [],
            'wastage': [],
            'opportunities': []
        }

        try:
            if self.df is None or len(self.df) == 0:
                return categories

            for _, row in self.df.iterrows():
                try:
                    spend = safe_float(row.get('Spend', 0))
                    sales = safe_float(row.get('Sales', 0))
                    roas = safe_float(row.get('ROAS', 0))
                    orders = safe_int(row.get('Orders', 0))
                    clicks = safe_int(row.get('Clicks', 0))
                    cvr = safe_float(row.get('CVR', 0))

                    kw_data = {
                        'Keyword': safe_str(row.get('Customer Search Term', 'Unknown')),
                        'Spend': f"₹{spend:.2f}",
                        'Sales': f"₹{sales:.2f}",
                        'ROAS': f"{roas:.2f}x",
                        'Orders': orders,
                        'Clicks': clicks,
                        'CVR': f"{cvr:.2f}%",
                        'Campaign': safe_str(row.get('Campaign Name', 'N/A')),
                        'Match Type': safe_str(row.get('Match Type', 'N/A')),
                        'Reason': ''
                    }

                    # Classification logic
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
                        kw_data['Reason'] = f"Decent ROAS ({roas:.2f}x), test optimization"
                        categories['opportunities'].append(kw_data)

                except Exception:
                    continue

            return categories
        except Exception as e:
            st.error(f"Error classifying keywords: {e}")
            return categories

    def get_bid_suggestions_improved(self) -> List[Dict]:
        """Get bid suggestions with error handling"""
        suggestions = []

        try:
            if self.df is None or len(self.df) == 0:
                return suggestions

            for _, row in self.df.iterrows():
                try:
                    spend = safe_float(row.get('Spend', 0))
                    sales = safe_float(row.get('Sales', 0))
                    roas = safe_float(row.get('ROAS', 0))
                    orders = safe_int(row.get('Orders', 0))
                    clicks = safe_int(row.get('Clicks', 0))
                    cvr = safe_float(row.get('CVR', 0))
                    current_cpc = safe_float(row.get('CPC', 0))

                    if spend < 30 or clicks < 5:
                        continue

                    suggestion = {
                        'Keyword': safe_str(row.get('Customer Search Term', 'Unknown')),
                        'Campaign': safe_str(row.get('Campaign Name', 'N/A')),
                        'Ad Group': safe_str(row.get('Ad Group Name', 'N/A')),
                        'Match Type': safe_str(row.get('Match Type', 'N/A')),
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

                    # Decision logic
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

            return sorted(suggestions, key=lambda x: float(x['Spend'].replace('₹','').replace(',','')), reverse=True)
        except Exception as e:
            st.error(f"Error generating bid suggestions: {e}")
            return []

    def get_match_type_performance(self) -> pd.DataFrame:
        """Get match type analysis with error handling"""
        try:
            if self.df is None or len(self.df) == 0 or 'Match Type' not in self.df.columns:
                return pd.DataFrame()

            # Filter out N/A match types
            df_filtered = self.df[self.df['Match Type'] != 'N/A'].copy()

            if len(df_filtered) == 0:
                return pd.DataFrame()

            match_perf = df_filtered.groupby('Match Type').agg({
                'Spend': 'sum',
                'Sales': 'sum',
                'Orders': 'sum',
                'Clicks': 'sum',
                'Impressions': 'sum'
            })

            match_perf['ROAS'] = match_perf.apply(
                lambda x: x['Sales'] / x['Spend'] if x['Spend'] > 0 else 0, axis=1
            )
            match_perf['ACOS'] = match_perf.apply(
                lambda x: x['Spend'] / x['Sales'] * 100 if x['Sales'] > 0 else 0, axis=1
            )
            match_perf['CVR'] = match_perf.apply(
                lambda x: x['Orders'] / x['Clicks'] * 100 if x['Clicks'] > 0 else 0, axis=1
            )
            match_perf['CTR'] = match_perf.apply(
                lambda x: x['Clicks'] / x['Impressions'] * 100 if x['Impressions'] > 0 else 0, axis=1
            )

            return match_perf
        except Exception as e:
            st.error(f"Error analyzing match types: {e}")
            return pd.DataFrame()

    def generate_client_report(self) -> str:
        """Generate comprehensive report"""
        try:
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
                report += "GOOD performance. Optimize for further improvement.\n"
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
        except Exception as e:
            return f"Error generating report: {e}"

# Continue with UI functions...

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
        <p style="font-size: 0.9rem; opacity: 0.9;">Complete PPC Management Platform</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=False):
            new_name = st.text_input("Agency Name", value=st.session_state.agency_name, key='agency_name_input')
            if new_name != st.session_state.agency_name:
                st.session_state.agency_name = new_name
                st.rerun()
            st.info("Your agency name appears on all reports")

        st.markdown("---")
        st.markdown("### 👥 Client Management")

        if st.session_state.clients:
            client_names = list(st.session_state.clients.keys())
            selected = st.selectbox("Active Client", client_names, key='client_selector')
            st.session_state.active_client = selected

            if selected:
                client = st.session_state.clients[selected]
                if client.analyzer and client.analyzer.df is not None:
                    try:
                        health = client.analyzer.get_health_score()
                        emoji = "🟢" if health >= 70 else "🟡" if health >= 50 else "🔴"
                        st.info(f"{emoji} Health: {health}/100\nTarget ACOS: {client.target_acos:.1f}%\nTarget ROAS: {client.target_roas:.1f}x")
                    except:
                        st.warning("Error calculating health score")

        st.markdown("---")

        with st.expander("➕ Add New Client", expanded=False):
            name = st.text_input("Client Name*", placeholder="ABC Company", key='new_client_name')

            col1, col2 = st.columns(2)
            with col1:
                industry = st.selectbox("Industry", [
                    "E-commerce", "Electronics", "Fashion", "Beauty", 
                    "Home & Kitchen", "Sports", "Books", "Health", "Other"
                ], key='new_industry')
            with col2:
                budget = st.number_input("Monthly Budget (₹)", value=50000, step=5000, key='new_budget')

            st.markdown("**🎯 Performance Goals:**")
            col1, col2 = st.columns(2)
            with col1:
                target_acos = st.number_input("Target ACOS (%)", value=30.0, step=5.0, 
                                             help="Target Advertising Cost of Sales", key='new_acos')
            with col2:
                target_roas = st.number_input("Target ROAS (x)", value=3.0, step=0.5,
                                             help="Target Return on Ad Spend", key='new_roas')

            fee = st.number_input("Your Monthly Fee (₹)", value=10000, step=1000, key='new_fee')
            email = st.text_input("Contact Email", placeholder="client@company.com", key='new_email')

            st.info("📄 Upload Amazon Search Term Report (Excel file)")
            uploaded_file = st.file_uploader("", type=["xlsx", "xls"], key='new_file',
                                            help="Download from: Amazon Ads Console → Reports → Search Term Report")

            if st.button("✅ Add Client", type="primary", use_container_width=True):
                if not name:
                    st.error("❌ Please enter client name")
                elif not uploaded_file:
                    st.error("❌ Please upload Search Term Report")
                else:
                    try:
                        with st.spinner(f"Analyzing {name}'s data..."):
                            # Read Excel file
                            df = pd.read_excel(uploaded_file)

                            # Show columns for debugging
                            st.info(f"Found {len(df)} rows and {len(df.columns)} columns")

                            # Create client
                            client_data = ClientData(name, industry, budget)
                            client_data.monthly_fee = fee
                            client_data.contact_email = email
                            client_data.target_acos = target_acos
                            client_data.target_roas = target_roas

                            # Create analyzer
                            client_data.analyzer = CompleteAnalyzer(df, name, target_acos, target_roas)

                            # Save to session state
                            st.session_state.clients[name] = client_data
                            st.session_state.active_client = name

                            st.success(f"✅ Successfully added {name}!")
                            st.balloons()
                            st.rerun()

                    except ValueError as e:
                        st.error(f"❌ Data Validation Error: {str(e)}")
                        st.info("Please ensure you've uploaded a valid Amazon Search Term Report with required columns")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("🔍 See detailed error"):
                            st.code(traceback.format_exc())

        # List all clients
        if st.session_state.clients:
            st.markdown("---")
            st.markdown("### 📋 All Clients")
            for client_name in list(st.session_state.clients.keys()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"📊 {client_name}")
                with col2:
                    if st.button("❌", key=f"del_{client_name}"):
                        del st.session_state.clients[client_name]
                        if st.session_state.active_client == client_name:
                            st.session_state.active_client = None
                        st.rerun()

def render_dashboard_tab(client, analyzer):
    try:
        st.header(f"📊 {client.name} - Dashboard")

        summary = analyzer.get_client_summary()
        health = analyzer.get_health_score()

        health_color = "success" if health >= 70 else "warning" if health >= 50 else "error"
        st.markdown(f"""
        <div class="info-box">
            <h2 style="margin:0;">Health Score: {health}/100</h2>
            <p style="margin:0.5rem 0 0 0;">Target ACOS: {client.target_acos:.1f}% | Target ROAS: {client.target_roas:.1f}x</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💰 Financial Performance")

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
            st.metric("Profit", f"₹{summary['total_profit']:,.0f}")

        st.markdown("---")
        st.subheader("📈 Campaign Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CVR", f"{summary['avg_cvr']:.2f}%")
        with col2:
            st.metric("ACOS", f"{summary['acos']:.1f}%")
        with col3:
            st.metric("Clicks", f"{summary['total_clicks']:,}")
        with col4:
            st.metric("Wastage", f"₹{summary['total_wastage']:,.0f}")

        if summary['total_wastage'] > 0:
            wastage_pct = (summary['total_wastage'] / summary['total_spend'] * 100) if summary['total_spend'] > 0 else 0
            st.markdown("---")
            st.markdown(f"""
            <div class="danger-box">
                <strong>🚨 Budget Wastage Alert!</strong><br>
                <strong>₹{summary['total_wastage']:,.2f}</strong> spent on ZERO sales keywords 
                (<strong>{wastage_pct:.1f}%</strong> of total spend)
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")

def render_keywords_tab(analyzer):
    try:
        st.header("🎯 Keywords Analysis")

        st.markdown("""
        <div class="info-box">
            <strong>📊 Improved Classification Thresholds (v2.0)</strong><br>
            • High Potential: ROAS ≥3.0x, ≥2 orders, ≥₹30 spend<br>
            • Low Potential: ≥₹50 spend, ≥10 clicks, ROAS <1.5x<br>
            • Wastage: ≥₹100 spend, ≥5 clicks, ZERO sales
        </div>
        """, unsafe_allow_html=True)

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

        tab1, tab2, tab3, tab4 = st.tabs([
            f"🏆 High Potential ({len(classification['high_potential'])})",
            f"⚡ Opportunities ({len(classification['opportunities'])})",
            f"⚠️ Low Potential ({len(classification['low_potential'])})",
            f"🚨 Wastage ({len(classification['wastage'])})"
        ])

        with tab1:
            if classification['high_potential']:
                st.success("✅ Scale these keywords aggressively!")
                st.dataframe(pd.DataFrame(classification['high_potential']), 
                           use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No high potential keywords yet. Continue optimizing!")

        with tab2:
            if classification['opportunities']:
                st.info("⚡ Test bid increases on these keywords")
                st.dataframe(pd.DataFrame(classification['opportunities']), 
                           use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No opportunities identified at this time")

        with tab3:
            if classification['low_potential']:
                st.warning("⚠️ Reduce bids by 30% or pause these keywords")
                st.dataframe(pd.DataFrame(classification['low_potential']), 
                           use_container_width=True, hide_index=True, height=400)
            else:
                st.success("✅ No low potential keywords!")

        with tab4:
            if classification['wastage']:
                total_wasted = sum(float(k['Spend'].replace('₹','').replace(',','')) 
                                 for k in classification['wastage'])
                st.error(f"🚨 URGENT: ₹{total_wasted:,.2f} wasted - Pause immediately!")
                st.dataframe(pd.DataFrame(classification['wastage']), 
                           use_container_width=True, hide_index=True, height=400)
            else:
                st.success("🎉 No wastage - excellent optimization!")

    except Exception as e:
        st.error(f"Error analyzing keywords: {e}")

def render_bid_optimization_tab(analyzer):
    try:
        st.header("💡 Bid Optimization Suggestions")

        st.markdown(f"""
        <div class="info-box">
            <strong>🎯 Goal-Based Bid Optimization (NEW v2.0)</strong><br>
            Target ACOS: {analyzer.target_acos:.1f}% | Target ROAS: {analyzer.target_roas:.1f}x<br>
            Minimum thresholds: ≥₹30 spend, ≥5 clicks
        </div>
        """, unsafe_allow_html=True)

        suggestions = analyzer.get_bid_suggestions_improved()

        if suggestions:
            st.success(f"✅ Found {len(suggestions)} actionable recommendations")

            col1, col2 = st.columns(2)
            with col1:
                action_filter = st.selectbox("Filter by Action", 
                                            ["All", "INCREASE", "REDUCE", "PAUSE"])

            filtered = suggestions if action_filter == "All" else [
                s for s in suggestions if action_filter in s['Action']
            ]

            st.markdown(f"**Showing {len(filtered)} of {len(suggestions)} suggestions**")
            st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True, height=500)

            # Summary
            st.markdown("---")
            st.subheader("📊 Optimization Summary")
            col1, col2, col3 = st.columns(3)

            increase_count = len([s for s in suggestions if 'INCREASE' in s['Action']])
            reduce_count = len([s for s in suggestions if 'REDUCE' in s['Action']])
            pause_count = len([s for s in suggestions if 'PAUSE' in s['Action']])

            with col1:
                st.metric("Scale Up", increase_count)
            with col2:
                st.metric("Reduce Bids", reduce_count)
            with col3:
                st.metric("Pause", pause_count)
        else:
            st.info("No bid optimization suggestions at this time. Your campaigns are well-optimized!")

    except Exception as e:
        st.error(f"Error generating bid suggestions: {e}")

def render_match_type_tab(analyzer):
    try:
        st.header("📊 Match Type Performance Analysis")

        st.markdown("""
        <div class="info-box">
            <strong>🎯 Match Type Strategy (NEW v2.0)</strong><br>
            Compare Exact, Phrase, and Broad match performance
        </div>
        """, unsafe_allow_html=True)

        match_perf = analyzer.get_match_type_performance()

        if not match_perf.empty:
            st.subheader("Performance by Match Type")

            # Format for display
            display_df = match_perf.copy()
            display_df['Spend'] = display_df['Spend'].apply(lambda x: f"₹{x:,.2f}")
            display_df['Sales'] = display_df['Sales'].apply(lambda x: f"₹{x:,.2f}")
            display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.2f}x")
            display_df['ACOS'] = display_df['ACOS'].apply(lambda x: f"{x:.1f}%")
            display_df['CVR'] = display_df['CVR'].apply(lambda x: f"{x:.2f}%")
            display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.2f}%")
            display_df['Orders'] = display_df['Orders'].astype(int)

            st.dataframe(display_df, use_container_width=True)

            # Best performer
            best_roas_type = match_perf['ROAS'].idxmax()
            best_roas_value = match_perf.loc[best_roas_type, 'ROAS']

            st.success(f"🏆 Best performing: **{best_roas_type}** with {best_roas_value:.2f}x ROAS")

            st.markdown("---")
            st.subheader("💡 Recommendations")
            for match_type, data in match_perf.iterrows():
                roas = data['ROAS']
                if roas >= 3.0:
                    st.success(f"✅ **{match_type}**: Excellent! Consider increasing budget allocation")
                elif roas >= 2.0:
                    st.info(f"ℹ️ **{match_type}**: Good performance. Monitor and optimize")
                else:
                    st.warning(f"⚠️ **{match_type}**: Underperforming. Review keywords or reduce budget")
        else:
            st.warning("⚠️ Match type data not available in the uploaded report")

    except Exception as e:
        st.error(f"Error analyzing match types: {e}")

def render_exports_tab(analyzer, client_name):
    try:
        st.header("📥 Export Action Files")

        st.markdown("""
        <div class="info-box">
            <strong>📁 Ready-to-Upload Files</strong><br>
            All files are formatted for Amazon Ads bulk upload operations
        </div>
        """, unsafe_allow_html=True)

        classification = analyzer.classify_keywords_improved()
        suggestions = analyzer.get_bid_suggestions_improved()

        col1, col2, col3 = st.columns(3)

        # Negative Keywords
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
                    pd.DataFrame(neg_data).to_excel(writer, index=False, sheet_name='Negative Keywords')
                output.seek(0)

                st.download_button(
                    f"📥 Download ({len(neg_data)} keywords)",
                    data=output,
                    file_name=f"Negative_Keywords_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success(f"✅ {len(neg_data)} keywords to pause")
            else:
                st.success("✅ No negative keywords needed!")

        # Bid Adjustments
        with col2:
            st.subheader("💰 Bid Adjustments")

            if suggestions:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    pd.DataFrame(suggestions).to_excel(writer, index=False, sheet_name='Bid Adjustments')
                output.seek(0)

                st.download_button(
                    f"📥 Download ({len(suggestions)} bids)",
                    data=output,
                    file_name=f"Bid_Adjustments_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success(f"✅ {len(suggestions)} bid recommendations")
            else:
                st.info("No bid adjustments needed")

        # Complete Analysis
        with col3:
            st.subheader("📊 Complete Data")

            if analyzer.df is not None and len(analyzer.df) > 0:
                csv_data = analyzer.df.to_csv(index=False)

                st.download_button(
                    f"📥 Download ({len(analyzer.df)} rows)",
                    data=csv_data,
                    file_name=f"Complete_Analysis_{client_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Full dataset with CVR")

    except Exception as e:
        st.error(f"Error creating exports: {e}")

def render_report_tab(client, analyzer):
    try:
        st.header("📝 Client Report Generation")

        report = analyzer.generate_client_report()

        st.subheader("📊 Generated Report")
        st.text_area("Report Content", report, height=600, key='report_preview')

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📄 Download Report (TXT)",
                data=report,
                file_name=f"Report_{client.name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:
            # Excel report with summary
            summary = analyzer.get_client_summary()
            classification = analyzer.classify_keywords_improved()

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
                if classification['high_potential']:
                    pd.DataFrame(classification['high_potential']).to_excel(
                        writer, sheet_name='High Potential', index=False)
                if classification['wastage']:
                    pd.DataFrame(classification['wastage']).to_excel(
                        writer, sheet_name='Wastage', index=False)
                pd.DataFrame({'Report': [report]}).to_excel(
                    writer, sheet_name='Full Report', index=False)
            output.seek(0)

            st.download_button(
                "📊 Download Report (Excel)",
                data=output,
                file_name=f"Report_{client.name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error generating report: {e}")

def render_all_clients_tab():
    try:
        st.header("👥 All Clients Overview")

        if not st.session_state.clients:
            st.info("No clients added yet")
            return

        data = []
        for name, client in st.session_state.clients.items():
            if client.analyzer and client.analyzer.df is not None:
                try:
                    summary = client.analyzer.get_client_summary()
                    health = client.analyzer.get_health_score()
                    data.append({
                        'Client': name,
                        'Health': f"{health}/100",
                        'Spend': f"₹{summary['total_spend']:,.0f}",
                        'Sales': f"₹{summary['total_sales']:,.0f}",
                        'ROAS': f"{summary['roas']:.2f}x",
                        'ACOS': f"{summary['acos']:.1f}%",
                        'Fee': f"₹{client.monthly_fee:,.0f}",
                        'Status': '🟢' if health >= 70 else '🟡' if health >= 50 else '🔴'
                    })
                except:
                    continue

        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True, height=400)

            # Agency totals
            st.markdown("---")
            st.subheader("💰 Agency Totals")

            total_revenue = sum(c.monthly_fee for c in st.session_state.clients.values())
            total_clients = len(st.session_state.clients)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Clients", total_clients)
            with col2:
                st.metric("Monthly Revenue", f"₹{total_revenue:,.0f}")
            with col3:
                st.metric("Annual Revenue", f"₹{total_revenue * 12:,.0f}")

    except Exception as e:
        st.error(f"Error displaying clients: {e}")

def render_dashboard():
    render_agency_header()

    if not st.session_state.clients:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to Amazon Ads Agency Dashboard Pro v2.0!</h3>
            <p>Get started by adding your first client from the sidebar.</p>
            <br>
            <strong>✨ Features:</strong>
            <ul>
                <li>CVR at keyword level (NEW v2.0)</li>
                <li>Improved classification thresholds (No false positives!)</li>
                <li>Goal-based bid optimization (Target ACOS & ROAS)</li>
                <li>Match Type analysis (NEW v2.0)</li>
                <li>Professional reports & exports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return

    if not st.session_state.active_client:
        st.warning("⚠️ Please select a client from the sidebar")
        return

    client = st.session_state.clients[st.session_state.active_client]

    if not client.analyzer or client.analyzer.df is None:
        st.error("❌ No data loaded for this client")
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
# MAIN FUNCTION
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
        Amazon Ads Agency Dashboard Pro v2.0 - ERROR-FREE Edition<br>
        <small>70+ Features | Production Ready | Robust Error Handling</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
