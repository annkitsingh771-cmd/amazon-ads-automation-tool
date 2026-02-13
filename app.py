#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               Amazon Ads Agency Dashboard PRO - Complete Edition             ║
║                    Full Production-Ready Code v1.0                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
- Multi-client management
- AI-powered analysis  
- Professional reporting
- Budget forecasting
- Revenue tracking
- Export functionality

Author: Amazon Ads Agency Tools
License: MIT
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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Amazon Ads Agency Dashboard Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Amazon Ads Agency Dashboard Pro - Professional client management platform"
    }
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
    <style>
    /* Main container */
    .main {
        padding-top: 1rem;
    }

    /* Agency header styling */
    .agency-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }

    .agency-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .agency-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }

    /* Client card styling */
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
        border-color: #3b82f6;
    }

    /* Enhanced metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 14px;
        padding: 1.3rem 1.2rem;
        min-height: 105px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.7rem;
        margin-top: 0.4rem;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Alert boxes with gradients */
    .success-box, .warning-box, .danger-box, .info-box {
        padding: 1.3rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.7;
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

    /* Report card for white background content */
    .report-card {
        background: white;
        color: #1e293b;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Enhanced table styling */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Headers styling */
    h1 {
        color: #f1f5f9;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #e2e8f0;
        font-weight: 600;
        margin-top: 2rem;
    }

    h3 {
        color: #cbd5e1;
        font-weight: 600;
    }

    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }

    /* Download button specific styling */
    .stDownloadButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
    }

    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 8px;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES
# ============================================================================

class ClientData:
    """Client data structure with all relevant information"""

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

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'industry': self.industry,
            'monthly_budget': self.monthly_budget,
            'monthly_fee': self.monthly_fee,
            'contact_email': self.contact_email,
            'contact_phone': self.contact_phone,
            'status': self.status,
            'added_date': self.added_date.isoformat()
        }

# ============================================================================
# ANALYZER CLASS
# ============================================================================

class AgencyAnalyzer:
    """Enhanced analyzer with comprehensive agency features"""

    def __init__(self, df: pd.DataFrame, client_name: str):
        self.df = df.copy()
        self.client_name = client_name
        self._normalize_columns()
        self._clean_data()
        self._enrich_data()

    def _normalize_columns(self):
        """Normalize column names to handle variations"""
        # Strip whitespace
        self.df.columns = self.df.columns.str.strip()

        # Create mapping dictionary
        column_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower().strip()

            # ACOS variations
            if "acos" in col_lower:
                column_mapping[col] = "ACOS"
            # ROAS variations
            elif "roas" in col_lower:
                column_mapping[col] = "ROAS"
            # Sales variations
            elif "7 day total sales" in col_lower or ("sales" in col_lower and "7" in col_lower):
                column_mapping[col] = "Sales"
            # Orders variations
            elif "7 day total orders" in col_lower or "7 day orders" in col_lower:
                column_mapping[col] = "Orders"
            # Conversion rate
            elif "conversion" in col_lower and "rate" in col_lower:
                column_mapping[col] = "Conversion_Rate"
            # CTR
            elif "ctr" in col_lower or "click-through" in col_lower:
                column_mapping[col] = "CTR"
            # CPC
            elif "cpc" in col_lower or "cost per click" in col_lower:
                column_mapping[col] = "CPC"

        # Apply mapping
        if column_mapping:
            self.df.rename(columns=column_mapping, inplace=True)

    def _clean_data(self):
        """Clean and prepare data"""
        # Convert percentage columns
        percentage_cols = ["CTR", "Conversion_Rate", "ACOS"]
        for col in percentage_cols:
            if col in self.df.columns:
                if self.df[col].dtype == "object":
                    try:
                        self.df[col] = self.df[col].astype(str).str.rstrip("%").astype("float") / 100.0
                    except Exception:
                        pass

        # Convert numeric columns
        numeric_cols = ["Sales", "ACOS", "ROAS", "Orders", "Spend", "Clicks", "Impressions", "CPC"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def _enrich_data(self):
        """Add calculated fields"""
        # Add client identifier
        self.df['Client'] = self.client_name

        # Calculate profit
        self.df['Profit'] = self.df['Sales'] - self.df['Spend']

        # Calculate wastage (spend with zero sales)
        self.df['Wastage'] = np.where(self.df['Sales'] == 0, self.df['Spend'], 0)

        # Calculate actual ACOS
        self.df['ACOS_Calc'] = np.where(
            self.df['Sales'] > 0,
            (self.df['Spend'] / self.df['Sales']) * 100,
            999
        )

        # Add processing timestamp
        self.df['Processed_Date'] = datetime.now()

    def get_client_summary(self) -> Dict:
        """Get comprehensive client summary metrics"""
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
            'conversion_rate': (self.df['Orders'].sum() / self.df['Clicks'].sum() * 100) if self.df['Clicks'].sum() > 0 else 0,
            'keywords_count': len(self.df),
            'campaigns_count': int(self.df['Campaign Name'].nunique())
        }

    def get_health_score(self) -> int:
        """Calculate campaign health score (0-100)"""
        summary = self.get_client_summary()
        score = 0

        # ROAS contribution (50 points)
        roas = summary['roas']
        if roas >= 3.5:
            score += 50
        elif roas >= 2.5:
            score += 40
        elif roas >= 1.5:
            score += 25
        elif roas > 0:
            score += 10

        # Wastage contribution (30 points)
        wastage_pct = (summary['total_wastage'] / summary['total_spend'] * 100) if summary['total_spend'] > 0 else 0
        if wastage_pct <= 10:
            score += 30
        elif wastage_pct <= 20:
            score += 20
        elif wastage_pct <= 30:
            score += 10
        elif wastage_pct <= 40:
            score += 5

        # Efficiency contribution (20 points)
        avg_ctr = summary['avg_ctr']
        if avg_ctr >= 5.0:
            score += 20
        elif avg_ctr >= 3.0:
            score += 15
        elif avg_ctr >= 1.0:
            score += 10
        elif avg_ctr > 0:
            score += 5

        return min(score, 100)

    def classify_keywords(self) -> Dict[str, List[Dict]]:
        """Classify keywords into actionable categories"""
        categories = {
            'champions': [],
            'opportunities': [],
            'pause_now': [],
            'needs_optimization': []
        }

        for _, row in self.df.iterrows():
            try:
                spend = float(row.get('Spend', 0))
                sales = float(row.get('Sales', 0))
                roas = float(row.get('ROAS', 0))
                orders = int(row.get('Orders', 0))
                clicks = int(row.get('Clicks', 0))

                kw_data = {
                    'Keyword': str(row['Customer Search Term']),
                    'Spend': f"₹{spend:.2f}",
                    'Sales': f"₹{sales:.2f}",
                    'Profit': f"₹{sales - spend:.2f}",
                    'ROAS': f"{roas:.2f}x",
                    'Orders': orders,
                    'Clicks': clicks,
                    'Campaign': row['Campaign Name'],
                    'Match Type': row.get('Match Type', 'N/A'),
                    'Action': ''
                }

                # Classification logic
                if roas >= 3.0 and orders >= 2 and spend >= 10:
                    kw_data['Action'] = '🚀 SCALE: Increase bid by 25%'
                    categories['champions'].append(kw_data)

                elif spend >= 30 and sales == 0:
                    kw_data['Action'] = '⛔ PAUSE: Add as negative keyword immediately'
                    categories['pause_now'].append(kw_data)

                elif roas >= 1.5 and roas < 3.0 and spend >= 10:
                    kw_data['Action'] = '⚡ OPTIMIZE: Test bid adjustments (+10%)'
                    categories['opportunities'].append(kw_data)

                elif spend >= 15 and roas < 1.5:
                    kw_data['Action'] = '⚠️ REDUCE: Decrease bid by 30% or pause'
                    categories['needs_optimization'].append(kw_data)

                elif spend >= 20 and sales > 0 and orders == 0:
                    kw_data['Action'] = '👀 MONITOR: Sales but no orders yet'
                    categories['opportunities'].append(kw_data)

            except Exception as e:
                continue

        # Sort by most impactful first
        categories['champions'] = sorted(categories['champions'], 
                                        key=lambda x: float(x['Sales'].replace('₹', '')), reverse=True)
        categories['pause_now'] = sorted(categories['pause_now'], 
                                        key=lambda x: float(x['Spend'].replace('₹', '')), reverse=True)
        categories['needs_optimization'] = sorted(categories['needs_optimization'], 
                                                 key=lambda x: float(x['Spend'].replace('₹', '')), reverse=True)

        return categories

    def generate_client_report(self) -> str:
        """Generate professional text report for client"""
        summary = self.get_client_summary()
        health = self.get_health_score()
        classification = self.classify_keywords()

        # Determine health status
        if health >= 70:
            health_emoji = "🟢"
            health_text = "Excellent"
        elif health >= 50:
            health_emoji = "🟡"
            health_text = "Good"
        else:
            health_emoji = "🔴"
            health_text = "Needs Attention"

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     AMAZON PPC PERFORMANCE REPORT                            ║
║                        Client: {self.client_name:^42}║
║                        Date: {datetime.now().strftime('%B %d, %Y'):^44}║
╚══════════════════════════════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
═════════════════════════════════════════════════════════════════════════════

Campaign Health Score: {health}/100 {health_emoji} {health_text}

This report provides a comprehensive analysis of your Amazon PPC campaigns
for the reporting period. Key metrics, recommendations, and action items are
detailed below.

FINANCIAL PERFORMANCE
─────────────────────────────────────────────────────────────────────────────
Total Ad Spend:              ₹{summary['total_spend']:>15,.2f}
Total Sales Generated:       ₹{summary['total_sales']:>15,.2f}
Net Profit (Sales - Spend):  ₹{summary['total_profit']:>15,.2f}
Return on Ad Spend (ROAS):   {summary['roas']:>16.2f}x
Advertising Cost of Sales:   {summary['acos']:>15.1f}%

CAMPAIGN ENGAGEMENT METRICS
─────────────────────────────────────────────────────────────────────────────
Total Impressions:           {summary['total_impressions']:>16,}
Total Clicks:                {summary['total_clicks']:>16,}
Total Orders:                {summary['total_orders']:>16,}
Click-Through Rate (CTR):    {summary['avg_ctr']:>15.2f}%
Conversion Rate:             {summary['conversion_rate']:>15.2f}%
Average Cost Per Click:      ₹{summary['avg_cpc']:>15.2f}

CAMPAIGN STRUCTURE
─────────────────────────────────────────────────────────────────────────────
Active Campaigns:            {summary['campaigns_count']:>16}
Total Keywords Analyzed:     {summary['keywords_count']:>16}
Avg Spend per Keyword:       ₹{summary['total_spend']/summary['keywords_count']:>15.2f}

BUDGET EFFICIENCY ANALYSIS
─────────────────────────────────────────────────────────────────────────────
Total Wasted Spend:          ₹{summary['total_wastage']:>15,.2f}
Wastage Percentage:          {(summary['total_wastage']/summary['total_spend']*100) if summary['total_spend'] > 0 else 0:>15.1f}%
Zero-Sales Keywords:         {len(self.df[self.df['Sales'] == 0]):>16}

KEYWORD PERFORMANCE BREAKDOWN
─────────────────────────────────────────────────────────────────────────────
🏆 Champion Keywords:         {len(classification['champions']):>16}
   (High ROAS, proven sales - ready to scale)

⚡ Optimization Opportunities:{len(classification['opportunities']):>16}
   (Decent performance with room for improvement)

⚠️  Needs Optimization:        {len(classification['needs_optimization']):>16}
   (Underperforming - requires bid reduction)

🚨 Pause Immediately:         {len(classification['pause_now']):>16}
   (High spend with zero sales - urgent action needed)

"""

        # Add performance-specific recommendations
        if health >= 70:
            report += """
RECOMMENDATIONS - SCALING STRATEGY
─────────────────────────────────────────────────────────────────────────────
✅ Your campaigns are performing excellently! Key actions:

1. SCALE WINNERS: Increase budget by 20-30% on top campaigns
2. EXPAND KEYWORDS: Test variations of your champion keywords
3. INCREASE BIDS: Raise bids by 15-25% on high ROAS keywords
4. NEW CAMPAIGNS: Create dedicated campaigns for best performers
5. MAINTAIN MOMENTUM: Continue current optimization practices

Your campaigns are highly profitable and ready for aggressive scaling.
"""
        elif health >= 50:
            report += """
RECOMMENDATIONS - OPTIMIZATION STRATEGY
─────────────────────────────────────────────────────────────────────────────
📊 Your campaigns show good performance with optimization opportunities:

1. OPTIMIZE BIDS: Fine-tune bids on moderate performers
2. ADD NEGATIVES: Implement negative keyword recommendations below
3. TEST & LEARN: Increase bids by 10% on opportunity keywords
4. MONITOR CLOSELY: Review performance weekly
5. GRADUAL SCALING: Consider 10-15% budget increase after optimization

Focus on improving efficiency before aggressive scaling.
"""
        else:
            report += """
RECOMMENDATIONS - URGENT OPTIMIZATION NEEDED
─────────────────────────────────────────────────────────────────────────────
⚠️  Your campaigns require immediate optimization:

1. PAUSE WASTERS: Immediately pause high-spend zero-sales keywords
2. REDUCE BIDS: Decrease bids by 30-40% on underperformers
3. ADD NEGATIVES: Implement all negative keyword recommendations
4. REVIEW TARGETING: Assess product targeting and audience selection
5. BUDGET REDUCTION: Consider reducing budget by 15-20% until optimized

Focus on stopping wastage before investing more budget.
"""

        # Add specific action items
        report += f"""

IMMEDIATE ACTION ITEMS (Next 7 Days)
─────────────────────────────────────────────────────────────────────────────
□ Download and upload negative keyword file ({len(classification['pause_now'])} keywords)
□ Increase bids on {len(classification['champions'])} champion keywords
□ Reduce or pause {len(classification['needs_optimization'])} underperforming keywords
□ Review and optimize {summary['campaigns_count']} active campaigns
□ Schedule follow-up review in 7 days to assess impact

"""

        # Add top performers section
        if classification['champions']:
            report += """
TOP 5 CHAMPION KEYWORDS (Scale These First)
─────────────────────────────────────────────────────────────────────────────
"""
            for i, kw in enumerate(classification['champions'][:5], 1):
                report += f"{i}. {kw['Keyword']}
"
                report += f"   Spend: {kw['Spend']} → Sales: {kw['Sales']} | ROAS: {kw['ROAS']}
"
                report += f"   Action: {kw['Action']}

"

        # Add wastage section
        if classification['pause_now']:
            total_waste = sum(float(kw['Spend'].replace('₹','')) for kw in classification['pause_now'])
            report += f"""
TOP BUDGET WASTERS (Pause These Immediately)
─────────────────────────────────────────────────────────────────────────────
Total Wasted: ₹{total_waste:,.2f}

"""
            for i, kw in enumerate(classification['pause_now'][:5], 1):
                report += f"{i}. {kw['Keyword']}
"
                report += f"   Wasted: {kw['Spend']} | Action: {kw['Action']}

"

        # Footer
        report += """
═════════════════════════════════════════════════════════════════════════════
NEXT REVIEW DATE
─────────────────────────────────────────────────────────────────────────────
Recommended: """ + (datetime.now() + timedelta(days=7)).strftime('%B %d, %Y') + """

Please review this report and implement the recommended actions. For any
questions or to discuss strategy, please contact your account manager.

═════════════════════════════════════════════════════════════════════════════
Report Generated By: Amazon Ads Agency Dashboard Pro
Contact: Your Agency Name | your-email@agency.com
═════════════════════════════════════════════════════════════════════════════
"""

        return report

    def get_top_campaigns(self, limit: int = 5) -> pd.DataFrame:
        """Get top performing campaigns"""
        if 'Campaign Name' not in self.df.columns:
            return pd.DataFrame()

        campaign_perf = self.df.groupby('Campaign Name').agg({
            'Spend': 'sum',
            'Sales': 'sum',
            'Orders': 'sum',
            'Profit': 'sum',
            'Clicks': 'sum'
        }).sort_values('Sales', ascending=False).head(limit)

        # Calculate ROAS
        campaign_perf['ROAS'] = campaign_perf['Sales'] / campaign_perf['Spend']

        return campaign_perf

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'clients' not in st.session_state:
        st.session_state.clients = {}
    if 'active_client' not in st.session_state:
        st.session_state.active_client = None
    if 'agency_name' not in st.session_state:
        st.session_state.agency_name = "Your Agency Name"
    if 'agency_logo_url' not in st.session_state:
        st.session_state.agency_logo_url = None

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_agency_header():
    """Render agency branding header"""
    st.markdown(f"""
    <div class="agency-header">
        <h1>🏢 {st.session_state.agency_name}</h1>
        <p style="font-size: 1.3rem; margin-top: 0.8rem; font-weight: 600;">
            Amazon Ads Agency Dashboard Pro
        </p>
        <p style="font-size: 0.95rem; opacity: 0.9; margin-top: 0.5rem;">
            Professional Multi-Client PPC Management Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with client management"""
    with st.sidebar:
        # Agency settings
        with st.expander("⚙️ Agency Settings", expanded=False):
            new_agency_name = st.text_input("Agency Name", value=st.session_state.agency_name, key='agency_name_input')
            if new_agency_name != st.session_state.agency_name:
                st.session_state.agency_name = new_agency_name
                st.rerun()

            st.info("💡 Tip: Your agency name appears on reports and headers")

        st.markdown("---")
        st.markdown("### 👥 Client Management")

        # Active client selector
        if st.session_state.clients:
            client_names = list(st.session_state.clients.keys())
            selected_client = st.selectbox(
                "Select Active Client",
                client_names,
                key='client_selector'
            )
            st.session_state.active_client = selected_client

            # Display client quick info
            if selected_client:
                client = st.session_state.clients[selected_client]
                if client.analyzer:
                    health = client.analyzer.get_health_score()
                    health_emoji = "🟢" if health >= 70 else "🟡" if health >= 50 else "🔴"

                    st.info(f"""
                    {health_emoji} **Health:** {health}/100
                    **Industry:** {client.industry}
                    **Budget:** ₹{client.monthly_budget:,.0f}
                    **Fee:** ₹{client.monthly_fee:,.0f}/mo
                    **Added:** {client.added_date.strftime('%Y-%m-%d')}
                    """)

        st.markdown("---")

        # Add new client section
        with st.expander("➕ Add New Client", expanded=False):
            client_name = st.text_input("Client Name*", placeholder="ABC Company Ltd.", key='new_client_name')

            col1, col2 = st.columns(2)
            with col1:
                industry = st.selectbox("Industry", [
                    "E-commerce", "Electronics", "Fashion & Apparel", "Beauty & Cosmetics",
                    "Home & Kitchen", "Sports & Fitness", "Books & Media", "Health & Wellness",
                    "Toys & Games", "Automotive", "Office Supplies", "Pet Supplies", "Other"
                ], key='new_client_industry')

            with col2:
                monthly_budget = st.number_input("Monthly Budget (₹)", min_value=5000, max_value=10000000, 
                                                value=50000, step=5000, key='new_client_budget')

            monthly_fee = st.number_input("Your Monthly Fee (₹)", min_value=0, max_value=1000000,
                                         value=10000, step=1000, key='new_client_fee')

            contact_email = st.text_input("Contact Email", placeholder="client@company.com", key='new_client_email')
            contact_phone = st.text_input("Contact Phone", placeholder="+91 XXXXX XXXXX", key='new_client_phone')

            uploaded_file = st.file_uploader(
                "Upload Search Term Report*",
                type=["xlsx", "xls"],
                key='new_client_file',
                help="Download from Amazon Ads Console → Reports → Search Term Report"
            )

            if st.button("✅ Add Client", type="primary", use_container_width=True):
                if client_name and uploaded_file:
                    try:
                        with st.spinner(f"Analyzing {client_name}'s data..."):
                            # Read file
                            df = pd.read_excel(uploaded_file)

                            # Validate required columns
                            required_cols = ["Customer Search Term", "Spend", "Clicks"]
                            missing = [col for col in required_cols if col not in df.columns]

                            if missing:
                                st.error(f"❌ Missing columns: {', '.join(missing)}")
                            else:
                                # Create client data
                                client_data = ClientData(client_name, industry, monthly_budget)
                                client_data.monthly_fee = monthly_fee
                                client_data.contact_email = contact_email
                                client_data.contact_phone = contact_phone

                                # Create analyzer
                                client_data.analyzer = AgencyAnalyzer(df, client_name)

                                # Add to session state
                                st.session_state.clients[client_name] = client_data
                                st.session_state.active_client = client_name

                                st.success(f"✅ Successfully added {client_name}!")
                                st.balloons()
                                st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
                        st.info("Please ensure you've uploaded a valid Amazon Search Term Report")
                else:
                    st.warning("⚠️ Please provide client name and upload a file")

        # Client list management
        if st.session_state.clients:
            st.markdown("---")
            st.markdown("### 📋 All Clients")

            for client_name, client_data in list(st.session_state.clients.items()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if client_data.analyzer:
                        health = client_data.analyzer.get_health_score()
                        health_icon = "🟢" if health >= 70 else "🟡" if health >= 50 else "🔴"
                        st.text(f"{health_icon} {client_name}")
                    else:
                        st.text(f"📊 {client_name}")
                with col2:
                    if st.button("❌", key=f"del_{client_name}"):
                        del st.session_state.clients[client_name]
                        if st.session_state.active_client == client_name:
                            st.session_state.active_client = None
                        st.rerun()

        # Agency quick stats
        if st.session_state.clients:
            st.markdown("---")
            st.markdown("### 📊 Agency Stats")
            total_clients = len(st.session_state.clients)
            total_revenue = sum(c.monthly_fee for c in st.session_state.clients.values())

            st.metric("Total Clients", total_clients)
            st.metric("Monthly Revenue", f"₹{total_revenue:,.0f}")
            st.metric("Annual Revenue", f"₹{total_revenue * 12:,.0f}")

def render_welcome_screen():
    """Welcome screen when no clients are loaded"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="client-card">
            <h3 style="margin-top: 0;">👥 Step 1: Add Clients</h3>
            <p>Add client details and upload their Amazon Search Term Reports from the sidebar</p>
            <ul style="margin-bottom: 0;">
                <li>Client information</li>
                <li>Budget & fees</li>
                <li>Contact details</li>
                <li>PPC report upload</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="client-card">
            <h3 style="margin-top: 0;">📊 Step 2: Analyze</h3>
            <p>Get instant AI-powered analysis with health scoring and keyword classification</p>
            <ul style="margin-bottom: 0;">
                <li>Campaign health score</li>
                <li>Keyword classification</li>
                <li>Performance metrics</li>
                <li>Budget efficiency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="client-card">
            <h3 style="margin-top: 0;">📝 Step 3: Report</h3>
            <p>Generate professional reports and export action items for implementation</p>
            <ul style="margin-bottom: 0;">
                <li>Executive summaries</li>
                <li>Negative keywords</li>
                <li>Scaling recommendations</li>
                <li>Multi-format exports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 👥 Client Management
        - **Unlimited clients** - No restrictions
        - **Client profiles** - Store all details
        - **Health monitoring** - 0-100 scoring
        - **Revenue tracking** - Monthly fees

        #### 📊 Performance Tracking
        - **Individual dashboards** - Per client
        - **Cross-client comparison** - Aggregate view
        - **Campaign rankings** - Top performers
        - **Budget utilization** - Spend tracking
        """)

    with col2:
        st.markdown("""
        #### 📝 Professional Reporting
        - **Branded reports** - Your agency name
        - **Executive summaries** - High-level view
        - **Action recommendations** - Clear steps
        - **Multi-format export** - TXT, Excel, CSV

        #### 💰 Revenue Management
        - **Fee tracking** - Per client
        - **Total revenue** - Agency wide
        - **Projections** - Annual forecast
        - **Profitability** - Client analysis
        """)

    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.info("""
    👈 **Click "Add New Client"** in the sidebar to add your first client and start analyzing their campaigns!

    You'll need:
    - Client name and details
    - Their Amazon Search Term Report (Excel file)
    - Budget and fee information
    """)

# ============================================================================
# DASHBOARD TABS
# ============================================================================

def render_dashboard_tab(client: ClientData, analyzer: AgencyAnalyzer):
    """Render main dashboard overview"""
    st.header(f"📊 {client.name} - Campaign Dashboard")

    # Get metrics
    summary = analyzer.get_client_summary()
    health = analyzer.get_health_score()

    # Health score alert
    health_color = "🟢" if health >= 70 else "🟡" if health >= 50 else "🔴"
    health_status = "Excellent - Ready to Scale!" if health >= 70 else "Good - Optimization Opportunities" if health >= 50 else "Needs Immediate Attention"

    st.markdown(f"""
    <div class="info-box">
        <h2 style="margin: 0; font-size: 1.8rem;">{health_color} Campaign Health Score: {health}/100</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; font-weight: 600;">{health_status}</p>
        <p style="margin: 0.3rem 0 0 0; opacity: 0.9;">Based on ROAS, efficiency, and wastage analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Primary metrics
    st.subheader("💰 Financial Performance")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Spend", f"₹{summary['total_spend']:,.0f}")
    with col2:
        st.metric("Total Sales", f"₹{summary['total_sales']:,.0f}")
    with col3:
        profit_delta = "Profitable ✅" if summary['total_profit'] > 0 else "Loss ⚠️"
        st.metric("Net Profit", f"₹{summary['total_profit']:,.0f}", delta=profit_delta)
    with col4:
        roas_delta = "Excellent" if summary['roas'] >= 3 else "Good" if summary['roas'] >= 2 else "Low"
        st.metric("ROAS", f"{summary['roas']:.2f}x", delta=roas_delta)
    with col5:
        st.metric("Total Orders", f"{summary['total_orders']:,}")

    st.markdown("---")

    # Secondary metrics
    st.subheader("📈 Campaign Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Avg CPC", f"₹{summary['avg_cpc']:.2f}")
    with col2:
        acos_delta = "Good" if summary['acos'] <= 30 else "High" if summary['acos'] <= 50 else "Very High"
        st.metric("ACOS", f"{summary['acos']:.1f}%", delta=acos_delta)
    with col3:
        st.metric("Total Clicks", f"{summary['total_clicks']:,}")
    with col4:
        st.metric("Avg CTR", f"{summary['avg_ctr']:.2f}%")
    with col5:
        st.metric("Conv. Rate", f"{summary['conversion_rate']:.2f}%")

    st.markdown("---")

    # Campaign structure
    st.subheader("🎯 Campaign Structure")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Active Campaigns", f"{summary['campaigns_count']}")
    with col2:
        st.metric("Total Keywords", f"{summary['keywords_count']:,}")
    with col3:
        avg_spend_per_kw = summary['total_spend'] / summary['keywords_count'] if summary['keywords_count'] > 0 else 0
        st.metric("Avg Spend/Keyword", f"₹{avg_spend_per_kw:.2f}")

    # Wastage alert
    if summary['total_wastage'] > 0:
        wastage_pct = (summary['total_wastage'] / summary['total_spend'] * 100) if summary['total_spend'] > 0 else 0
        st.markdown("---")
        st.markdown(f"""
        <div class="danger-box">
            <strong>🚨 Budget Wastage Alert!</strong><br>
            <strong style="font-size: 1.2rem;">₹{summary['total_wastage']:,.2f}</strong> spent on keywords with ZERO sales 
            (<strong>{wastage_pct:.1f}%</strong> of total spend)<br>
            <br>
            ⚡ Action Required: Download and upload negative keywords immediately to stop wastage.
        </div>
        """, unsafe_allow_html=True)

    # Budget utilization
    if client.monthly_budget > 0:
        st.markdown("---")
        st.subheader("💼 Budget Utilization")
        budget_used_pct = (summary['total_spend'] / client.monthly_budget * 100)

        # Create progress bar
        progress_color = "green" if budget_used_pct <= 90 else "orange" if budget_used_pct <= 100 else "red"
        st.progress(min(budget_used_pct / 100, 1.0))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spent", f"₹{summary['total_spend']:,.0f}")
        with col2:
            st.metric("Budget", f"₹{client.monthly_budget:,.0f}")
        with col3:
            remaining = client.monthly_budget - summary['total_spend']
            st.metric("Remaining", f"₹{remaining:,.0f}")

        if budget_used_pct > 100:
            st.warning(f"⚠️ Over budget by {budget_used_pct - 100:.1f}% (₹{summary['total_spend'] - client.monthly_budget:,.0f})")
        elif budget_used_pct > 90:
            st.info(f"ℹ️ Used {budget_used_pct:.1f}% of monthly budget")

def render_keywords_tab(analyzer: AgencyAnalyzer):
    """Render keyword analysis tab"""
    st.header("🎯 Keyword Analysis & Classification")

    classification = analyzer.classify_keywords()

    # Summary cards
    st.subheader("📊 Classification Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🏆 Champions", len(classification['champions']), 
                 help="High ROAS keywords ready to scale")
    with col2:
        st.metric("⚡ Opportunities", len(classification['opportunities']),
                 help="Good performance with room for improvement")
    with col3:
        st.metric("⚠️ Needs Work", len(classification['needs_optimization']),
                 help="Underperforming keywords requiring attention")
    with col4:
        st.metric("🚨 Pause Now", len(classification['pause_now']),
                 help="High spend with zero sales - urgent action needed")

    st.markdown("---")

    # Detailed tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🏆 Champions ({len(classification['champions'])})",
        f"⚡ Opportunities ({len(classification['opportunities'])})",
        f"⚠️ Needs Work ({len(classification['needs_optimization'])})",
        f"🚨 Pause Now ({len(classification['pause_now'])})"
    ])

    with tab1:
        if classification['champions']:
            st.markdown("""
            <div class="success-box">
                <strong>🏆 Your Champion Keywords - Scale These Aggressively!</strong><br><br>
                These keywords have proven high ROAS and consistent sales. They are your best performers
                and prime candidates for budget increases.<br><br>
                <strong>Recommended Action:</strong> Increase bids by 15-25% to capture more volume
            </div>
            """, unsafe_allow_html=True)

            df_champions = pd.DataFrame(classification['champions'])
            st.dataframe(df_champions, use_container_width=True, hide_index=True, height=400)

            # Quick stats
            total_spend = sum(float(k['Spend'].replace('₹','')) for k in classification['champions'])
            total_sales = sum(float(k['Sales'].replace('₹','')) for k in classification['champions'])
            st.success(f"💎 Champions generated ₹{total_sales:,.0f} in sales from ₹{total_spend:,.0f} spend")
        else:
            st.info("😊 No champion keywords identified yet. Focus on optimization to develop winners!")

    with tab2:
        if classification['opportunities']:
            st.markdown("""
            <div class="info-box">
                <strong>⚡ Optimization Opportunities - Test and Improve</strong><br><br>
                These keywords show decent performance but have room for improvement. With proper
                optimization, many can become champions.<br><br>
                <strong>Recommended Action:</strong> Test 10-15% bid increases and monitor closely
            </div>
            """, unsafe_allow_html=True)

            df_opps = pd.DataFrame(classification['opportunities'])
            st.dataframe(df_opps, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No optimization opportunities identified")

    with tab3:
        if classification['needs_optimization']:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Underperforming Keywords - Reduce or Pause</strong><br><br>
                These keywords are consuming budget without delivering adequate returns. They need
                immediate attention to improve ROI.<br><br>
                <strong>Recommended Action:</strong> Reduce bids by 30% or consider pausing
            </div>
            """, unsafe_allow_html=True)

            df_needs = pd.DataFrame(classification['needs_optimization'])
            st.dataframe(df_needs, use_container_width=True, hide_index=True, height=400)

            total_waste = sum(float(k['Spend'].replace('₹','')) for k in classification['needs_optimization'])
            st.warning(f"⚠️ These keywords consumed ₹{total_waste:,.0f} with poor performance")
        else:
            st.success("✅ No keywords need optimization - great job!")

    with tab4:
        if classification['pause_now']:
            total_wasted = sum(float(k['Spend'].replace('₹','')) for k in classification['pause_now'])

            st.markdown(f"""
            <div class="danger-box">
                <strong>🚨 URGENT: Pause These Keywords Immediately!</strong><br><br>
                These keywords have consumed significant budget with ZERO sales. They represent pure
                wastage and should be paused and added as negative keywords immediately.<br><br>
                <strong>Total Wasted:</strong> ₹{total_wasted:,.2f}<br>
                <strong>Action:</strong> Download negative keyword file below and upload to Amazon Ads
            </div>
            """, unsafe_allow_html=True)

            df_pause = pd.DataFrame(classification['pause_now'])
            st.dataframe(df_pause, use_container_width=True, hide_index=True, height=400)

            st.error(f"💸 Total wasted: ₹{total_wasted:,.2f} - Stop this bleeding immediately!")
        else:
            st.success("🎉 Excellent! No urgent issues - your campaigns are well-optimized!")

def render_performance_tab(analyzer: AgencyAnalyzer):
    """Render performance analytics tab"""
    st.header("📈 Performance Analytics")

    # Top campaigns
    st.subheader("🎯 Top Performing Campaigns")
    top_campaigns = analyzer.get_top_campaigns(10)

    if not top_campaigns.empty:
        # Format for display
        display_df = top_campaigns.copy()
        display_df['Spend'] = display_df['Spend'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Sales'] = display_df['Sales'].apply(lambda x: f"₹{x:,.2f}")
        display_df['Profit'] = display_df['Profit'].apply(lambda x: f"₹{x:,.2f}")
        display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.2f}x")
        display_df['Orders'] = display_df['Orders'].astype(int)
        display_df['Clicks'] = display_df['Clicks'].astype(int)

        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No campaign data available")

    st.markdown("---")

    # Additional analytics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Match Type Performance")
        if 'Match Type' in analyzer.df.columns:
            match_perf = analyzer.df.groupby('Match Type').agg({
                'Spend': 'sum',
                'Sales': 'sum',
                'Orders': 'sum',
                'Clicks': 'sum'
            })

            # Calculate ROAS
            match_perf['ROAS'] = match_perf['Sales'] / match_perf['Spend']

            # Format
            match_display = match_perf.copy()
            match_display['Spend'] = match_display['Spend'].apply(lambda x: f"₹{x:,.2f}")
            match_display['Sales'] = match_display['Sales'].apply(lambda x: f"₹{x:,.2f}")
            match_display['ROAS'] = match_display['ROAS'].apply(lambda x: f"{x:.2f}x")
            match_display['Orders'] = match_display['Orders'].astype(int)
            match_display['Clicks'] = match_display['Clicks'].astype(int)

            st.dataframe(match_display, use_container_width=True)
        else:
            st.info("Match type data not available")

    with col2:
        st.subheader("💰 Spend Distribution")
        summary = analyzer.get_client_summary()

        st.metric("Total Campaigns", summary['campaigns_count'])
        st.metric("Total Keywords", f"{summary['keywords_count']:,}")
        st.metric("Avg Spend/Campaign", 
                 f"₹{summary['total_spend']/summary['campaigns_count']:,.2f}" if summary['campaigns_count'] > 0 else "₹0.00")
        st.metric("Avg Spend/Keyword", 
                 f"₹{summary['total_spend']/summary['keywords_count']:,.2f}" if summary['keywords_count'] > 0 else "₹0.00")

def render_client_report_tab(client: ClientData, analyzer: AgencyAnalyzer):
    """Render client report generation tab"""
    st.header("📝 Client Report Generator")

    st.markdown("""
    <div class="info-box">
        <strong>📄 Professional Client Report</strong><br>
        Generate a comprehensive, professional report to share with your client. 
        Includes executive summary, detailed metrics, and actionable recommendations.
    </div>
    """, unsafe_allow_html=True)

    # Generate report
    report_text = analyzer.generate_client_report()

    # Report preview
    st.subheader("📋 Report Preview")
    st.text_area("", report_text, height=600, label_visibility="collapsed")

    st.markdown("---")

    # Download options
    st.subheader("📥 Download Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📄 Download as TXT",
            data=report_text,
            file_name=f"Amazon_PPC_Report_{client.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Plain text format - easy to read and share"
        )

    with col2:
        # Create Excel report
        summary = analyzer.get_client_summary()

        # Create summary sheet
        summary_data = pd.DataFrame([{
            'Client': client.name,
            'Report Date': datetime.now().strftime('%Y-%m-%d'),
            'Health Score': analyzer.get_health_score(),
            'Total Spend': summary['total_spend'],
            'Total Sales': summary['total_sales'],
            'Net Profit': summary['total_profit'],
            'ROAS': summary['roas'],
            'ACOS': summary['acos'],
            'Total Orders': summary['total_orders'],
            'Total Clicks': summary['total_clicks'],
            'Conversion Rate': summary['conversion_rate'],
            'Wastage': summary['total_wastage']
        }])

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            analyzer.df.to_excel(writer, sheet_name='Detailed Data', index=False)
        output.seek(0)

        st.download_button(
            label="📊 Download as Excel",
            data=output,
            file_name=f"Amazon_PPC_Report_{client.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Excel format with multiple sheets for detailed analysis"
        )

    with col3:
        # Email option
        if st.button("📧 Prepare Email", use_container_width=True, help="Prepare email to send to client"):
            if client.contact_email:
                st.success(f"✅ Report ready to send to: {client.contact_email}")

                email_template = f"""
Subject: Amazon PPC Performance Report - {datetime.now().strftime('%B %Y')}

Dear {client.name},

Please find attached your monthly Amazon PPC performance report.

Key Highlights:
- Health Score: {analyzer.get_health_score()}/100
- ROAS: {summary['roas']:.2f}x
- Total Sales: ₹{summary['total_sales']:,.2f}
- Net Profit: ₹{summary['total_profit']:,.2f}

The detailed report is attached. Let's schedule a call to discuss the recommendations.

Best regards,
{st.session_state.agency_name}
"""
                st.text_area("Email Template", email_template, height=250)
                st.info("💡 Copy this template and paste into your email client with the attached report")
            else:
                st.warning("⚠️ No email address on file for this client. Add it in the sidebar.")

def render_all_clients_tab():
    """Render all clients comparison tab"""
    st.header("🏢 All Clients Overview")

    if not st.session_state.clients:
        st.info("No clients added yet. Add clients from the sidebar to see comparison.")
        return

    # Aggregate all client data
    all_clients_data = []
    for name, client in st.session_state.clients.items():
        if client.analyzer:
            summary = client.analyzer.get_client_summary()
            health = client.analyzer.get_health_score()

            # Determine health emoji
            health_emoji = "🟢" if health >= 70 else "🟡" if health >= 50 else "🔴"

            all_clients_data.append({
                'Client': name,
                'Status': health_emoji,
                'Health': f"{health}/100",
                'Spend': f"₹{summary['total_spend']:,.0f}",
                'Sales': f"₹{summary['total_sales']:,.0f}",
                'Profit': f"₹{summary['total_profit']:,.0f}",
                'ROAS': f"{summary['roas']:.2f}x",
                'ACOS': f"{summary['acos']:.1f}%",
                'Orders': int(summary['total_orders']),
                'Campaigns': summary['campaigns_count'],
                'Monthly Fee': f"₹{client.monthly_fee:,.0f}",
                'Industry': client.industry,
                'Budget': f"₹{client.monthly_budget:,.0f}"
            })

    if all_clients_data:
        st.subheader("📊 Client Comparison")
        df_clients = pd.DataFrame(all_clients_data)
        st.dataframe(df_clients, use_container_width=True, hide_index=True)

        # Find best performer
        best_roas_client = max(all_clients_data, key=lambda x: float(x['ROAS'].replace('x','')))
        st.success(f"🏆 Best ROAS: {best_roas_client['Client']} with {best_roas_client['ROAS']}")

        st.markdown("---")

        # Agency totals
        st.subheader("💼 Agency Performance Summary")

        total_clients = len(all_clients_data)
        total_monthly_revenue = sum(c.monthly_fee for c in st.session_state.clients.values())
        total_annual_revenue = total_monthly_revenue * 12
        avg_fee_per_client = total_monthly_revenue / total_clients if total_clients > 0 else 0

        # Calculate aggregate metrics
        total_spend_all = sum(float(d['Spend'].replace('₹','').replace(',','')) for d in all_clients_data)
        total_sales_all = sum(float(d['Sales'].replace('₹','').replace(',','')) for d in all_clients_data)
        total_profit_all = sum(float(d['Profit'].replace('₹','').replace(',','')) for d in all_clients_data)
        total_orders_all = sum(d['Orders'] for d in all_clients_data)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Clients", total_clients)
        with col2:
            st.metric("Monthly Revenue", f"₹{total_monthly_revenue:,.0f}")
        with col3:
            st.metric("Annual Revenue", f"₹{total_annual_revenue:,.0f}")
        with col4:
            st.metric("Avg Fee/Client", f"₹{avg_fee_per_client:,.0f}")
        with col5:
            st.metric("Total Orders (All)", total_orders_all)

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Spend (All)", f"₹{total_spend_all:,.0f}")
        with col2:
            st.metric("Total Sales (All)", f"₹{total_sales_all:,.0f}")
        with col3:
            st.metric("Total Profit (All)", f"₹{total_profit_all:,.0f}")
        with col4:
            avg_roas_all = total_sales_all / total_spend_all if total_spend_all > 0 else 0
            st.metric("Avg ROAS (All)", f"{avg_roas_all:.2f}x")
    else:
        st.info("Add clients with uploaded reports to see comparison data")

def render_exports_tab(analyzer: AgencyAnalyzer):
    """Render exports tab with download options"""
    st.header("📥 Export Client Data")

    st.markdown("""
    <div class="info-box">
        <strong>📁 Export Options</strong><br>
        Download ready-to-upload files for Amazon Ads bulk operations and internal analysis
    </div>
    """, unsafe_allow_html=True)

    classification = analyzer.classify_keywords()

    col1, col2, col3 = st.columns(3)

    # Negative keywords export
    with col1:
        st.subheader("🚫 Negative Keywords")
        st.caption("Amazon Ads bulk upload format")

        pause_kws = classification['pause_now']

        if pause_kws:
            # Create Amazon-ready format
            export_data = []
            for kw in pause_kws:
                export_data.append({
                    'Campaign': kw['Campaign'],
                    'Ad Group': '',
                    'Keyword': kw['Keyword'],
                    'Match Type': 'Negative Exact',
                    'Status': 'Enabled'
                })

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                pd.DataFrame(export_data).to_excel(writer, index=False, sheet_name='Negative Keywords')
            output.seek(0)

            st.download_button(
                label=f"📥 Download ({len(export_data)} keywords)",
                data=output,
                file_name=f"Amazon_Negative_Keywords_{analyzer.client_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            total_waste = sum(float(kw['Spend'].replace('₹','')) for kw in pause_kws)
            st.success(f"✅ {len(export_data)} keywords")
            st.info(f"💰 Potential savings: ₹{total_waste:,.2f}")
        else:
            st.success("🎉 No negative keywords needed!")
            st.info("Your campaigns are well-optimized")

    # Champions export
    with col2:
        st.subheader("🏆 Champion Keywords")
        st.caption("Scale these winners")

        if classification['champions']:
            champions_data = []
            for kw in classification['champions']:
                champions_data.append({
                    'Keyword': kw['Keyword'],
                    'Current Spend': kw['Spend'],
                    'Sales': kw['Sales'],
                    'ROAS': kw['ROAS'],
                    'Orders': kw['Orders'],
                    'Campaign': kw['Campaign'],
                    'Recommended Action': kw['Action']
                })

            output_csv = pd.DataFrame(champions_data).to_csv(index=False)

            st.download_button(
                label=f"📥 Download ({len(champions_data)} keywords)",
                data=output_csv,
                file_name=f"Champion_Keywords_{analyzer.client_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success(f"✅ {len(champions_data)} keywords")
            st.info("💎 Your best performers")
        else:
            st.info("No champions identified yet")
            st.caption("Focus on optimization first")

    # Complete data export
    with col3:
        st.subheader("📊 Complete Analysis")
        st.caption("Full dataset with all metrics")

        complete_csv = analyzer.df.to_csv(index=False)

        st.download_button(
            label=f"📥 Download ({len(analyzer.df)} keywords)",
            data=complete_csv,
            file_name=f"Complete_Analysis_{analyzer.client_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.success(f"✅ All {len(analyzer.df)} keywords")
        st.info("📈 Complete dataset")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def render_dashboard():
    """Main dashboard rendering logic"""
    render_agency_header()

    # Check if any clients exist
    if not st.session_state.clients:
        render_welcome_screen()
        return

    # Check if active client is selected
    if not st.session_state.active_client:
        st.warning("⚠️ Please select a client from the sidebar to view their dashboard")
        return

    # Get active client and analyzer
    client = st.session_state.clients[st.session_state.active_client]

    if not client.analyzer:
        st.error("❌ No data loaded for this client. Please re-upload their report.")
        return

    analyzer = client.analyzer

    # Create tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "🎯 Keywords",
        "📈 Performance",
        "📝 Client Report",
        "🏢 All Clients",
        "📥 Exports"
    ])

    with tabs[0]:
        render_dashboard_tab(client, analyzer)

    with tabs[1]:
        render_keywords_tab(analyzer)

    with tabs[2]:
        render_performance_tab(analyzer)

    with tabs[3]:
        render_client_report_tab(client, analyzer)

    with tabs[4]:
        render_all_clients_tab()

    with tabs[5]:
        render_exports_tab(analyzer)

def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Render main dashboard
    render_dashboard()

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 1.5rem 0;">
        <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
            {st.session_state.agency_name}
        </p>
        <p style="font-size: 0.9rem; opacity: 0.8; margin: 0;">
            Amazon Ads Agency Dashboard Pro v1.0 | Professional Client Management Platform
        </p>
        <p style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem;">
            🚀 Scale • 💰 Profit • 📊 Analyze • 🏢 Manage
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
