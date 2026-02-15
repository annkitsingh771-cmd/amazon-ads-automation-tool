#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Ads Agency Dashboard Pro v2.0
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

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 1.5rem 1.3rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            min-height: 120px;
        }

        .metric-container {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
            border: 1px solid rgba(148, 163, 184, 0.4);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .success-box {
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.25) 0%, rgba(22, 163, 74, 0.1) 100%);
            border-left: 5px solid #22c55e;
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .warning-box {
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.25) 0%, rgba(234, 179, 8, 0.1) 100%);
            border-left: 5px solid #facc15;
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .danger-box {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(220, 38, 38, 0.1) 100%);
            border-left: 5px solid #ef4444;
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .info-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(59, 130, 246, 0.1) 100%);
            border-left: 5px solid #3b82f6;
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .purple-box {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.25) 0%, rgba(168, 85, 247, 0.1) 100%);
            border-left: 5px solid #a855f7;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return float(value)
    except:
        return default

def safe_int(value, default=0):
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return int(float(value))
    except:
        return default

def safe_str(value, default='N/A'):
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        return str(value).strip()
    except:
        return default

def format_currency(value):
    return f"₹{value:,.2f}"

def format_number(value):
    return f"{value:,.0f}"

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
        self.contact_email = ""
        self.target_acos = None  # Optional now
        self.target_roas = None  # Optional now

# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER CLASS - FIXED WASTAGE + FUTURE OPPORTUNITIES
# ═══════════════════════════════════════════════════════════════════════════

class CompleteAnalyzer:
    REQUIRED_COLUMNS = ['Customer Search Term', 'Campaign Name', 'Spend', 'Clicks']

    # Classification thresholds
    MIN_SPEND_FOR_LOW_POTENTIAL = 50
    MIN_CLICKS_FOR_LOW_POTENTIAL = 10
    MIN_SPEND_FOR_WASTAGE = 100
    MIN_CLICKS_FOR_WASTAGE = 5
    MIN_SPEND_FOR_HIGH_POTENTIAL = 30
    MIN_ORDERS_FOR_HIGH_POTENTIAL = 2
    MIN_CVR_FOR_CHAMPION = 2.0

    # Future opportunity thresholds
    MIN_CLICKS_FOR_FUTURE = 5
    MAX_SPEND_FOR_FUTURE = 200

    def __init__(self, df: pd.DataFrame, client_name: str, target_acos: float = None, target_roas: float = None):
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
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty")

        df = df.copy()
        df.columns = df.columns.str.strip()

        # Column mapping
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

        df.columns = df.columns.str.lower().str.strip()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        df.columns = [col.title() for col in df.columns]

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Add missing columns
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

        # Convert numeric columns
        numeric_cols = ['Spend', 'Sales', 'Clicks', 'Impressions', 'Orders', 'CPC']
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('[₹$,]', '', regex=True)
                    df[col] = df[col].str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df = df[(df['Spend'] > 0) | (df['Clicks'] > 0)].copy()

        if len(df) == 0:
            raise ValueError("No valid data rows found")

        # FIXED: Wastage calculation - only count spend with ZERO sales
        df['Profit'] = df['Sales'] - df['Spend']
        df['Wastage'] = df.apply(lambda x: x['Spend'] if x['Sales'] == 0 else 0, axis=1)

        # Calculate metrics
        df['CVR'] = df.apply(lambda x: (x['Orders'] / x['Clicks'] * 100) if x['Clicks'] > 0 else 0, axis=1)
        df['ROAS'] = df.apply(lambda x: (x['Sales'] / x['Spend']) if x['Spend'] > 0 else 0, axis=1)
        df['ACOS'] = df.apply(lambda x: (x['Spend'] / x['Sales'] * 100) if x['Sales'] > 0 else 0, axis=1)
        df['CTR'] = df.apply(lambda x: (x['Clicks'] / x['Impressions'] * 100) if x['Impressions'] > 0 else 0, axis=1)

        df['Client'] = self.client_name
        df['Processed_Date'] = datetime.now()

        return df

    def get_client_summary(self) -> Dict:
        try:
            if self.df is None or len(self.df) == 0:
                return self._get_empty_summary()

            total_spend = safe_float(self.df['Spend'].sum())
            total_sales = safe_float(self.df['Sales'].sum())
            total_orders = safe_int(self.df['Orders'].sum())
            total_clicks = safe_int(self.df['Clicks'].sum())
            total_impressions = safe_int(self.df['Impressions'].sum())

            # FIXED: Wastage = only spend with zero sales
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
        except Exception:
            return self._get_empty_summary()

    def _get_empty_summary(self) -> Dict:
        return {
            'total_spend': 0, 'total_sales': 0, 'total_profit': 0, 'total_orders': 0,
            'total_clicks': 0, 'total_impressions': 0, 'total_wastage': 0,
            'roas': 0, 'acos': 0, 'avg_cpc': 0, 'avg_ctr': 0, 'avg_cvr': 0,
            'conversion_rate': 0, 'keywords_count': 0, 'campaigns_count': 0
        }

    def get_health_score(self) -> int:
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
        """Enhanced classification with future opportunities"""
        categories = {
            'high_potential': [],
            'low_potential': [],
            'wastage': [],
            'opportunities': [],
            'future_watch': []  # NEW: Keywords to monitor
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
                        'Spend': format_currency(spend),
                        'Sales': format_currency(sales),
                        'ROAS': f"{roas:.2f}x",
                        'Orders': orders,
                        'Clicks': clicks,
                        'CVR': f"{cvr:.2f}%",
                        'Campaign': safe_str(row.get('Campaign Name', 'N/A')),
                        'Match Type': safe_str(row.get('Match Type', 'N/A')),
                        'Reason': ''
                    }

                    # High Potential - SCALE NOW
                    if (roas >= 3.0 and orders >= self.MIN_ORDERS_FOR_HIGH_POTENTIAL and 
                        spend >= self.MIN_SPEND_FOR_HIGH_POTENTIAL and cvr > 0):
                        kw_data['Reason'] = f"Champion! ROAS {roas:.2f}x, {orders} orders"
                        categories['high_potential'].append(kw_data)

                    # Wastage - PAUSE NOW (only if zero sales!)
                    elif (spend >= self.MIN_SPEND_FOR_WASTAGE and sales == 0 and 
                          clicks >= self.MIN_CLICKS_FOR_WASTAGE):
                        kw_data['Reason'] = f"Rs{spend:.0f} spent, {clicks} clicks, ZERO sales"
                        categories['wastage'].append(kw_data)

                    # Low Potential - REDUCE BIDS
                    elif (spend >= self.MIN_SPEND_FOR_LOW_POTENTIAL and 
                          clicks >= self.MIN_CLICKS_FOR_LOW_POTENTIAL and roas < 1.5):
                        kw_data['Reason'] = f"Poor ROAS ({roas:.2f}x), CVR {cvr:.2f}%"
                        categories['low_potential'].append(kw_data)

                    # Opportunities - TEST OPTIMIZATION
                    elif spend >= 20 and roas >= 1.5 and roas < 3.0 and clicks >= 5:
                        kw_data['Reason'] = f"Decent ROAS ({roas:.2f}x), test 10-15% increase"
                        categories['opportunities'].append(kw_data)

                    # NEW: Future Watch - relevant keywords with some data
                    elif (clicks >= self.MIN_CLICKS_FOR_FUTURE and 
                          spend <= self.MAX_SPEND_FOR_FUTURE and 
                          sales == 0):
                        kw_data['Reason'] = f"{clicks} clicks, relevant term, needs more data"
                        categories['future_watch'].append(kw_data)

                except Exception:
                    continue

            return categories
        except Exception:
            return categories

    def get_future_scale_keywords(self) -> List[Dict]:
        """NEW: Identify keywords with potential for future scaling"""
        future_keywords = []

        try:
            if self.df is None or len(self.df) == 0:
                return future_keywords

            for _, row in self.df.iterrows():
                try:
                    spend = safe_float(row.get('Spend', 0))
                    sales = safe_float(row.get('Sales', 0))
                    roas = safe_float(row.get('ROAS', 0))
                    orders = safe_int(row.get('Orders', 0))
                    clicks = safe_int(row.get('Clicks', 0))

                    # Keywords showing promise but need more data
                    if clicks >= 3 and spend < 150:
                        if orders == 1:  # Got 1 order, promising!
                            future_keywords.append({
                                'Keyword': safe_str(row.get('Customer Search Term')),
                                'Match Type': safe_str(row.get('Match Type', 'N/A')),
                                'Clicks': clicks,
                                'Orders': orders,
                                'Spend': format_currency(spend),
                                'Status': '🟡 Promising',
                                'Action': 'Keep monitoring, 1 order already',
                                'Recommendation': 'Continue at current bid, watch for more orders'
                            })
                        elif clicks >= 5 and sales == 0:
                            # Relevant but no conversion yet
                            future_keywords.append({
                                'Keyword': safe_str(row.get('Customer Search Term')),
                                'Match Type': safe_str(row.get('Match Type', 'N/A')),
                                'Clicks': clicks,
                                'Orders': orders,
                                'Spend': format_currency(spend),
                                'Status': '⚪ Watching',
                                'Action': 'Relevant keyword, needs more data',
                                'Recommendation': 'Give it more time, relevant clicks indicate interest'
                            })

                except Exception:
                    continue

            return future_keywords
        except Exception:
            return []

    def get_match_type_strategy(self) -> Dict:
        """NEW: Suggest match type strategy based on data"""
        strategy = {
            'current_performance': {},
            'recommendations': []
        }

        try:
            if self.df is None or 'Match Type' not in self.df.columns:
                return strategy

            # Analyze current match types
            for match_type in ['EXACT', 'PHRASE', 'BROAD']:
                mt_data = self.df[self.df['Match Type'].str.upper() == match_type]
                if len(mt_data) > 0:
                    total_spend = mt_data['Spend'].sum()
                    total_sales = mt_data['Sales'].sum()
                    roas = (total_sales / total_spend) if total_spend > 0 else 0

                    strategy['current_performance'][match_type] = {
                        'spend': total_spend,
                        'sales': total_sales,
                        'roas': roas,
                        'keywords': len(mt_data)
                    }

                    # Recommendations based on performance
                    if match_type == 'EXACT' and roas >= 3.0:
                        strategy['recommendations'].append({
                            'match_type': 'EXACT',
                            'action': '✅ Scale aggressively',
                            'reason': f'High ROAS ({roas:.2f}x) - these are your proven winners',
                            'priority': 'HIGH'
                        })

                    elif match_type == 'PHRASE' and roas >= 2.0:
                        strategy['recommendations'].append({
                            'match_type': 'PHRASE',
                            'action': '⚡ Test & optimize',
                            'reason': f'Good ROAS ({roas:.2f}x) - convert top performers to EXACT',
                            'priority': 'MEDIUM'
                        })

                    elif match_type == 'BROAD' and roas < 1.5:
                        strategy['recommendations'].append({
                            'match_type': 'BROAD',
                            'action': '⚠️ Reduce or pause',
                            'reason': f'Low ROAS ({roas:.2f}x) - too expensive for discovery',
                            'priority': 'HIGH'
                        })

            return strategy
        except Exception:
            return strategy

    def get_roas_improvement_plan(self) -> Dict:
        """NEW: Step-by-step plan to improve ROAS"""
        current_summary = self.get_client_summary()
        current_roas = current_summary['roas']
        classification = self.classify_keywords_improved()

        plan = {
            'current_roas': current_roas,
            'target_roas': self.target_roas or 3.0,
            'gap': (self.target_roas or 3.0) - current_roas,
            'immediate_actions': [],
            'short_term': [],
            'long_term': []
        }

        # Immediate actions (next 24-48 hours)
        wastage_count = len(classification['wastage'])
        if wastage_count > 0:
            wastage_spend = sum(float(k['Spend'].replace('₹','').replace(',','')) 
                              for k in classification['wastage'])
            plan['immediate_actions'].append({
                'priority': '🚨 URGENT',
                'action': f'Pause {wastage_count} wastage keywords',
                'impact': f'Save {format_currency(wastage_spend)}/month',
                'how': 'Exports tab → Download Negatives → Upload to Amazon'
            })

        # Short term (next 7 days)
        high_potential = len(classification['high_potential'])
        if high_potential > 0:
            plan['short_term'].append({
                'priority': '🏆 HIGH',
                'action': f'Scale {high_potential} winning keywords',
                'impact': 'Increase sales by 20-30%',
                'how': 'Bids tab → Increase bids by 15-25% on champions'
            })

        # Add specific recommendations
        if current_roas < 1.0:
            plan['immediate_actions'].insert(0, {
                'priority': '🚨 CRITICAL',
                'action': 'Pause ALL campaigns temporarily',
                'impact': 'Stop losing money',
                'how': 'Losing money on every sale - fix product/pricing first'
            })

        plan['short_term'].append({
            'priority': '⚡ MEDIUM',
            'action': 'Optimize product listings',
            'impact': 'Improve conversion rate by 50-100%',
            'how': 'Better images, A+ content, reviews, competitive pricing'
        })

        plan['long_term'].append({
            'priority': '📊 ONGOING',
            'action': 'Test new keywords from Phrase & Broad',
            'impact': 'Discover new high-performers',
            'how': 'Review search terms weekly, add winners as EXACT match'
        })

        return plan

    def get_bid_suggestions_improved(self) -> List[Dict]:
        """Enhanced bid suggestions"""
        suggestions = []

        try:
            if self.df is None or len(self.df) == 0:
                return suggestions

            # Use smart defaults if targets not set
            target_acos = self.target_acos or 30.0
            target_roas = self.target_roas or 3.0

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
                        'Current CPC': format_currency(current_cpc),
                        'Spend': format_currency(spend),
                        'ROAS': f"{roas:.2f}x",
                        'CVR': f"{cvr:.2f}%",
                        'Orders': orders,
                        'Action': '',
                        'Suggested Bid': '',
                        'Change (%)': 0,
                        'Reason': ''
                    }

                    acos_current = (spend / sales * 100) if sales > 0 else 999

                    # Champion keywords
                    if roas >= 3.5 and cvr >= self.MIN_CVR_FOR_CHAMPION and orders >= 2:
                        new_bid = current_cpc * 1.25
                        suggestion.update({
                            'Action': 'INCREASE',
                            'Suggested Bid': format_currency(new_bid),
                            'Change (%)': 25,
                            'Reason': f"Champion! ROAS {roas:.2f}x, CVR {cvr:.2f}%"
                        })

                    # Above target
                    elif roas >= target_roas and cvr >= 1.0 and orders >= 1:
                        new_bid = current_cpc * 1.15
                        suggestion.update({
                            'Action': 'INCREASE',
                            'Suggested Bid': format_currency(new_bid),
                            'Change (%)': 15,
                            'Reason': f"Above target ROAS ({target_roas:.1f}x)"
                        })

                    # Wastage
                    elif sales == 0 and spend >= self.MIN_SPEND_FOR_WASTAGE:
                        suggestion.update({
                            'Action': 'PAUSE',
                            'Suggested Bid': '₹0.00',
                            'Change (%)': -100,
                            'Reason': f"Rs{spend:.0f} wasted, ZERO sales"
                        })

                    # Poor ROAS
                    elif roas < 1.5 and spend >= 50:
                        new_bid = current_cpc * 0.7
                        suggestion.update({
                            'Action': 'REDUCE',
                            'Suggested Bid': format_currency(new_bid),
                            'Change (%)': -30,
                            'Reason': f"Poor ROAS ({roas:.2f}x)"
                        })

                    # Above target ACOS
                    elif acos_current > target_acos and spend >= 50:
                        reduction = min(30, (acos_current - target_acos) / target_acos * 100)
                        new_bid = current_cpc * (1 - reduction/100)
                        suggestion.update({
                            'Action': 'REDUCE',
                            'Suggested Bid': format_currency(new_bid),
                            'Change (%)': -int(reduction),
                            'Reason': f"ACOS {acos_current:.1f}% > Target {target_acos:.1f}%"
                        })

                    else:
                        continue

                    suggestions.append(suggestion)

                except Exception:
                    continue

            return sorted(suggestions, key=lambda x: float(x['Spend'].replace('₹','').replace(',','')), reverse=True)
        except Exception:
            return []

    def get_match_type_performance(self) -> pd.DataFrame:
        try:
            if self.df is None or len(self.df) == 0 or 'Match Type' not in self.df.columns:
                return pd.DataFrame()

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
        except Exception:
            return pd.DataFrame()

    def generate_client_report(self) -> str:
        try:
            summary = self.get_client_summary()
            health = self.get_health_score()
            classification = self.classify_keywords_improved()

            health_status = "EXCELLENT" if health >= 70 else "GOOD" if health >= 50 else "NEEDS ATTENTION"

            target_acos_str = f"{self.target_acos:.1f}%" if self.target_acos else "Not Set"
            target_roas_str = f"{self.target_roas:.1f}x" if self.target_roas else "Not Set"

            report = f"""
================================================================================
                    AMAZON PPC PERFORMANCE REPORT                            
                        Client: {self.client_name}
                        Date: {datetime.now().strftime('%B %d, %Y')}
================================================================================

EXECUTIVE SUMMARY
================================================================================

Campaign Health Score: {health}/100 - {health_status}
Target ACOS: {target_acos_str} | Target ROAS: {target_roas_str}

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
Budget Wastage:              Rs {summary['total_wastage']:>15,.2f}

KEYWORD PERFORMANCE
--------------------------------------------------------------------------------
High Potential (Scale):      {len(classification['high_potential']):>16}
Opportunities (Test):        {len(classification['opportunities']):>16}
Future Watch:                {len(classification['future_watch']):>16}
Low Potential:               {len(classification['low_potential']):>16}
Wastage (Pause):             {len(classification['wastage']):>16}

RECOMMENDATIONS
--------------------------------------------------------------------------------
"""

            if health >= 70:
                report += "EXCELLENT performance! Continue scaling winners.\n"
            elif health >= 50:
                report += "GOOD performance. Focus on optimization.\n"
            else:
                report += "IMMEDIATE action required - see recommendations below.\n"

            report += f"""

IMMEDIATE ACTION ITEMS
--------------------------------------------------------------------------------
1. PAUSE {len(classification['wastage'])} wastage keywords (save money now)
2. SCALE {len(classification['high_potential'])} high potential keywords (15-25% bid increase)
3. MONITOR {len(classification['future_watch'])} keywords showing promise
4. OPTIMIZE {len(classification['opportunities'])} keywords with potential
5. Review and implement match type strategy

================================================================================
Report Generated By: Amazon Ads Agency Dashboard Pro v2.0 - PERFECT Edition
================================================================================
"""

            return report
        except Exception as e:
            return f"Error generating report: {e}"

# Continue in next part...

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
        <p style="font-size: 0.9rem; opacity: 0.9;">PERFECT Edition - Fixed Wastage + Future Insights</p>
    </div>
    """, unsafe_allow_html=True)

def render_custom_metric(label, value):
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=False):
            new_name = st.text_input("Agency Name", value=st.session_state.agency_name, key='agency_name_input')
            if new_name != st.session_state.agency_name:
                st.session_state.agency_name = new_name
                st.rerun()

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
                        target_acos = f"{client.target_acos:.1f}%" if client.target_acos else "Not Set"
                        target_roas = f"{client.target_roas:.1f}x" if client.target_roas else "Not Set"
                        st.info(f"{emoji} Health: {health}/100\nTarget ACOS: {target_acos}\nTarget ROAS: {target_roas}")
                    except:
                        pass

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

            st.markdown("**🎯 Performance Goals (Optional):**")
            st.info("Leave at 0 to skip targets - dashboard will use smart defaults")

            col1, col2 = st.columns(2)
            with col1:
                target_acos = st.number_input("Target ACOS (%)", value=0.0, step=5.0, 
                                             help="0 = No target (smart defaults)", key='new_acos')
            with col2:
                target_roas = st.number_input("Target ROAS (x)", value=0.0, step=0.5,
                                             help="0 = No target (smart defaults)", key='new_roas')

            email = st.text_input("Contact Email (Optional)", placeholder="client@company.com", key='new_email')

            st.info("📄 Upload Amazon Search Term Report")
            uploaded_file = st.file_uploader("", type=["xlsx", "xls"], key='new_file')

            if st.button("✅ Add Client", type="primary", use_container_width=True):
                if not name:
                    st.error("❌ Please enter client name")
                elif not uploaded_file:
                    st.error("❌ Please upload Search Term Report")
                else:
                    try:
                        with st.spinner(f"Analyzing {name}'s data..."):
                            df = pd.read_excel(uploaded_file)
                            st.info(f"Found {len(df)} rows and {len(df.columns)} columns")

                            client_data = ClientData(name, industry, budget)
                            client_data.contact_email = email

                            # Handle optional targets
                            client_data.target_acos = target_acos if target_acos > 0 else None
                            client_data.target_roas = target_roas if target_roas > 0 else None

                            client_data.analyzer = CompleteAnalyzer(
                                df, name, 
                                target_acos if target_acos > 0 else None,
                                target_roas if target_roas > 0 else None
                            )

                            st.session_state.clients[name] = client_data
                            st.session_state.active_client = name

                            st.success(f"✅ Successfully added {name}!")
                            st.balloons()
                            st.rerun()

                    except ValueError as e:
                        st.error(f"❌ Data Validation Error: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("🔍 See detailed error"):
                            st.code(traceback.format_exc())

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

        target_acos_display = f"{client.target_acos:.1f}%" if client.target_acos else "Smart Defaults (30%)"
        target_roas_display = f"{client.target_roas:.1f}x" if client.target_roas else "Smart Defaults (3.0x)"

        st.markdown(f"""
        <div class="info-box">
            <h2 style="margin:0;">Health Score: {health}/100</h2>
            <p style="margin:0.5rem 0 0 0;">Target ACOS: {target_acos_display} | Target ROAS: {target_roas_display}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💰 Financial Performance")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_custom_metric("Total Spend", format_currency(summary['total_spend']))
        with col2:
            render_custom_metric("Total Sales", format_currency(summary['total_sales']))
        with col3:
            render_custom_metric("ROAS", f"{summary['roas']:.2f}x")
        with col4:
            render_custom_metric("Orders", format_number(summary['total_orders']))
        with col5:
            render_custom_metric("Profit/Loss", format_currency(summary['total_profit']))

        st.markdown("---")
        st.subheader("📈 Campaign Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_custom_metric("CVR", f"{summary['avg_cvr']:.2f}%")
        with col2:
            render_custom_metric("ACOS", f"{summary['acos']:.1f}%")
        with col3:
            render_custom_metric("Clicks", format_number(summary['total_clicks']))
        with col4:
            # FIXED: Wastage calculation
            wastage_pct = (summary['total_wastage'] / summary['total_spend'] * 100) if summary['total_spend'] > 0 else 0
            render_custom_metric("Wastage", f"{format_currency(summary['total_wastage'])} ({wastage_pct:.1f}%)")

        # ROAS Improvement Plan
        st.markdown("---")
        st.subheader("🎯 ROAS Improvement Plan")

        improvement_plan = analyzer.get_roas_improvement_plan()

        st.markdown(f"""
        <div class="purple-box">
            <strong>📊 Current ROAS: {improvement_plan['current_roas']:.2f}x</strong><br>
            <strong>🎯 Target ROAS: {improvement_plan['target_roas']:.2f}x</strong><br>
            <strong>📈 Gap to Close: {improvement_plan['gap']:.2f}x</strong>
        </div>
        """, unsafe_allow_html=True)

        if improvement_plan['immediate_actions']:
            st.markdown("#### 🚨 IMMEDIATE ACTIONS (Next 24-48 hours)")
            for action in improvement_plan['immediate_actions']:
                st.markdown(f"""
                <div class="danger-box">
                    <strong>{action['priority']}: {action['action']}</strong><br>
                    💰 Impact: {action['impact']}<br>
                    📋 How: {action['how']}
                </div>
                """, unsafe_allow_html=True)

        if improvement_plan['short_term']:
            st.markdown("#### ⚡ SHORT TERM (Next 7 days)")
            for action in improvement_plan['short_term']:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>{action['priority']}: {action['action']}</strong><br>
                    💰 Impact: {action['impact']}<br>
                    📋 How: {action['how']}
                </div>
                """, unsafe_allow_html=True)

        if improvement_plan['long_term']:
            st.markdown("#### 📊 LONG TERM (Ongoing)")
            for action in improvement_plan['long_term']:
                st.markdown(f"""
                <div class="info-box">
                    <strong>{action['priority']}: {action['action']}</strong><br>
                    💰 Impact: {action['impact']}<br>
                    📋 How: {action['how']}
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")

def render_keywords_tab(analyzer):
    try:
        st.header("🎯 Keywords Analysis")

        classification = analyzer.classify_keywords_improved()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🏆 High Potential", len(classification['high_potential']))
        with col2:
            st.metric("⚡ Opportunities", len(classification['opportunities']))
        with col3:
            st.metric("👀 Future Watch", len(classification['future_watch']))
        with col4:
            st.metric("⚠️ Low Potential", len(classification['low_potential']))
        with col5:
            st.metric("🚨 Wastage", len(classification['wastage']))

        st.markdown("---")

        tabs = st.tabs([
            f"🏆 Scale Now ({len(classification['high_potential'])})",
            f"⚡ Test ({len(classification['opportunities'])})",
            f"👀 Future Watch ({len(classification['future_watch'])})",
            f"⚠️ Low ({len(classification['low_potential'])})",
            f"🚨 Pause ({len(classification['wastage'])})"
        ])

        with tabs[0]:
            if classification['high_potential']:
                st.success("✅ SCALE THESE NOW! Increase bids by 15-25%")
                st.dataframe(pd.DataFrame(classification['high_potential']), 
                           use_container_width=True, hide_index=True, height=450)
            else:
                st.info("No champions yet. Need keywords with ROAS ≥3.0x AND ≥2 orders")

        with tabs[1]:
            if classification['opportunities']:
                st.info("⚡ Test 10-15% bid increases")
                st.dataframe(pd.DataFrame(classification['opportunities']), 
                           use_container_width=True, hide_index=True, height=450)
            else:
                st.info("No opportunities at this time")

        with tabs[2]:
            if classification['future_watch']:
                st.markdown("""
                <div class="purple-box">
                    <strong>👀 Future Watch Keywords</strong><br>
                    These have clicks and are relevant, but need more data.<br>
                    <strong>Action:</strong> Keep running at current bids, monitor for conversions
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(classification['future_watch']), 
                           use_container_width=True, hide_index=True, height=450)

                # Show future scaling potential
                future_scale = analyzer.get_future_scale_keywords()
                if future_scale:
                    st.markdown("---")
                    st.markdown("### 🔮 Keywords to Scale in Future")
                    st.dataframe(pd.DataFrame(future_scale), 
                               use_container_width=True, hide_index=True, height=300)
            else:
                st.info("No keywords under observation")

        with tabs[3]:
            if classification['low_potential']:
                st.warning("⚠️ Reduce bids by 30% or pause")
                st.dataframe(pd.DataFrame(classification['low_potential']), 
                           use_container_width=True, hide_index=True, height=450)
            else:
                st.success("✅ No low performers")

        with tabs[4]:
            if classification['wastage']:
                total_wasted = sum(float(k['Spend'].replace('₹','').replace(',','')) 
                                 for k in classification['wastage'])
                st.error(f"🚨 FIXED: {format_currency(total_wasted)} wasted on ZERO sales keywords")
                st.markdown("""
                <div class="danger-box">
                    <strong>Note:</strong> Wastage = Spend on keywords with ZERO sales<br>
                    <strong>Action:</strong> Exports tab → Download Negatives → Upload to Amazon
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(classification['wastage']), 
                           use_container_width=True, hide_index=True, height=450)
            else:
                st.success("🎉 No wastage!")

    except Exception as e:
        st.error(f"Error analyzing keywords: {e}")

def render_match_type_tab(analyzer):
    try:
        st.header("📊 Match Type Strategy")

        # Performance comparison
        match_perf = analyzer.get_match_type_performance()

        if not match_perf.empty:
            st.subheader("Current Performance")

            display_df = match_perf.copy()
            display_df['Spend'] = display_df['Spend'].apply(lambda x: format_currency(x))
            display_df['Sales'] = display_df['Sales'].apply(lambda x: format_currency(x))
            display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.2f}x")
            display_df['ACOS'] = display_df['ACOS'].apply(lambda x: f"{x:.1f}%")
            display_df['CVR'] = display_df['CVR'].apply(lambda x: f"{x:.2f}%")
            display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.2f}%")

            st.dataframe(display_df, use_container_width=True)

        # Strategy recommendations
        st.markdown("---")
        st.subheader("🎯 Match Type Strategy Guide")

        strategy = analyzer.get_match_type_strategy()

        if strategy.get('recommendations'):
            for rec in strategy['recommendations']:
                priority_class = "danger-box" if rec['priority'] == 'HIGH' else "warning-box" if rec['priority'] == 'MEDIUM' else "info-box"
                st.markdown(f"""
                <div class="{priority_class}">
                    <strong>{rec['match_type']} Match:</strong> {rec['action']}<br>
                    <strong>Why:</strong> {rec['reason']}<br>
                    <strong>Priority:</strong> {rec['priority']}
                </div>
                """, unsafe_allow_html=True)

        # General strategy guide
        st.markdown("---")
        st.markdown("### 📚 When to Use Each Match Type")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>🎯 EXACT Match</h4>
                <strong>When to use:</strong><br>
                • Proven winners (ROAS ≥3.0x)<br>
                • High conversion keywords<br>
                • Maximum control<br><br>
                <strong>Bid Strategy:</strong><br>
                Aggressive - scale winners<br><br>
                <strong>Example:</strong><br>
                "blue water bottle 1 litre"
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>📝 PHRASE Match</h4>
                <strong>When to use:</strong><br>
                • Discovery mode<br>
                • Related variations<br>
                • Balance control & reach<br><br>
                <strong>Bid Strategy:</strong><br>
                Moderate - test and optimize<br><br>
                <strong>Example:</strong><br>
                "water bottle" matches<br>
                "best water bottle"
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>🌐 BROAD Match</h4>
                <strong>When to use:</strong><br>
                • Research only<br>
                • Find new keywords<br>
                • Low budget tests<br><br>
                <strong>Bid Strategy:</strong><br>
                Conservative - low bids<br><br>
                <strong>Example:</strong><br>
                "bottle" matches anything<br>
                with "bottle"
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔄 Optimization Workflow")

        st.markdown("""
        <div class="purple-box">
            <strong>Step-by-Step Match Type Optimization:</strong><br><br>
            <strong>1. START with PHRASE match</strong> (moderate bids)<br>
            → Discover which variations work<br><br>
            <strong>2. ANALYZE search terms weekly</strong><br>
            → Look for exact terms with sales<br><br>
            <strong>3. CONVERT winners to EXACT</strong> (higher bids)<br>
            → Move proven keywords to exact match<br><br>
            <strong>4. ADD losers as NEGATIVES</strong><br>
            → Block irrelevant terms<br><br>
            <strong>5. REPEAT weekly</strong><br>
            → Continuous optimization cycle
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# Continue in next part...

def render_bid_optimization_tab(analyzer):
    try:
        st.header("💡 Bid Optimization")

        target_acos_display = f"{analyzer.target_acos:.1f}%" if analyzer.target_acos else "30% (default)"
        target_roas_display = f"{analyzer.target_roas:.1f}x" if analyzer.target_roas else "3.0x (default)"

        st.markdown(f"""
        <div class="info-box">
            <strong>🎯 Optimization Targets</strong><br>
            Target ACOS: {target_acos_display} | Target ROAS: {target_roas_display}
        </div>
        """, unsafe_allow_html=True)

        suggestions = analyzer.get_bid_suggestions_improved()

        if suggestions:
            col1, col2 = st.columns(2)
            with col1:
                action_filter = st.selectbox("Filter", ["All", "INCREASE", "REDUCE", "PAUSE"])

            filtered = suggestions if action_filter == "All" else [
                s for s in suggestions if action_filter in s['Action']
            ]

            increase_count = len([s for s in suggestions if 'INCREASE' in s['Action']])
            reduce_count = len([s for s in suggestions if 'REDUCE' in s['Action']])
            pause_count = len([s for s in suggestions if 'PAUSE' in s['Action']])

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⬆️ Scale Up", increase_count)
            with col2:
                st.metric("⬇️ Reduce", reduce_count)
            with col3:
                st.metric("⏸️ Pause", pause_count)

            st.markdown("---")
            st.markdown(f"**Showing {len(filtered)} of {len(suggestions)}**")
            st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("No suggestions - need more data or all optimized")

    except Exception as e:
        st.error(f"Error: {e}")

def render_exports_tab(analyzer, client_name):
    try:
        st.header("📥 Export Files")

        classification = analyzer.classify_keywords_improved()
        suggestions = analyzer.get_bid_suggestions_improved()

        col1, col2, col3 = st.columns(3)

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
                    f"📥 Download ({len(neg_data)})",
                    data=output,
                    file_name=f"Negatives_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.error(f"🚨 {len(neg_data)} to pause")
            else:
                st.success("✅ No negatives needed")

        with col2:
            st.subheader("💰 Bid Adjustments")

            if suggestions:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    pd.DataFrame(suggestions).to_excel(writer, index=False)
                output.seek(0)

                st.download_button(
                    f"📥 Download ({len(suggestions)})",
                    data=output,
                    file_name=f"Bids_{client_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success(f"✅ {len(suggestions)} suggestions")
            else:
                st.info("No adjustments")

        with col3:
            st.subheader("📊 Complete Data")

            if analyzer.df is not None:
                csv_data = analyzer.df.to_csv(index=False)

                st.download_button(
                    f"📥 CSV ({len(analyzer.df)} rows)",
                    data=csv_data,
                    file_name=f"Full_Data_{client_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Full dataset")

    except Exception as e:
        st.error(f"Error: {e}")

def render_report_tab(client, analyzer):
    try:
        st.header("📝 Client Report")

        report = analyzer.generate_client_report()
        st.text_area("Report", report, height=600)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📄 Download TXT",
                data=report,
                file_name=f"Report_{client.name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:
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
            output.seek(0)

            st.download_button(
                "📊 Download Excel",
                data=output,
                file_name=f"Report_{client.name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {e}")

def render_all_clients_tab():
    try:
        st.header("👥 All Clients")

        if not st.session_state.clients:
            st.info("No clients yet")
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
                        'Spend': format_currency(summary['total_spend']),
                        'Sales': format_currency(summary['total_sales']),
                        'ROAS': f"{summary['roas']:.2f}x",
                        'ACOS': f"{summary['acos']:.1f}%",
                        'Status': '🟢' if health >= 70 else '🟡' if health >= 50 else '🔴'
                    })
                except:
                    continue

        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("📊 Agency Overview")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Clients", len(data))
            with col2:
                healthy = len([d for d in data if '🟢' in d['Status']])
                st.metric("Healthy", healthy)
            with col3:
                attention = len([d for d in data if '🔴' in d['Status']])
                st.metric("Need Attention", attention)

    except Exception as e:
        st.error(f"Error: {e}")

def render_dashboard():
    render_agency_header()

    if not st.session_state.clients:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to Amazon Ads Dashboard Pro v2.0 - PERFECT Edition!</h3>
            <br>
            <strong>✨ What's Fixed & New:</strong>
            <ul>
                <li>✅ FIXED: Wastage calculation (only zero-sales keywords)</li>
                <li>✅ NEW: Optional targets (leave at 0 for smart defaults)</li>
                <li>✅ NEW: Future watch keywords (promising keywords to monitor)</li>
                <li>✅ NEW: Match type strategy guide (when to use Broad/Phrase/Exact)</li>
                <li>✅ NEW: ROAS improvement roadmap (step-by-step)</li>
                <li>✅ NEW: Keywords to scale in future</li>
            </ul>
            <br>
            <strong>👈 Get started by adding a client from the sidebar!</strong>
        </div>
        """, unsafe_allow_html=True)
        return

    if not st.session_state.active_client:
        st.warning("⚠️ Select a client")
        return

    client = st.session_state.clients[st.session_state.active_client]

    if not client.analyzer or client.analyzer.df is None:
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

def main():
    load_custom_css()
    init_session_state()
    render_sidebar()
    render_dashboard()

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 1rem;">
        <strong>{st.session_state.agency_name}</strong><br>
        Amazon Ads Agency Dashboard Pro v2.0 - PERFECT Edition<br>
        <small>Fixed Wastage | Optional Targets | Future Insights | Match Type Strategy</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
