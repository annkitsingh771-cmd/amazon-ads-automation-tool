#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Ads Agency Dashboard Pro v6.0 

"""

import io, traceback, copy, re
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Amazon Ads Dashboard Pro",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    css = """
    <style>
    .main { padding-top: 0.5rem; }
    .agency-header {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 1.4rem;
        color: #fff;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.2;
    }
    .success-box { background: rgba(22, 163, 74, 0.2); border-left: 4px solid #22c55e; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .warning-box { background: rgba(234, 179, 8, 0.2); border-left: 4px solid #facc15; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .danger-box { background: rgba(220, 38, 38, 0.2); border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .info-box { background: rgba(59, 130, 246, 0.2); border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .purple-box { background: rgba(168, 85, 247, 0.2); border-left: 4px solid #a855f7; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .cyan-box { background: rgba(6, 182, 212, 0.2); border-left: 4px solid #06b6d4; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        val_str = str(value).replace('₹', '').replace('$', '').replace(',', '').replace('%', '').strip()
        return float(val_str) if val_str else default
    except:
        return default

def safe_int(value, default=0):
    try:
        if pd.isna(value) or value == '' or value is None:
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return int(float(str(value).replace(',', '').replace('₹', '').replace('$', '').strip()))
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
    try:
        val = safe_float(value, 0)
        if val >= 10000000:
            return f"₹{val/10000000:.2f}Cr"
        elif val >= 100000:
            return f"₹{val/100000:.2f}L"
        else:
            return f"₹{val:,.2f}"
    except:
        return "₹0.00"

def format_number(value):
    try:
        val = safe_int(value, 0)
        if val >= 10000000:
            return f"{val/10000000:.2f}Cr"
        elif val >= 100000:
            return f"{val/100000:.2f}L"
        elif val >= 1000:
            return f"{val:,}"
        else:
            return str(val)
    except:
        return "0"

def is_asin(value):
    """Check if value is an Amazon ASIN (starts with B0, etc.)"""
    if not value or pd.isna(value):
        return False
    val_str = str(value).strip().upper()
    # ASIN pattern: B followed by alphanumeric, typically 10 chars
    return bool(re.match(r'^[B][0-9A-Z]{9,}$', val_str))

def get_negative_type(value):
    """Determine if negative is keyword or product ASIN"""
    if is_asin(value):
        return 'PRODUCT'
    return 'KEYWORD'

class ClientData:
    def __init__(self, name, industry="E-commerce", budget=50000):
        self.name = name
        self.industry = industry
        self.monthly_budget = budget
        self.analyzer = None
        self.added_date = datetime.now()
        self.contact_email = ""
        self.target_acos = None
        self.target_roas = None
        self.target_cpa = None
        self.target_tcoas = None

class CompleteAnalyzer:
    REQUIRED_COLUMNS = ['Customer Search Term', 'Campaign Name', 'Spend', 'Clicks']

    def __init__(self, df, client_name, target_acos=None, target_roas=None, target_cpa=None, target_tcoas=None):
        self.client_name = client_name
        self.target_acos = target_acos
        self.target_roas = target_roas
        self.target_cpa = target_cpa
        self.target_tcoas = target_tcoas
        self.df = None
        self.raw_df = None
        self.error = None
        self.column_mapping = {}
        
        try:
            self.raw_df = df.copy(deep=True)
            self.df = self._validate_and_prepare_data(df.copy(deep=True))
        except Exception as e:
            self.error = str(e)
            raise ValueError(f"Validation failed: {e}")

    def _validate_and_prepare_data(self, df):
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame")

        original_columns = list(df.columns)
        
        # Clean column names
        df.columns = df.columns.str.strip()

        # EXTENDED column mapping
        mapping = {
            'customer search term': 'Customer Search Term',
            'search term': 'Customer Search Term',
            'keyword': 'Customer Search Term',
            'searchterm': 'Customer Search Term',
            'customer_search_term': 'Customer Search Term',
            'search terms': 'Customer Search Term',
            
            'campaign': 'Campaign Name',
            'campaign name': 'Campaign Name',
            'campaign_name': 'Campaign Name',
            
            'ad group': 'Ad Group Name',
            'ad group name': 'Ad Group Name',
            'adgroup': 'Ad Group Name',
            'ad_group_name': 'Ad Group Name',
            
            'match type': 'Match Type',
            'matchtype': 'Match Type',
            'match_type': 'Match Type',
            
            # SALES
            '7 day total sales': 'Sales',
            '7 day total sales (₹)': 'Sales',
            '7 day total sales ($)': 'Sales',
            '7 day sales': 'Sales',
            '7 day total revenue': 'Sales',
            'total sales': 'Sales',
            'sales': 'Sales',
            'revenue': 'Sales',
            '7 day total sales ': 'Sales',
            '7 Day Total Sales': 'Sales',
            'sales (₹)': 'Sales',
            'sales ($)': 'Sales',
            
            # ORDERS
            '7 day total orders': 'Orders',
            '7 day total orders (#)': 'Orders',
            '7 day orders': 'Orders',
            '7 day total units': 'Orders',
            'total orders': 'Orders',
            'orders': 'Orders',
            'units': 'Orders',
            '7 day ordered units': 'Orders',
            '7 Day Total Orders': 'Orders',
            
            # Spend/Cost
            'cost': 'Spend',
            'spend': 'Spend',
            'ad spend': 'Spend',
            'spend (₹)': 'Spend',
            'spend ($)': 'Spend',
            
            # Impressions
            'impressions': 'Impressions',
            'imps': 'Impressions',
            
            # Clicks
            'clicks': 'Clicks',
            
            # CPC
            'cpc': 'CPC',
            'cost per click': 'CPC',
            'avg cpc': 'CPC',
            'average cpc': 'CPC',
        }

        df.columns = df.columns.str.lower().str.strip()
        
        for old, new in mapping.items():
            old_clean = old.lower().strip()
            if old_clean in df.columns:
                df.rename(columns={old_clean: new}, inplace=True)
                self.column_mapping[old] = new

        df.columns = [c.title() for c in df.columns]

        # Check required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            available = list(df.columns)
            raise ValueError(f"Missing required columns: {missing}. Available: {available}")

        # Add optional columns
        optional_numeric = ['Sales', 'Orders', 'Impressions', 'Cpc']
        optional_text = ['Ad Group Name', 'Match Type']
        
        for col in optional_numeric:
            if col not in df.columns:
                df[col] = 0
                
        for col in optional_text:
            if col not in df.columns:
                df[col] = 'N/A'

        # Ensure CPC column exists
        if 'Cpc' in df.columns and 'CPC' not in df.columns:
            df['CPC'] = df['Cpc']

        # Convert numeric columns
        numeric_cols = ['Spend', 'Sales', 'Clicks', 'Impressions', 'Orders', 'CPC']
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('₹', '').str.replace('$', '').str.replace(',', '').str.replace('%', '').str.replace('(', '-').str.replace(')', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # FIXED: Calculate CPC from Spend/Clicks if CPC is 0 or missing
        df['CPC_Calculated'] = df.apply(
            lambda x: safe_float(x.get('Spend', 0)) / safe_float(x.get('Clicks', 1)) 
            if safe_float(x.get('Clicks', 0)) > 0 else 0, 
            axis=1
        )
        
        # Use calculated CPC if original CPC is 0
        df['CPC'] = df.apply(
            lambda x: safe_float(x.get('CPC', 0)) if safe_float(x.get('CPC', 0)) > 0 else safe_float(x.get('CPC_Calculated', 0)),
            axis=1
        )

        # Filter rows with activity
        df = df[(df['Spend'] > 0) | (df['Clicks'] > 0)].copy()

        if len(df) == 0:
            raise ValueError("No valid data after filtering")

        # Calculate derived metrics
        df['Profit'] = df['Sales'] - df['Spend']
        df['Wastage'] = df.apply(lambda x: safe_float(x.get('Spend', 0)) if safe_float(x.get('Sales', 0)) == 0 else 0, axis=1)
        df['CVR'] = df.apply(lambda x: (safe_float(x.get('Orders', 0)) / safe_float(x.get('Clicks', 1)) * 100) if safe_float(x.get('Clicks', 0)) > 0 else 0, axis=1)
        df['ROAS'] = df.apply(lambda x: (safe_float(x.get('Sales', 0)) / safe_float(x.get('Spend', 1))) if safe_float(x.get('Spend', 0)) > 0 else 0, axis=1)
        df['ACOS'] = df.apply(lambda x: (safe_float(x.get('Spend', 0)) / safe_float(x.get('Sales', 1)) * 100) if safe_float(x.get('Sales', 0)) > 0 else 0, axis=1)
        df['CTR'] = df.apply(lambda x: (safe_float(x.get('Clicks', 0)) / safe_float(x.get('Impressions', 1)) * 100) if safe_float(x.get('Impressions', 0)) > 0 else 0, axis=1)
        df['CPA'] = df.apply(lambda x: (safe_float(x.get('Spend', 0)) / safe_float(x.get('Orders', 1))) if safe_float(x.get('Orders', 0)) > 0 else 0, axis=1)
        df['TCoAS'] = df['ACOS']
        
        # Add negative type classification
        df['Negative_Type'] = df['Customer Search Term'].apply(get_negative_type)
        
        df['Client'] = self.client_name
        df['Processed_Date'] = datetime.now()

        return df

    def get_client_summary(self):
        try:
            if self.df is None or len(self.df) == 0:
                return self._empty_summary()

            ts = safe_float(self.df['Spend'].sum())
            tsa = safe_float(self.df['Sales'].sum())
            to = safe_int(self.df['Orders'].sum())
            tc = safe_int(self.df['Clicks'].sum())
            ti = safe_int(self.df['Impressions'].sum())
            tw = safe_float(self.df['Wastage'].sum())
            tp = safe_float(self.df['Profit'].sum())

            # FIXED: Calculate avg CPC properly
            avg_cpc = 0
            if 'CPC' in self.df.columns:
                # Filter out 0 CPC values for average calculation
                cpc_values = self.df[self.df['CPC'] > 0]['CPC']
                if len(cpc_values) > 0:
                    avg_cpc = safe_float(cpc_values.mean())
            
            # Fallback: calculate from totals
            if avg_cpc == 0 and tc > 0:
                avg_cpc = ts / tc

            avg_ctr = safe_float(self.df['CTR'].mean())
            avg_cvr = safe_float(self.df['CVR'].mean())
            avg_cpa = (ts / to) if to > 0 else 0
            avg_acos = (ts / tsa * 100) if tsa > 0 else 0
            avg_roas = (tsa / ts) if ts > 0 else 0
            avg_tcoas = avg_acos

            return {
                'total_spend': ts,
                'total_sales': tsa,
                'total_profit': tp,
                'total_orders': to,
                'total_clicks': tc,
                'total_impressions': ti,
                'total_wastage': tw,
                'roas': avg_roas,
                'acos': avg_acos,
                'tcoas': avg_tcoas,
                'avg_cpc': avg_cpc,
                'avg_ctr': avg_ctr,
                'avg_cvr': avg_cvr,
                'avg_cpa': avg_cpa,
                'conversion_rate': (to / tc * 100) if tc > 0 else 0,
                'keywords_count': len(self.df),
                'campaigns_count': safe_int(self.df['Campaign Name'].nunique()),
                'ad_groups_count': safe_int(self.df['Ad Group Name'].nunique()) if 'Ad Group Name' in self.df.columns else 0,
            }
        except Exception as e:
            st.error(f"Summary error: {e}")
            return self._empty_summary()

    def _empty_summary(self):
        return {k: 0 for k in ['total_spend', 'total_sales', 'total_profit', 'total_orders',
                                'total_clicks', 'total_impressions', 'total_wastage', 'roas',
                                'acos', 'tcoas', 'avg_cpc', 'avg_ctr', 'avg_cvr', 'avg_cpa',
                                'conversion_rate', 'keywords_count', 'campaigns_count', 'ad_groups_count']}

    def get_health_score(self):
        try:
            s = self.get_client_summary()
            score = 0
            
            r = s['roas']
            if r >= 4.0: score += 40
            elif r >= 3.0: score += 35
            elif r >= 2.5: score += 30
            elif r >= 2.0: score += 25
            elif r >= 1.5: score += 15
            elif r > 0: score += 5

            wp = (s['total_wastage'] / s['total_spend'] * 100) if s['total_spend'] > 0 else 0
            if wp <= 5: score += 25
            elif wp <= 15: score += 20
            elif wp <= 25: score += 15
            elif wp <= 35: score += 10
            else: score += 5

            ctr = s['avg_ctr']
            if ctr >= 5: score += 20
            elif ctr >= 3: score += 15
            elif ctr >= 1.5: score += 10
            elif ctr >= 0.5: score += 5

            cvr = s['avg_cvr']
            if cvr >= 10: score += 15
            elif cvr >= 5: score += 10
            elif cvr >= 2: score += 5

            return min(score, 100)
        except:
            return 0

    def get_performance_insights(self):
        summary = self.get_client_summary()
        insights = {
            'ctr_insights': [], 'cvr_insights': [], 'roas_insights': [],
            'acos_insights': [], 'cpa_insights': [], 'tcoas_insights': [],
            'content_suggestions': [], 'action_items': []
        }

        try:
            avg_ctr = summary['avg_ctr']
            if avg_ctr < 0.3:
                insights['ctr_insights'].append({
                    'level': 'critical', 'metric': 'CTR', 'value': f'{avg_ctr:.2f}%',
                    'issue': 'CRITICAL: Extremely low CTR - ads not resonating',
                    'action': 'URGENT: Complete ad overhaul needed'
                })
                insights['content_suggestions'].extend([
                    '🎯 Add power words: "Best", "Top-Rated", "Premium"',
                    '💰 Show pricing: "50% Off", "Under ₹999", "Free Shipping"',
                    '⭐ Add badges: "4.5★ Rated", "10K+ Sold"',
                    '🎁 Use urgency: "Limited Time", "Today Only"',
                    '📸 Use high-res lifestyle images with infographics'
                ])
            elif avg_ctr < 0.8:
                insights['ctr_insights'].append({
                    'level': 'warning', 'metric': 'CTR', 'value': f'{avg_ctr:.2f}%',
                    'issue': 'Low CTR - below industry average (1-2%)',
                    'action': 'Improve ad copy and test new creatives'
                })
            elif avg_ctr >= 2.0:
                insights['ctr_insights'].append({
                    'level': 'success', 'metric': 'CTR', 'value': f'{avg_ctr:.2f}%',
                    'issue': 'Excellent CTR! Above industry average',
                    'action': 'Keep current strategy - document what works'
                })

            avg_cvr = summary['avg_cvr']
            if avg_cvr < 1.0:
                insights['cvr_insights'].append({
                    'level': 'critical', 'metric': 'CVR', 'value': f'{avg_cvr:.2f}%',
                    'issue': 'Poor conversion - traffic not converting',
                    'action': 'Fix product page, pricing, or targeting'
                })
            elif avg_cvr < 3.0:
                insights['cvr_insights'].append({
                    'level': 'warning', 'metric': 'CVR', 'value': f'{avg_cvr:.2f}%',
                    'issue': 'Below average conversion',
                    'action': 'Optimize product pages and test pricing'
                })

            roas = summary['roas']
            if roas < 1.0:
                insights['roas_insights'].append({
                    'level': 'critical', 'metric': 'ROAS', 'value': f'{roas:.2f}x',
                    'issue': 'LOSING MONEY - spending more than earning',
                    'action': 'PAUSE campaigns immediately and investigate'
                })
            elif roas < 2.0:
                insights['roas_insights'].append({
                    'level': 'warning', 'metric': 'ROAS', 'value': f'{roas:.2f}x',
                    'issue': 'Low profitability',
                    'action': 'Reduce bids on poor performers, optimize targeting'
                })
            elif roas >= 3.0:
                insights['roas_insights'].append({
                    'level': 'success', 'metric': 'ROAS', 'value': f'{roas:.2f}x',
                    'issue': 'Excellent ROAS! Profitable campaigns',
                    'action': 'Scale winning campaigns aggressively'
                })

            acos = summary['acos']
            target_acos = self.target_acos or 30
            if acos > target_acos * 1.5:
                insights['acos_insights'].append({
                    'level': 'critical', 'metric': 'ACOS', 'value': f'{acos:.1f}%',
                    'issue': f'Way above target ({target_acos:.1f}%)',
                    'action': 'Reduce bids immediately, add negatives'
                })
            elif acos > target_acos:
                insights['acos_insights'].append({
                    'level': 'warning', 'metric': 'ACOS', 'value': f'{acos:.1f}%',
                    'issue': f'Above target ({target_acos:.1f}%)',
                    'action': 'Optimize bids and targeting'
                })

            avg_cpa = summary['avg_cpa']
            if self.target_cpa and avg_cpa > self.target_cpa:
                insights['cpa_insights'].append({
                    'level': 'warning', 'metric': 'CPA', 'value': format_currency(avg_cpa),
                    'issue': f'Above target {format_currency(self.target_cpa)}',
                    'action': 'Reduce bids or improve conversion rate'
                })

            return insights
        except Exception as e:
            st.error(f"Insights error: {e}")
            return insights

    def classify_keywords_improved(self):
        cats = {
            'high_potential': [], 'low_potential': [], 'wastage': [],
            'opportunities': [], 'future_watch': []
        }

        try:
            if self.df is None or len(self.df) == 0:
                return cats

            for _, r in self.df.iterrows():
                try:
                    sp = safe_float(r.get('Spend', 0))
                    sa = safe_float(r.get('Sales', 0))
                    ro = safe_float(r.get('ROAS', 0))
                    o = safe_int(r.get('Orders', 0))
                    c = safe_int(r.get('Clicks', 0))
                    cv = safe_float(r.get('CVR', 0))
                    kw = safe_str(r.get('Customer Search Term'))
                    camp = safe_str(r.get('Campaign Name'))
                    mt = safe_str(r.get('Match Type'))
                    cpc = safe_float(r.get('CPC', 0))

                    if sp <= 0 and c <= 0:
                        continue

                    kd = {
                        'Keyword': kw,
                        'Spend': format_currency(sp),
                        'Sales': format_currency(sa),
                        'ROAS': f"{ro:.2f}x",
                        'Orders': o,
                        'Clicks': c,
                        'CVR': f"{cv:.2f}%",
                        'CPC': format_currency(cpc),
                        'Campaign': camp,
                        'Match Type': mt,
                        'Reason': ''
                    }

                    if ro >= 2.5 and o >= 1 and sp >= 20:
                        kd['Reason'] = f"Champion! ROAS {ro:.2f}x, {o} orders"
                        cats['high_potential'].append(kd)
                    elif sp >= 50 and sa == 0 and c >= 3:
                        kd['Reason'] = f"₹{sp:.0f} spent, ZERO sales - PAUSE"
                        cats['wastage'].append(kd)
                    elif sp >= 30 and ro < 1.0 and c >= 5:
                        kd['Reason'] = f"Poor ROAS {ro:.2f}x - reduce 30%"
                        cats['low_potential'].append(kd)
                    elif sp >= 20 and ro >= 1.5 and ro < 2.5 and c >= 3:
                        kd['Reason'] = f"Good potential ROAS {ro:.2f}x - test +10-15%"
                        cats['opportunities'].append(kd)
                    elif c >= 3 and sp < 50 and sa == 0:
                        kd['Reason'] = f"{c} clicks, ₹{sp:.0f} - gather more data"
                        cats['future_watch'].append(kd)

                except Exception as e:
                    continue

            return cats
        except Exception as e:
            st.error(f"Classify error: {e}")
            return cats

    def get_future_scale_keywords(self):
        fk = []
        try:
            if self.df is None:
                return fk

            for _, r in self.df.iterrows():
                try:
                    sp = safe_float(r.get('Spend', 0))
                    sa = safe_float(r.get('Sales', 0))
                    o = safe_int(r.get('Orders', 0))
                    c = safe_int(r.get('Clicks', 0))
                    cv = safe_float(r.get('CVR', 0))
                    ro = safe_float(r.get('ROAS', 0))
                    cpc = safe_float(r.get('CPC', 0))

                    if c >= 3 and sp < 100:
                        if o == 1 and ro >= 1.5:
                            fk.append({
                                'Keyword': safe_str(r.get('Customer Search Term')),
                                'Match Type': safe_str(r.get('Match Type')),
                                'Clicks': c, 'Orders': o,
                                'Spend': format_currency(sp),
                                'CPC': format_currency(cpc),
                                'ROAS': f"{ro:.2f}x",
                                'Status': '🟡 Promising',
                                'Action': '1 order, monitor for more',
                                'Recommendation': 'Keep current bid, watch closely'
                            })
                        elif c >= 5 and o == 0 and sp < 50:
                            fk.append({
                                'Keyword': safe_str(r.get('Customer Search Term')),
                                'Match Type': safe_str(r.get('Match Type')),
                                'Clicks': c, 'Orders': o,
                                'Spend': format_currency(sp),
                                'CPC': format_currency(cpc),
                                'ROAS': '0.00x',
                                'Status': '⚪ Watching',
                                'Action': 'Relevant but no sales yet',
                                'Recommendation': 'Give 50 more clicks before deciding'
                            })
                except:
                    continue

            return fk
        except:
            return []

    def get_match_type_strategy(self):
        s = {'current_performance': {}, 'recommendations': [], 'summary': {}}

        try:
            if self.df is None or 'Match Type' not in self.df.columns:
                return s

            df_clean = self.df.copy()
            df_clean['Match Type'] = df_clean['Match Type'].fillna('UNKNOWN').str.upper().str.strip()
            
            for mt in ['EXACT', 'PHRASE', 'BROAD']:
                md = df_clean[df_clean['Match Type'] == mt]
                if len(md) > 0:
                    ts = safe_float(md['Spend'].sum())
                    tsa = safe_float(md['Sales'].sum())
                    to = safe_int(md['Orders'].sum())
                    tc = safe_int(md['Clicks'].sum())
                    ti = safe_int(md['Impressions'].sum())
                    ro = (tsa / ts) if ts > 0 else 0
                    ac = (ts / tsa * 100) if tsa > 0 else 0
                    cv = (to / tc * 100) if tc > 0 else 0
                    ct = (tc / ti * 100) if ti > 0 else 0

                    s['current_performance'][mt] = {
                        'spend': ts, 'sales': tsa, 'roas': ro, 'orders': to,
                        'acos': ac, 'cvr': cv, 'ctr': ct, 'clicks': tc,
                        'impressions': ti, 'keywords': len(md)
                    }

                    if mt == 'EXACT':
                        if ro >= 3.0:
                            s['recommendations'].append({
                                'match_type': 'EXACT', 'action': '✅ SCALE AGGRESSIVELY',
                                'reason': f'Excellent ROAS {ro:.2f}x - increase bids 20-30%', 'priority': 'HIGH'
                            })
                        elif ro >= 2.0:
                            s['recommendations'].append({
                                'match_type': 'EXACT', 'action': '⚡ SCALE MODERATELY',
                                'reason': f'Good ROAS {ro:.2f}x - increase bids 10-15%', 'priority': 'MEDIUM'
                            })
                    elif mt == 'PHRASE':
                        if ro >= 2.5:
                            s['recommendations'].append({
                                'match_type': 'PHRASE', 'action': '⚡ SCALE & TEST',
                                'reason': f'Strong ROAS {ro:.2f}x - find more exact matches', 'priority': 'MEDIUM'
                            })
                    elif mt == 'BROAD':
                        if ro < 1.0:
                            s['recommendations'].append({
                                'match_type': 'BROAD', 'action': '🚨 REDUCE/PAUSE',
                                'reason': f'Poor ROAS {ro:.2f}x - losing money', 'priority': 'HIGH'
                            })

            total_spend = sum(p['spend'] for p in s['current_performance'].values())
            total_sales = sum(p['sales'] for p in s['current_performance'].values())
            s['summary'] = {
                'total_spend': total_spend, 'total_sales': total_sales,
                'overall_roas': (total_sales / total_spend) if total_spend > 0 else 0
            }

            return s
        except Exception as e:
            st.error(f"Match type error: {e}")
            return s

    def get_match_type_performance(self):
        try:
            if self.df is None or 'Match Type' not in self.df.columns:
                return pd.DataFrame()

            df_clean = self.df.copy()
            df_clean['Match Type'] = df_clean['Match Type'].fillna('UNKNOWN').str.upper().str.strip()
            
            df2 = df_clean[df_clean['Match Type'].isin(['EXACT', 'PHRASE', 'BROAD'])].copy()
            
            if len(df2) == 0:
                return pd.DataFrame()

            mp = df2.groupby('Match Type').agg({
                'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum',
                'Clicks': 'sum', 'Impressions': 'sum',
                'Customer Search Term': 'count'
            }).rename(columns={'Customer Search Term': 'Keywords'})

            mp['ROAS'] = mp.apply(lambda x: (safe_float(x['Sales']) / safe_float(x['Spend'])) if safe_float(x['Spend']) > 0 else 0, axis=1)
            mp['ACOS'] = mp.apply(lambda x: (safe_float(x['Spend']) / safe_float(x['Sales']) * 100) if safe_float(x['Sales']) > 0 else 0, axis=1)
            mp['CVR'] = mp.apply(lambda x: (safe_float(x['Orders']) / safe_float(x['Clicks']) * 100) if safe_float(x['Clicks']) > 0 else 0, axis=1)
            mp['CTR'] = mp.apply(lambda x: (safe_float(x['Clicks']) / safe_float(x['Impressions']) * 100) if safe_float(x['Impressions']) > 0 else 0, axis=1)
            mp['CPA'] = mp.apply(lambda x: (safe_float(x['Spend']) / safe_float(x['Orders'])) if safe_float(x['Orders']) > 0 else 0, axis=1)
            mp['AOV'] = mp.apply(lambda x: (safe_float(x['Sales']) / safe_float(x['Orders'])) if safe_float(x['Orders']) > 0 else 0, axis=1)

            return mp
        except Exception as e:
            st.error(f"Match type performance error: {e}")
            return pd.DataFrame()

    def get_roas_improvement_plan(self):
        s = self.get_client_summary()
        cr = s['roas']
        c = self.classify_keywords_improved()

        p = {
            'current_roas': cr, 'target_roas': self.target_roas or 3.0,
            'gap': (self.target_roas or 3.0) - cr,
            'immediate_actions': [], 'short_term': [], 'long_term': []
        }

        wc = len(c['wastage'])
        if wc > 0:
            p['immediate_actions'].append({
                'priority': '🚨 URGENT', 'action': f'Pause {wc} wastage keywords',
                'impact': 'Significant cost savings',
                'how': 'Go to Exports → Download Negatives → Upload to Amazon'
            })

        hp = len(c['high_potential'])
        if hp > 0:
            p['short_term'].append({
                'priority': '🏆 HIGH', 'action': f'Scale {hp} winning keywords',
                'impact': 'Sales increase 20-30%', 'how': 'Increase bids 15-25% on champions'
            })

        op = len(c['opportunities'])
        if op > 0:
            p['short_term'].append({
                'priority': '⚡ MEDIUM', 'action': f'Test {op} opportunity keywords',
                'impact': 'Find new winners', 'how': 'Increase bids 10-15%, monitor closely'
            })

        if cr < 1.0:
            p['immediate_actions'].insert(0, {
                'priority': '🚨 CRITICAL', 'action': 'PAUSE UNDERPERFORMING CAMPAIGNS',
                'impact': 'Stop losing money immediately',
                'how': 'Review product pricing, listings, and targeting'
            })

        p['short_term'].append({
            'priority': '📈 MEDIUM', 'action': 'Optimize product listings',
            'impact': 'CVR improvement 50-100%', 'how': 'Better images, A+ content, reviews'
        })

        p['long_term'].append({
            'priority': '🔍 ONGOING', 'action': 'Test new keywords weekly',
            'impact': 'Continuous growth', 'how': 'Add 10-20 new keywords per week'
        })

        return p

    def get_bid_suggestions_improved(self):
        sug = []

        try:
            if self.df is None:
                return sug

            ta = self.target_acos or 30.0
            tr = self.target_roas or 3.0

            for _, r in self.df.iterrows():
                try:
                    sp = safe_float(r.get('Spend', 0))
                    sa = safe_float(r.get('Sales', 0))
                    ro = safe_float(r.get('ROAS', 0))
                    o = safe_int(r.get('Orders', 0))
                    c = safe_int(r.get('Clicks', 0))
                    cv = safe_float(r.get('CVR', 0))
                    cpc = safe_float(r.get('CPC', 0))

                    # FIXED: Skip if CPC is 0 (can't suggest bid)
                    if sp < 20 or c < 3 or cpc <= 0:
                        continue

                    s = {
                        'Keyword': safe_str(r.get('Customer Search Term')),
                        'Campaign': safe_str(r.get('Campaign Name')),
                        'Ad Group': safe_str(r.get('Ad Group Name')),
                        'Match Type': safe_str(r.get('Match Type')),
                        'Current CPC': format_currency(cpc),
                        'Spend': format_currency(sp),
                        'Sales': format_currency(sa),
                        'ROAS': f"{ro:.2f}x",
                        'CVR': f"{cv:.2f}%",
                        'Orders': o,
                        'Action': '',
                        'Suggested Bid': '',
                        'Change (%)': 0,
                        'Reason': ''
                    }

                    ac = (sp / sa * 100) if sa > 0 else 999

                    if ro >= 3.0 and cv >= 2.0 and o >= 2:
                        nb = cpc * 1.25
                        s.update({
                            'Action': '⬆️ INCREASE',
                            'Suggested Bid': format_currency(nb),
                            'Change (%)': 25,
                            'Reason': f"Champion keyword! ROAS {ro:.2f}x"
                        })
                        sug.append(s)
                    elif ro >= tr and cv >= 1.0 and o >= 1:
                        nb = cpc * 1.15
                        s.update({
                            'Action': '⬆️ INCREASE',
                            'Suggested Bid': format_currency(nb),
                            'Change (%)': 15,
                            'Reason': 'Above target ROAS'
                        })
                        sug.append(s)
                    elif sa == 0 and sp >= 50:
                        s.update({
                            'Action': '⏸️ PAUSE',
                            'Suggested Bid': '₹0.00',
                            'Change (%)': -100,
                            'Reason': f"₹{sp:.0f} wasted, no sales"
                        })
                        sug.append(s)
                    elif ro < 1.5 and sp >= 30:
                        nb = cpc * 0.7
                        s.update({
                            'Action': '⬇️ REDUCE',
                            'Suggested Bid': format_currency(nb),
                            'Change (%)': -30,
                            'Reason': f"Poor ROAS {ro:.2f}x"
                        })
                        sug.append(s)
                    elif ac > ta and sp >= 30:
                        red = min(30, (ac - ta) / ta * 100)
                        nb = cpc * (1 - red / 100)
                        s.update({
                            'Action': '⬇️ REDUCE',
                            'Suggested Bid': format_currency(nb),
                            'Change (%)': -int(red),
                            'Reason': f"ACOS {ac:.1f}% above target {ta:.1f}%"
                        })
                        sug.append(s)

                except:
                    continue

            return sorted(sug, key=lambda x: safe_float(x['Spend'].replace('₹', '').replace(',', '').replace('L','').replace('Cr','')), reverse=True)
        except:
            return []

    def generate_client_report(self):
        try:
            s = self.get_client_summary()
            h = self.get_health_score()
            c = self.classify_keywords_improved()
            
            hs = "EXCELLENT" if h >= 70 else "GOOD" if h >= 50 else "NEEDS ATTENTION"
            tas = f"{self.target_acos:.1f}%" if self.target_acos else "Not Set (Default: 30%)"
            trs = f"{self.target_roas:.1f}x" if self.target_roas else "Not Set (Default: 3.0x)"
            tcpa = format_currency(self.target_cpa) if self.target_cpa else "Not Set"
            ttcoas = f"{self.target_tcoas:.1f}%" if self.target_tcoas else "Not Set"

            return f"""
================================================================================
                    AMAZON PPC PERFORMANCE REPORT
================================================================================
Client: {self.client_name}
Date: {datetime.now().strftime('%B %d, %Y')}
================================================================================

📊 OVERALL HEALTH: {h}/100 - {hs}

🎯 TARGETS:
   ACOS:  {tas}
   ROAS:  {trs}
   CPA:   {tcpa}
   TCoAS: {ttcoas}

💰 FINANCIAL PERFORMANCE
--------------------------------------------------------------------------------
   Total Spend:      {format_currency(s['total_spend'])}
   Total Sales:      {format_currency(s['total_sales'])}
   Total Profit:     {format_currency(s['total_profit'])}
   ROAS:             {s['roas']:.2f}x
   ACOS:             {s['acos']:.1f}%
   TCoAS:            {s['tcoas']:.1f}%

📈 KEY METRICS
--------------------------------------------------------------------------------
   Total Orders:     {format_number(s['total_orders'])}
   Total Clicks:     {format_number(s['total_clicks'])}
   Total Impressions:{format_number(s['total_impressions'])}
   CVR:              {s['avg_cvr']:.2f}%
   CTR:              {s['avg_ctr']:.2f}%
   CPA:              {format_currency(s['avg_cpa'])}
   Avg CPC:          {format_currency(s['avg_cpc'])}

💸 WASTAGE ANALYSIS
--------------------------------------------------------------------------------
   Total Wastage:    {format_currency(s['total_wastage'])}
   Wastage %:        {(s['total_wastage'] / s['total_spend'] * 100) if s['total_spend'] > 0 else 0:.1f}%

🎯 KEYWORD CLASSIFICATION
--------------------------------------------------------------------------------
   🏆 Scale Now:      {len(c['high_potential'])}
   ⚡ Test:           {len(c['opportunities'])}
   👀 Watch Future:   {len(c['future_watch'])}
   ⚠️ Reduce:         {len(c['low_potential'])}
   🚨 Pause:          {len(c['wastage'])}

✅ RECOMMENDED ACTIONS
--------------------------------------------------------------------------------
1. Pause {len(c['wastage'])} wastage keywords immediately
2. Scale {len(c['high_potential'])} winning keywords
3. Monitor {len(c['future_watch'])} future opportunities
4. Test bids on {len(c['opportunities'])} opportunity keywords

================================================================================
Generated by Amazon Ads Dashboard Pro v6.0
================================================================================
"""
        except Exception as e:
            return f"Error generating report: {e}"


def init_session_state():
    if 'clients' not in st.session_state:
        st.session_state.clients = {}
    if 'active_client' not in st.session_state:
        st.session_state.active_client = None
    if 'agency_name' not in st.session_state:
        st.session_state.agency_name = "Your Agency"

def render_agency_header():
    st.markdown(f'<div class="agency-header"><h1>🏢 {st.session_state.agency_name}</h1><p>Amazon Ads Dashboard Pro v6.0 - CPC & NEGATIVES FIXED</p><small>✅ CPC Calculation Fixed | ✅ ASIN Negatives Handled</small></div>', unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        with st.expander("⚙️ Settings"):
            nn = st.text_input("Agency Name", value=st.session_state.agency_name)
            if nn != st.session_state.agency_name:
                st.session_state.agency_name = nn
                st.rerun()

        st.markdown("---")
        st.markdown("### 👥 Clients")

        if st.session_state.clients:
            cn = list(st.session_state.clients.keys())
            
            if st.session_state.active_client not in cn:
                st.session_state.active_client = cn[0] if cn else None
            
            sel = st.selectbox("Active Client", cn, 
                              index=cn.index(st.session_state.active_client) if st.session_state.active_client in cn else 0,
                              key="client_selector")
            
            if sel != st.session_state.active_client:
                st.session_state.active_client = sel
                st.rerun()
            
            if sel:
                cl = st.session_state.clients[sel]
                if cl.analyzer and cl.analyzer.df is not None:
                    try:
                        h = cl.analyzer.get_health_score()
                        em = "🟢" if h >= 70 else "🟡" if h >= 50 else "🔴"
                        st.info(f"{em} Health Score: {h}/100")
                    except:
                        pass

        st.markdown("---")
        
        with st.expander("➕ Add New Client"):
            nm = st.text_input("Client Name*", key="add_client_name")
            ind = st.selectbox("Industry", 
                ["E-commerce", "Electronics", "Fashion", "Beauty", "Home", "Sports", "Books", "Health", "Other"], 
                key="add_industry")
            bug = st.number_input("Monthly Budget (₹)", value=50000, step=5000, key="add_budget")

            st.info("🎯 Performance Targets (Optional)")
            c1, c2 = st.columns(2)
            with c1:
                tacos = st.number_input("Target ACOS %", value=0.0, step=5.0, key="add_acos")
                troas = st.number_input("Target ROAS", value=0.0, step=0.5, key="add_roas")
            with c2:
                tcpa = st.number_input("Target CPA ₹", value=0.0, step=50.0, key="add_cpa")
                ttcoas = st.number_input("Target TCoAS %", value=0.0, step=5.0, key="add_tcoas")

            em = st.text_input("Contact Email", key="add_email")
            up = st.file_uploader("Upload Search Term Report*", type=["xlsx", "xls", "csv"], key="add_file")

            if st.button("✅ Add Client", type="primary", use_container_width=True, key="add_btn"):
                if not nm:
                    st.error("❌ Please enter client name")
                elif not up:
                    st.error("❌ Please upload a report file")
                elif nm in st.session_state.clients:
                    st.error(f"❌ Client '{nm}' already exists")
                else:
                    try:
                        with st.spinner(f"Processing data for {nm}..."):
                            if up.name.endswith('.csv'):
                                df = pd.read_csv(up)
                            else:
                                df = pd.read_excel(up)
                            
                            st.info(f"📊 Loaded {len(df)} rows for {nm}")

                            cd = ClientData(nm, ind, bug)
                            cd.contact_email = em
                            cd.target_acos = tacos if tacos > 0 else None
                            cd.target_roas = troas if troas > 0 else None
                            cd.target_cpa = tcpa if tcpa > 0 else None
                            cd.target_tcoas = ttcoas if ttcoas > 0 else None

                            cd.analyzer = CompleteAnalyzer(
                                df.copy(deep=True), nm,
                                cd.target_acos, cd.target_roas,
                                cd.target_cpa, cd.target_tcoas
                            )

                            st.session_state.clients[nm] = cd
                            st.session_state.active_client = nm

                            st.success(f"✅ Successfully added {nm}!")
                            st.balloons()
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("🔍 Debug Details"):
                            st.code(traceback.format_exc())

        if st.session_state.clients:
            st.markdown("---")
            st.markdown("### 📋 Client List")
            for cn in list(st.session_state.clients.keys()):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.text(f"📊 {cn}")
                with c2:
                    if st.button("🗑️", key=f"del_{cn}"):
                        del st.session_state.clients[cn]
                        if st.session_state.active_client == cn:
                            st.session_state.active_client = None
                        st.rerun()


def render_metric_card(label, value, color="#fff"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """

def render_dashboard_tab(cl, an):
    try:
        st.header(f"📊 {cl.name} Dashboard")
        
        s = an.get_client_summary()
        h = an.get_health_score()

        tad = f"{cl.target_acos:.1f}%" if cl.target_acos else "30% (default)"
        trd = f"{cl.target_roas:.1f}x" if cl.target_roas else "3.0x (default)"
        tcpa = format_currency(cl.target_cpa) if cl.target_cpa else "Not Set"
        ttcoas = f"{cl.target_tcoas:.1f}%" if cl.target_tcoas else "Not Set"

        health_color = "#22c55e" if h >= 70 else "#facc15" if h >= 50 else "#ef4444"
        st.markdown(f'<div class="info-box"><h2 style="color:{health_color}">Health Score: {h}/100</h2><p><strong>Targets:</strong> ACOS: {tad} | ROAS: {trd} | CPA: {tcpa} | TCoAS: {ttcoas}</p></div>', unsafe_allow_html=True)

        # Financial Metrics
        st.markdown("---")
        st.markdown("### 💰 Financial Performance")
        
        fin_cols = st.columns(5)
        fin_metrics = [
            ("Total Spend", format_currency(s['total_spend']), "#fff"),
            ("Total Sales", format_currency(s['total_sales']), "#22c55e" if s['total_sales'] > 0 else "#fff"),
            ("ROAS", f"{s['roas']:.2f}x", "#22c55e" if s['roas'] >= 3 else "#facc15" if s['roas'] >= 2 else "#ef4444"),
            ("Orders", format_number(s['total_orders']), "#fff"),
            ("Profit", format_currency(s['total_profit']), "#22c55e" if s['total_profit'] > 0 else "#ef4444")
        ]
        
        for col, (label, value, color) in zip(fin_cols, fin_metrics):
            with col:
                st.markdown(render_metric_card(label, value, color), unsafe_allow_html=True)

        # Key Metrics
        st.markdown("---")
        st.markdown("### 📈 Key Metrics")
        
        key_cols = st.columns(5)
        wp = (s['total_wastage'] / s['total_spend'] * 100) if s['total_spend'] > 0 else 0
        
        key_metrics = [
            ("CVR", f"{s['avg_cvr']:.2f}%"),
            ("ACOS", f"{s['acos']:.1f}%"),
            ("CTR", f"{s['avg_ctr']:.2f}%"),
            ("CPA", format_currency(s['avg_cpa'])),
            ("Wastage", f"{format_currency(s['total_wastage'])} ({wp:.0f}%)")
        ]
        
        for col, (label, value) in zip(key_cols, key_metrics):
            with col:
                st.markdown(render_metric_card(label, value), unsafe_allow_html=True)

        # Additional Metrics
        st.markdown("---")
        st.markdown("### 📊 Additional Metrics")
        
        add_cols = st.columns(4)
        add_metrics = [
            ("Avg CPC", format_currency(s['avg_cpc'])),
            ("TCoAS", f"{s['tcoas']:.1f}%"),
            ("Keywords", format_number(s['keywords_count'])),
            ("Campaigns", format_number(s['campaigns_count']))
        ]
        
        for col, (label, value) in zip(add_cols, add_metrics):
            with col:
                st.markdown(render_metric_card(label, value), unsafe_allow_html=True)

        # Performance Insights
        st.markdown("---")
        st.markdown("### 💡 Performance Insights")
        insights = an.get_performance_insights()

        for insight_type in ['ctr_insights', 'cvr_insights', 'roas_insights', 'acos_insights', 'cpa_insights', 'tcoas_insights']:
            if insights.get(insight_type):
                for ins in insights[insight_type]:
                    box_class = 'danger-box' if ins['level'] == 'critical' else 'warning-box' if ins['level'] == 'warning' else 'success-box' if ins['level'] == 'success' else 'info-box'
                    st.markdown(f'<div class="{box_class}"><strong>{ins["metric"]}: {ins["value"]}</strong><br>📌 Issue: {ins["issue"]}<br>✅ Action: {ins["action"]}</div>', unsafe_allow_html=True)

        if insights.get('action_items'):
            st.markdown("---")
            st.markdown("### 🎯 Priority Action Items")
            for item in insights['action_items']:
                st.markdown(f'<div class="danger-box">🚨 {item}</div>', unsafe_allow_html=True)

        if insights.get('content_suggestions'):
            st.markdown("---")
            st.markdown("### 📝 Content & Listing Suggestions")
            st.markdown('<div class="cyan-box"><strong>💡 Recommendations to improve performance:</strong></div>', unsafe_allow_html=True)
            for sug in insights['content_suggestions']:
                st.markdown(f"- {sug}")

        # ROAS Improvement Plan
        st.markdown("---")
        st.markdown("### 🎯 ROAS Improvement Plan")
        p = an.get_roas_improvement_plan()
        st.markdown(f'<div class="info-box"><strong>Current: {p["current_roas"]:.2f}x | Target: {p["target_roas"]:.2f}x | Gap: {p["gap"]:.2f}x</strong></div>', unsafe_allow_html=True)

        if p['immediate_actions']:
            st.markdown("#### 🚨 IMMEDIATE ACTIONS")
            for a in p['immediate_actions']:
                st.markdown(f'<div class="danger-box"><strong>{a["priority"]}: {a["action"]}</strong><br>💰 Impact: {a["impact"]}<br>📋 How: {a["how"]}</div>', unsafe_allow_html=True)

        if p['short_term']:
            st.markdown("#### ⚡ SHORT TERM ACTIONS")
            for a in p['short_term']:
                st.markdown(f'<div class="warning-box"><strong>{a["priority"]}: {a["action"]}</strong><br>💰 Impact: {a["impact"]}<br>📋 How: {a["how"]}</div>', unsafe_allow_html=True)

        if p['long_term']:
            st.markdown("#### 📈 LONG TERM ACTIONS")
            for a in p['long_term']:
                st.markdown(f'<div class="purple-box"><strong>{a["priority"]}: {a["action"]}</strong><br>💰 Impact: {a["impact"]}<br>📋 How: {a["how"]}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        with st.expander("🔍 Debug"):
            st.code(traceback.format_exc())


def render_keywords_tab(an):
    try:
        st.header("🎯 Keywords Analysis")
        c = an.classify_keywords_improved()

        st.markdown("### 📊 Keyword Classification Summary")
        kw_cols = st.columns(5)
        kw_metrics = [
            ("🏆 Scale", len(c['high_potential']), "#22c55e"),
            ("⚡ Test", len(c['opportunities']), "#facc15"),
            ("👀 Watch", len(c['future_watch']), "#3b82f6"),
            ("⚠️ Reduce", len(c['low_potential']), "#f97316"),
            ("🚨 Pause", len(c['wastage']), "#ef4444")
        ]
        
        for col, (label, count, color) in zip(kw_cols, kw_metrics):
            with col:
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.95);border:2px solid {color};border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-size:1.1rem;color:{color};font-weight:bold;white-space:nowrap;">{label}</div>
                    <div style="font-size:1.8rem;color:#fff;font-weight:bold;">{count}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        tabs = st.tabs([
            f"🏆 Scale Now ({len(c['high_potential'])})",
            f"⚡ Test Opportunities ({len(c['opportunities'])})",
            f"👀 Future Watch ({len(c['future_watch'])})",
            f"⚠️ Reduce ({len(c['low_potential'])})",
            f"🚨 Pause/Wastage ({len(c['wastage'])})",
            "🔮 Future Scale"
        ])

        with tabs[0]:
            if c['high_potential']:
                st.success("✅ These are your champion keywords! Scale bids by 15-25%")
                df_hp = pd.DataFrame(c['high_potential'])
                st.dataframe(df_hp, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("💡 No champion keywords yet. Look for keywords with ROAS ≥2.5x AND at least 1 order")

        with tabs[1]:
            if c['opportunities']:
                st.info("⚡ These keywords show promise. Test with +10-15% bid increases")
                df_op = pd.DataFrame(c['opportunities'])
                st.dataframe(df_op, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No opportunity keywords found.")

        with tabs[2]:
            if c['future_watch']:
                st.markdown('<div class="info-box"><strong>👀 Future Watch</strong><br>Keywords with clicks but need more data before deciding</div>', unsafe_allow_html=True)
                df_fw = pd.DataFrame(c['future_watch'])
                st.dataframe(df_fw, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No keywords in watch list")

        with tabs[3]:
            if c['low_potential']:
                st.warning("⚠️ These keywords are underperforming. Reduce bids by 30% or pause")
                df_lp = pd.DataFrame(c['low_potential'])
                st.dataframe(df_lp, use_container_width=True, hide_index=True, height=400)
            else:
                st.success("✅ No low performers found!")

        with tabs[4]:
            if c['wastage']:
                try:
                    tw = sum(float(k['Spend'].replace('₹', '').replace(',', '').replace('L','').replace('Cr','')) for k in c['wastage'])
                except:
                    tw = 0
                st.error(f"🚨 {format_currency(tw)} wasted on keywords with ZERO sales")
                st.markdown('<div class="danger-box"><strong>⚠️ Wastage Definition:</strong> Keywords with significant spend but ZERO sales<br><strong>Action:</strong> Download as negatives → Upload to Amazon Campaign Manager</div>', unsafe_allow_html=True)
                df_w = pd.DataFrame(c['wastage'])
                st.dataframe(df_w, use_container_width=True, hide_index=True, height=400)
            else:
                st.success("🎉 No wastage! All keywords have generated sales!")

        with tabs[5]:
            fsk = an.get_future_scale_keywords()
            if fsk:
                st.markdown("### 🔮 Keywords to Scale in Future")
                st.info("These keywords show early promise but need more data")
                df_fsk = pd.DataFrame(fsk)
                st.dataframe(df_fsk, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No future scale candidates yet.")

    except Exception as e:
        st.error(f"❌ Keywords error: {e}")
        with st.expander("🔍 Debug"):
            st.code(traceback.format_exc())


def render_match_type_tab(an):
    try:
        st.header("📊 Match Type Strategy")

        mp = an.get_match_type_performance()
        
        if not mp.empty:
            st.subheader("📈 Performance by Match Type")
            
            dm = pd.DataFrame()
            dm['Match Type'] = mp.index
            dm['Spend'] = mp['Spend'].apply(format_currency)
            dm['Sales'] = mp['Sales'].apply(format_currency)
            dm['Orders'] = mp['Orders'].apply(format_number)
            dm['Clicks'] = mp['Clicks'].apply(format_number)
            dm['Impressions'] = mp['Impressions'].apply(format_number)
            dm['Keywords'] = mp['Keywords'].apply(format_number)
            dm['ROAS'] = mp['ROAS'].apply(lambda x: f"{x:.2f}x")
            dm['ACOS'] = mp['ACOS'].apply(lambda x: f"{x:.1f}%")
            dm['CVR'] = mp['CVR'].apply(lambda x: f"{x:.2f}%")
            dm['CTR'] = mp['CTR'].apply(lambda x: f"{x:.2f}%")
            dm['CPA'] = mp['CPA'].apply(format_currency)
            dm['AOV'] = mp['AOV'].apply(format_currency)
            
            st.dataframe(dm, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("### 📊 Match Type Summary")
            
            for mt in mp.index:
                p = mp.loc[mt]
                cols = st.columns(4)
                metrics = [
                    (f"{mt} Spend", format_currency(p['Spend'])),
                    (f"{mt} Sales", format_currency(p['Sales'])),
                    (f"{mt} ROAS", f"{p['ROAS']:.2f}x"),
                    (f"{mt} Orders", format_number(p['Orders']))
                ]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.markdown(render_metric_card(label, value), unsafe_allow_html=True)
        else:
            st.info("No match type data available.")

        st.markdown("---")
        st.markdown("### 🎯 Your Match Type Strategy")
        s = an.get_match_type_strategy()

        if s.get('recommendations'):
            for r in s['recommendations']:
                box = 'danger-box' if r['priority'] == 'HIGH' else 'warning-box' if r['priority'] == 'MEDIUM' else 'info-box'
                st.markdown(f'<div class="{box}"><strong>{r["match_type"]}:</strong> {r["action"]}<br>📌 Reason: {r["reason"]}<br>🔥 Priority: {r["priority"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📚 Match Type Guide")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="success-box"><h4>🎯 EXACT Match</h4><strong>When:</strong> Proven winners (ROAS ≥3.0x)<br><strong>Bid:</strong> Aggressive (scale these)<br><strong>Example:</strong> "blue water bottle 1 litre"</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="warning-box"><h4>📝 PHRASE Match</h4><strong>When:</strong> Discovery & testing<br><strong>Bid:</strong> Moderate (find winners)<br><strong>Example:</strong> "water bottle" → matches "best water bottle"</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="info-box"><h4>🌐 BROAD Match</h4><strong>When:</strong> Research only<br><strong>Bid:</strong> Low budget tests<br><strong>Example:</strong> "bottle" → matches anything</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Match type error: {e}")


def render_bid_tab(an):
    try:
        st.header("💡 Bid Optimization")

        tad = f"{an.target_acos:.1f}%" if an.target_acos else "30% (default)"
        trd = f"{an.target_roas:.1f}x" if an.target_roas else "3.0x (default)"
        tcpa = format_currency(an.target_cpa) if an.target_cpa else "Not Set"

        st.markdown(f'<div class="info-box"><strong>🎯 Optimization Targets</strong><br>ACOS: {tad} | ROAS: {trd} | CPA: {tcpa}</div>', unsafe_allow_html=True)

        sug = an.get_bid_suggestions_improved()

        if sug:
            af = st.selectbox("Filter by Action", ["All", "⬆️ INCREASE", "⬇️ REDUCE", "⏸️ PAUSE"], key="bid_filter")
            filt = sug if af == "All" else [s for s in sug if af in s['Action']]

            inc = len([s for s in sug if 'INCREASE' in s['Action']])
            red = len([s for s in sug if 'REDUCE' in s['Action']])
            pau = len([s for s in sug if 'PAUSE' in s['Action']])

            st.markdown("---")
            bid_cols = st.columns(3)
            bid_metrics = [
                ("⬆️ Scale Up", inc, "#22c55e"),
                ("⬇️ Reduce", red, "#f97316"),
                ("⏸️ Pause", pau, "#ef4444")
            ]
            
            for col, (label, count, color) in zip(bid_cols, bid_metrics):
                with col:
                    st.markdown(f"""
                    <div style="background:rgba(30,41,59,0.95);border:2px solid {color};border-radius:12px;padding:1rem;text-align:center;">
                        <div style="font-size:1.1rem;color:{color};font-weight:bold;white-space:nowrap;">{label}</div>
                        <div style="font-size:1.8rem;color:#fff;font-weight:bold;">{count}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"---")
            st.markdown(f"**Showing {len(filt)} of {len(sug)} bid suggestions**")
            
            df_sug = pd.DataFrame(filt)
            st.dataframe(df_sug, use_container_width=True, hide_index=True, height=500)
        else:
            st.info("💡 No bid suggestions available. Need keywords with CPC > 0, min ₹20 spend and 3 clicks.")

    except Exception as e:
        st.error(f"❌ Bid optimization error: {e}")


def render_exports_tab(an, cn):
    """FIXED: Handle both keyword negatives and product ASIN negatives"""
    try:
        st.header("📥 Export Files")
        c = an.classify_keywords_improved()
        sug = an.get_bid_suggestions_improved()

        # Separate keywords and ASINs from wastage
        wastage_keywords = []
        wastage_asins = []
        
        for k in c['wastage']:
            kw = k.get('Keyword', '')
            if is_asin(kw):
                wastage_asins.append(k)
            else:
                wastage_keywords.append(k)

        exp_cols = st.columns(3)

        with exp_cols[0]:
            st.subheader("🚫 Negative Keywords & Products")
            
            # Create two sheets: one for keywords, one for ASINs
            if wastage_keywords or wastage_asins:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                    # Sheet 1: Negative Keywords (with match types)
                    if wastage_keywords:
                        nk_data = []
                        for k in wastage_keywords:
                            nk_data.append({
                                'Campaign': k['Campaign'],
                                'Ad Group': '',
                                'Keyword': k['Keyword'],
                                'Match Type': 'Negative Exact',
                                'Status': 'Enabled',
                                'Type': 'Keyword Negative'
                            })
                        pd.DataFrame(nk_data).to_excel(wr, sheet_name='Negative Keywords', index=False)
                    
                    # Sheet 2: Negative Products (ASINs - no match type)
                    if wastage_asins:
                        np_data = []
                        for k in wastage_asins:
                            np_data.append({
                                'Campaign': k['Campaign'],
                                'Ad Group': '',
                                'ASIN': k['Keyword'],  # This is the ASIN
                                'Status': 'Enabled',
                                'Type': 'Product Negative',
                                'Note': 'Add to Product Targeting > Negative Products'
                            })
                        pd.DataFrame(np_data).to_excel(wr, sheet_name='Negative Products', index=False)
                    
                    # Sheet 3: Combined Summary
                    summary_data = []
                    if wastage_keywords:
                        summary_data.append({'Type': 'Keyword Negatives', 'Count': len(wastage_keywords), 'Instructions': 'Upload to Campaign > Negative Keywords'})
                    if wastage_asins:
                        summary_data.append({'Type': 'Product Negatives (ASINs)', 'Count': len(wastage_asins), 'Instructions': 'Upload to Product Targeting > Negative Products'})
                    if summary_data:
                        pd.DataFrame(summary_data).to_excel(wr, sheet_name='Instructions', index=False)
                
                out.seek(0)
                
                total_negatives = len(wastage_keywords) + len(wastage_asins)
                st.download_button(
                    f"📥 Download Negatives ({total_negatives} total)",
                    data=out,
                    file_name=f"Negatives_{cn}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="neg_download"
                )
                
                # Show breakdown
                if wastage_keywords:
                    st.error(f"🚨 {len(wastage_keywords)} keyword negatives")
                if wastage_asins:
                    st.warning(f"📦 {len(wastage_asins)} product ASIN negatives")
                    
                st.info("""
                **How to upload:**
                - **Keywords**: Campaign → Negative Keywords → Upload
                - **ASINs**: Product Targeting → Negative Products → Enter ASIN
                """)
            else:
                st.success("✅ No negatives needed")

        with exp_cols[1]:
            st.subheader("💰 Bid Adjustments")
            if sug:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                    pd.DataFrame(sug).to_excel(wr, index=False, sheet_name='Bid Suggestions')
                out.seek(0)
                st.download_button(
                    f"📥 Download Bids ({len(sug)} suggestions)",
                    data=out,
                    file_name=f"Bids_{cn}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="bid_download"
                )
                st.success(f"✅ {len(sug)} bid suggestions ready")
            else:
                st.info("No bid adjustments needed")

        with exp_cols[2]:
            st.subheader("📊 Complete Data")
            if an.df is not None:
                csv = an.df.to_csv(index=False)
                st.download_button(
                    f"📥 Full Data CSV ({len(an.df)} rows)",
                    data=csv,
                    file_name=f"Full_Data_{cn}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="csv_download"
                )
                st.success("✅ Full dataset available")
            else:
                st.info("No data available")

    except Exception as e:
        st.error(f"❌ Export error: {e}")


def render_report_tab(cl, an):
    try:
        st.header("📝 Client Report")
        rep = an.generate_client_report()
        st.text_area("Report Preview", rep, height=500, key="report_text")

        rep_cols = st.columns(2)

        with rep_cols[0]:
            st.download_button(
                "📄 Download Report (TXT)",
                data=rep,
                file_name=f"Report_{cl.name}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="txt_download"
            )

        with rep_cols[1]:
            s = an.get_client_summary()
            c = an.classify_keywords_improved()
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                summary_df = pd.DataFrame([{
                    'Metric': 'Total Spend', 'Value': format_currency(s['total_spend'])
                }, {
                    'Metric': 'Total Sales', 'Value': format_currency(s['total_sales'])
                }, {
                    'Metric': 'ROAS', 'Value': f"{s['roas']:.2f}x"
                }, {
                    'Metric': 'ACOS', 'Value': f"{s['acos']:.1f}%"
                }, {
                    'Metric': 'CVR', 'Value': f"{s['avg_cvr']:.2f}%"
                }, {
                    'Metric': 'CTR', 'Value': f"{s['avg_ctr']:.2f}%"
                }, {
                    'Metric': 'Avg CPC', 'Value': format_currency(s['avg_cpc'])
                }, {
                    'Metric': 'Health Score', 'Value': f"{an.get_health_score()}/100"
                }])
                summary_df.to_excel(wr, sheet_name='Summary', index=False)
                
                if c['high_potential']:
                    pd.DataFrame(c['high_potential']).to_excel(wr, sheet_name='High Potential', index=False)
                if c['opportunities']:
                    pd.DataFrame(c['opportunities']).to_excel(wr, sheet_name='Opportunities', index=False)
                if c['wastage']:
                    pd.DataFrame(c['wastage']).to_excel(wr, sheet_name='Wastage', index=False)
            out.seek(0)
            st.download_button(
                "📊 Download Report (Excel)",
                data=out,
                file_name=f"Report_{cl.name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="excel_download"
            )

    except Exception as e:
        st.error(f"❌ Report error: {e}")


def render_all_clients_tab():
    try:
        st.header("👥 All Clients Overview")

        if not st.session_state.clients:
            st.info("📊 No clients added yet. Add clients from the sidebar!")
            return

        data = []
        for n, c in st.session_state.clients.items():
            if c.analyzer and c.analyzer.df is not None:
                try:
                    s = c.analyzer.get_client_summary()
                    h = c.analyzer.get_health_score()
                    data.append({
                        'Client': n, 'Health': f"{h}/100",
                        'Spend': format_currency(s['total_spend']),
                        'Sales': format_currency(s['total_sales']),
                        'ROAS': f"{s['roas']:.2f}x",
                        'ACOS': f"{s['acos']:.1f}%",
                        'CVR': f"{s['avg_cvr']:.2f}%",
                        'Orders': format_number(s['total_orders']),
                        'Keywords': format_number(s['keywords_count']),
                        'Status': '🟢' if h >= 70 else '🟡' if h >= 50 else '🔴'
                    })
                except Exception as e:
                    continue

        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True, height=400)

            st.markdown("---")
            st.markdown("### 📊 Agency Overview")
            
            ov_cols = st.columns(4)
            ov_metrics = [
                ("Total Clients", len(data)),
                ("🟢 Healthy", len([d for d in data if '🟢' in d['Status']])),
                ("🟡 Okay", len([d for d in data if '🟡' in d['Status']])),
                ("🔴 Attention", len([d for d in data if '🔴' in d['Status']]))
            ]
            
            for col, (label, count) in zip(ov_cols, ov_metrics):
                with col:
                    st.markdown(render_metric_card(label, str(count)), unsafe_allow_html=True)
        else:
            st.info("No client data to display")

    except Exception as e:
        st.error(f"❌ All clients error: {e}")


def render_dashboard():
    render_agency_header()

    if not st.session_state.clients:
        welcome_msg = """
        <div class="info-box">
        <h3>👋 Welcome to Amazon Ads Dashboard Pro v6.0!</h3>
        <br>
        <strong>✅ All Issues Fixed:</strong>
        <ul>
        <li>✅ FIXED: CPC calculation from Spend/Clicks when CPC column missing</li>
        <li>✅ FIXED: Bid suggestions now show correct Current CPC and Suggested Bid</li>
        <li>✅ FIXED: Negative keywords vs Product ASINs handled separately</li>
        <li>✅ Keywords get match types (Negative Exact/Phrase)</li>
        <li>✅ ASINs don't need match types - added to Product Targeting negatives</li>
        </ul>
        <br>
        <strong>👈 Get started by adding a client from the sidebar!</strong>
        </div>
        """
        st.markdown(welcome_msg, unsafe_allow_html=True)
        return

    if not st.session_state.active_client:
        st.warning("⚠️ Please select a client from the sidebar")
        return

    cl = st.session_state.clients.get(st.session_state.active_client)
    
    if not cl:
        st.error("❌ Client not found. Please select another client.")
        return

    if not cl.analyzer or cl.analyzer.df is None:
        st.error("❌ No data loaded for this client. Please re-upload.")
        return

    an = cl.analyzer

    tabs = st.tabs([
        "📊 Dashboard", "🎯 Keywords", "💡 Bid Optimization",
        "📊 Match Types", "📝 Reports", "👥 All Clients", "📥 Exports"
    ])

    with tabs[0]:
        render_dashboard_tab(cl, an)
    with tabs[1]:
        render_keywords_tab(an)
    with tabs[2]:
        render_bid_tab(an)
    with tabs[3]:
        render_match_type_tab(an)
    with tabs[4]:
        render_report_tab(cl, an)
    with tabs[5]:
        render_all_clients_tab()
    with tabs[6]:
        render_exports_tab(an, cl.name)


def main():
    load_custom_css()
    init_session_state()
    render_sidebar()
    render_dashboard()

    footer = f"""
    <div style="text-align:center;color:#94a3b8;padding:1rem;margin-top:2rem;border-top:1px solid rgba(148,163,184,0.3);">
    <strong>{st.session_state.agency_name}</strong><br>
    Amazon Ads Dashboard Pro v6.0 - CPC & NEGATIVES FIXED<br>
    <small>✅ CPC Calculation Fixed | ✅ ASIN Negatives Handled</small>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
