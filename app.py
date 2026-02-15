#!/usr/bin/env python3
import io, traceback
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Amazon Ads Dashboard Pro", page_icon="🏢", layout="wide")

def load_custom_css():
    st.markdown("""<style>
.main{padding-top:1rem;}
.agency-header{background:linear-gradient(135deg,#1e40af,#3b82f6);padding:2rem;border-radius:16px;margin-bottom:2rem;text-align:center;color:white;}
div[data-testid="stMetric"]{background:rgba(30,41,59,0.8);border:1px solid rgba(148,163,184,0.3);border-radius:12px;padding:1.5rem;}
.success-box{background:rgba(22,163,74,0.2);border-left:4px solid #22c55e;padding:1rem;border-radius:8px;margin:1rem 0;}
.warning-box{background:rgba(234,179,8,0.2);border-left:4px solid #facc15;padding:1rem;border-radius:8px;margin:1rem 0;}
.danger-box{background:rgba(220,38,38,0.2);border-left:4px solid #ef4444;padding:1rem;border-radius:8px;margin:1rem 0;}
.info-box{background:rgba(59,130,246,0.2);border-left:4px solid #3b82f6;padding:1rem;border-radius:8px;margin:1rem 0;}
</style>""", unsafe_allow_html=True)

def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or value == '' or value is None: return default
        return float(value)
    except: return default

def safe_int(value, default=0):
    try:
        if pd.isna(value) or value == '' or value is None: return default
        return int(float(value))
    except: return default

def safe_str(value, default='N/A'):
    try:
        if pd.isna(value) or value == '' or value is None: return default
        return str(value).strip()
    except: return default

def format_currency(v): return f"₹{v:,.2f}"
def format_number(v): return f"{v:,.0f}"

class ClientData:
    def __init__(self, name, industry="E-commerce", budget=50000):
        self.name, self.industry, self.monthly_budget = name, industry, budget
        self.analyzer, self.added_date, self.contact_email = None, datetime.now(), ""
        self.target_acos, self.target_roas = None, None

class CompleteAnalyzer:
    REQUIRED_COLUMNS = ['Customer Search Term', 'Campaign Name', 'Spend', 'Clicks']

    def __init__(self, df, client_name, target_acos=None, target_roas=None):
        self.client_name, self.target_acos, self.target_roas = client_name, target_acos, target_roas
        self.df, self.error = None, None
        try: self.df = self._validate_and_prepare_data(df)
        except Exception as e: self.error = str(e); raise ValueError(f"Validation failed: {e}")

    def _validate_and_prepare_data(self, df):
        if df is None or len(df) == 0: raise ValueError("Empty DataFrame")
        df = df.copy()
        df.columns = df.columns.str.strip()
        mapping = {'customer search term':'Customer Search Term','search term':'Customer Search Term','keyword':'Customer Search Term','campaign':'Campaign Name','campaign name':'Campaign Name','ad group':'Ad Group Name','match type':'Match Type','7 day total sales':'Sales','7 day total orders':'Orders','total sales':'Sales','total orders':'Orders','cost':'Spend'}
        df.columns = df.columns.str.lower().str.strip()
        for old, new in mapping.items():
            if old in df.columns: df.rename(columns={old:new}, inplace=True)
        df.columns = [c.title() for c in df.columns]
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing: raise ValueError(f"Missing: {missing}")
        for col in ['Sales','Orders','Impressions','CPC','Ad Group Name','Match Type']:
            if col not in df.columns:
                df[col] = 0 if col in ['Sales','Orders','Impressions','CPC'] else 'N/A'
        if 'Cpc' in df.columns: df['CPC'] = df['Cpc']
        for col in ['Spend','Sales','Clicks','Impressions','Orders','CPC']:
            if col in df.columns:
                if df[col].dtype == 'object': df[col] = df[col].astype(str).str.replace('[₹$,]','',regex=True).str.replace('%','')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df[(df['Spend']>0)|(df['Clicks']>0)].copy()
        if len(df)==0: raise ValueError("No valid data")
        df['Profit'] = df['Sales'] - df['Spend']
        df['Wastage'] = df.apply(lambda x: x['Spend'] if x['Sales']==0 else 0, axis=1)
        df['CVR'] = df.apply(lambda x: (x['Orders']/x['Clicks']*100) if x['Clicks']>0 else 0, axis=1)
        df['ROAS'] = df.apply(lambda x: (x['Sales']/x['Spend']) if x['Spend']>0 else 0, axis=1)
        df['ACOS'] = df.apply(lambda x: (x['Spend']/x['Sales']*100) if x['Sales']>0 else 0, axis=1)
        df['CTR'] = df.apply(lambda x: (x['Clicks']/x['Impressions']*100) if x['Impressions']>0 else 0, axis=1)
        df['Client'], df['Processed_Date'] = self.client_name, datetime.now()
        return df

    def get_client_summary(self):
        try:
            if self.df is None or len(self.df)==0: return self._empty()
            ts, tsa, to, tc, ti, tw = safe_float(self.df['Spend'].sum()), safe_float(self.df['Sales'].sum()), safe_int(self.df['Orders'].sum()), safe_int(self.df['Clicks'].sum()), safe_int(self.df['Impressions'].sum()), safe_float(self.df['Wastage'].sum())
            return {'total_spend':ts,'total_sales':tsa,'total_profit':safe_float(self.df['Profit'].sum()),'total_orders':to,'total_clicks':tc,'total_impressions':ti,'total_wastage':tw,'roas':(tsa/ts if ts>0 else 0),'acos':(ts/tsa*100 if tsa>0 else 0),'avg_cpc':safe_float(self.df['CPC'].mean()),'avg_ctr':safe_float(self.df['CTR'].mean()),'avg_cvr':safe_float(self.df['CVR'].mean()),'conversion_rate':(to/tc*100 if tc>0 else 0),'keywords_count':len(self.df),'campaigns_count':safe_int(self.df['Campaign Name'].nunique())}
        except: return self._empty()

    def _empty(self): return {k:0 for k in ['total_spend','total_sales','total_profit','total_orders','total_clicks','total_impressions','total_wastage','roas','acos','avg_cpc','avg_ctr','avg_cvr','conversion_rate','keywords_count','campaigns_count']}

    def get_health_score(self):
        try:
            s = self.get_client_summary()
            score = 0
            r = s['roas']
            if r>=3.5: score+=50
            elif r>=2.5: score+=40
            elif r>=1.5: score+=25
            elif r>0: score+=10
            wp = (s['total_wastage']/s['total_spend']*100) if s['total_spend']>0 else 0
            if wp<=10: score+=30
            elif wp<=20: score+=20
            elif wp<=30: score+=10
            ctr = s['avg_ctr']
            if ctr>=5: score+=20
            elif ctr>=3: score+=15
            elif ctr>=1: score+=10
            return min(score,100)
        except: return 0

    def classify_keywords_improved(self):
        cats = {'high_potential':[],'low_potential':[],'wastage':[],'opportunities':[],'future_watch':[]}
        try:
            if self.df is None or len(self.df)==0: return cats
            for _, r in self.df.iterrows():
                try:
                    sp,sa,ro,o,c,cv = safe_float(r.get('Spend',0)),safe_float(r.get('Sales',0)),safe_float(r.get('ROAS',0)),safe_int(r.get('Orders',0)),safe_int(r.get('Clicks',0)),safe_float(r.get('CVR',0))
                    kd = {'Keyword':safe_str(r.get('Customer Search Term')),'Spend':format_currency(sp),'Sales':format_currency(sa),'ROAS':f"{ro:.2f}x",'Orders':o,'Clicks':c,'CVR':f"{cv:.2f}%",'Campaign':safe_str(r.get('Campaign Name')),'Match Type':safe_str(r.get('Match Type')),'Reason':''}
                    if ro>=3.0 and o>=2 and sp>=30 and cv>0: kd['Reason']=f"Champion! ROAS {ro:.2f}x"; cats['high_potential'].append(kd)
                    elif sp>=100 and sa==0 and c>=5: kd['Reason']=f"Rs{sp:.0f} wasted"; cats['wastage'].append(kd)
                    elif sp>=50 and c>=10 and ro<1.5: kd['Reason']=f"Poor ROAS {ro:.2f}x"; cats['low_potential'].append(kd)
                    elif sp>=20 and ro>=1.5 and ro<3.0 and c>=5: kd['Reason']=f"Test optimization"; cats['opportunities'].append(kd)
                    elif c>=5 and sp<=200 and sa==0: kd['Reason']=f"Needs more data"; cats['future_watch'].append(kd)
                except: continue
            return cats
        except: return cats

    def get_future_scale_keywords(self):
        fk = []
        try:
            if self.df is None: return fk
            for _, r in self.df.iterrows():
                try:
                    sp,o,c = safe_float(r.get('Spend',0)),safe_int(r.get('Orders',0)),safe_int(r.get('Clicks',0))
                    if c>=3 and sp<150:
                        if o==1: fk.append({'Keyword':safe_str(r.get('Customer Search Term')),'Match Type':safe_str(r.get('Match Type')),'Clicks':c,'Orders':o,'Spend':format_currency(sp),'Status':'🟡 Promising','Action':'1 order already','Recommendation':'Continue current bid'})
                        elif c>=5 and o==0: fk.append({'Keyword':safe_str(r.get('Customer Search Term')),'Match Type':safe_str(r.get('Match Type')),'Clicks':c,'Orders':o,'Spend':format_currency(sp),'Status':'⚪ Watching','Action':'Needs more data','Recommendation':'Give it more time'})
                except: continue
            return fk
        except: return []

    def get_match_type_strategy(self):
        s = {'current_performance':{},'recommendations':[]}
        try:
            if self.df is None or 'Match Type' not in self.df.columns: return s
            for mt in ['EXACT','PHRASE','BROAD']:
                md = self.df[self.df['Match Type'].str.upper()==mt]
                if len(md)>0:
                    ts,tsa = md['Spend'].sum(),md['Sales'].sum()
                    ro = (tsa/ts) if ts>0 else 0
                    s['current_performance'][mt] = {'spend':ts,'sales':tsa,'roas':ro,'keywords':len(md)}
                    if mt=='EXACT' and ro>=3.0: s['recommendations'].append({'match_type':'EXACT','action':'✅ Scale','reason':f'High ROAS {ro:.2f}x','priority':'HIGH'})
                    elif mt=='PHRASE' and ro>=2.0: s['recommendations'].append({'match_type':'PHRASE','action':'⚡ Test','reason':f'Good ROAS {ro:.2f}x','priority':'MEDIUM'})
                    elif mt=='BROAD' and ro<1.5: s['recommendations'].append({'match_type':'BROAD','action':'⚠️ Reduce','reason':f'Low ROAS {ro:.2f}x','priority':'HIGH'})
            return s
        except: return s

    def get_roas_improvement_plan(self):
        s = self.get_client_summary()
        cr = s['roas']
        c = self.classify_keywords_improved()
        p = {'current_roas':cr,'target_roas':self.target_roas or 3.0,'gap':(self.target_roas or 3.0)-cr,'immediate_actions':[],'short_term':[],'long_term':[]}
        wc = len(c['wastage'])
        if wc>0:
            ws = sum(float(k['Spend'].replace('₹','').replace(',','')) for k in c['wastage'])
            p['immediate_actions'].append({'priority':'🚨 URGENT','action':f'Pause {wc} wastage keywords','impact':f'Save {format_currency(ws)}/month','how':'Exports → Negatives → Upload'})
        hp = len(c['high_potential'])
        if hp>0: p['short_term'].append({'priority':'🏆 HIGH','action':f'Scale {hp} winners','impact':'Increase sales 20-30%','how':'Bids → Increase 15-25%'})
        if cr<1.0: p['immediate_actions'].insert(0,{'priority':'🚨 CRITICAL','action':'Pause ALL campaigns','impact':'Stop losing money','how':'Fix product/pricing first'})
        p['short_term'].append({'priority':'⚡ MEDIUM','action':'Optimize listings','impact':'Improve CVR 50-100%','how':'Better images, A+ content'})
        p['long_term'].append({'priority':'📊 ONGOING','action':'Test new keywords','impact':'Discover new winners','how':'Weekly review, add to EXACT'})
        return p

    def get_bid_suggestions_improved(self):
        sug = []
        try:
            if self.df is None: return sug
            ta, tr = self.target_acos or 30.0, self.target_roas or 3.0
            for _, r in self.df.iterrows():
                try:
                    sp,sa,ro,o,c,cv,cpc = safe_float(r.get('Spend',0)),safe_float(r.get('Sales',0)),safe_float(r.get('ROAS',0)),safe_int(r.get('Orders',0)),safe_int(r.get('Clicks',0)),safe_float(r.get('CVR',0)),safe_float(r.get('CPC',0))
                    if sp<30 or c<5: continue
                    s = {'Keyword':safe_str(r.get('Customer Search Term')),'Campaign':safe_str(r.get('Campaign Name')),'Ad Group':safe_str(r.get('Ad Group Name')),'Match Type':safe_str(r.get('Match Type')),'Current CPC':format_currency(cpc),'Spend':format_currency(sp),'ROAS':f"{ro:.2f}x",'CVR':f"{cv:.2f}%",'Orders':o,'Action':'','Suggested Bid':'','Change (%)':0,'Reason':''}
                    ac = (sp/sa*100) if sa>0 else 999
                    if ro>=3.5 and cv>=2.0 and o>=2: nb=cpc*1.25; s.update({'Action':'INCREASE','Suggested Bid':format_currency(nb),'Change (%)':25,'Reason':f"Champion {ro:.2f}x"})
                    elif ro>=tr and cv>=1.0 and o>=1: nb=cpc*1.15; s.update({'Action':'INCREASE','Suggested Bid':format_currency(nb),'Change (%)':15,'Reason':f"Above target"})
                    elif sa==0 and sp>=100: s.update({'Action':'PAUSE','Suggested Bid':'₹0.00','Change (%)':-100,'Reason':f"Rs{sp:.0f} wasted"})
                    elif ro<1.5 and sp>=50: nb=cpc*0.7; s.update({'Action':'REDUCE','Suggested Bid':format_currency(nb),'Change (%)':-30,'Reason':f"Poor ROAS {ro:.2f}x"})
                    elif ac>ta and sp>=50: red=min(30,(ac-ta)/ta*100); nb=cpc*(1-red/100); s.update({'Action':'REDUCE','Suggested Bid':format_currency(nb),'Change (%)':-int(red),'Reason':f"ACOS {ac:.1f}% high"})
                    else: continue
                    sug.append(s)
                except: continue
            return sorted(sug, key=lambda x: float(x['Spend'].replace('₹','').replace(',','')), reverse=True)
        except: return []

    def get_match_type_performance(self):
        try:
            if self.df is None or 'Match Type' not in self.df.columns: return pd.DataFrame()
            df2 = self.df[self.df['Match Type']!='N/A'].copy()
            if len(df2)==0: return pd.DataFrame()
            mp = df2.groupby('Match Type').agg({'Spend':'sum','Sales':'sum','Orders':'sum','Clicks':'sum','Impressions':'sum'})
            mp['ROAS'] = mp.apply(lambda x: x['Sales']/x['Spend'] if x['Spend']>0 else 0, axis=1)
            mp['ACOS'] = mp.apply(lambda x: x['Spend']/x['Sales']*100 if x['Sales']>0 else 0, axis=1)
            mp['CVR'] = mp.apply(lambda x: x['Orders']/x['Clicks']*100 if x['Clicks']>0 else 0, axis=1)
            mp['CTR'] = mp.apply(lambda x: x['Clicks']/x['Impressions']*100 if x['Impressions']>0 else 0, axis=1)
            return mp
        except: return pd.DataFrame()

    def generate_client_report(self):
        try:
            s = self.get_client_summary()
            h = self.get_health_score()
            c = self.classify_keywords_improved()
            hs = "EXCELLENT" if h>=70 else "GOOD" if h>=50 else "NEEDS ATTENTION"
            tas = f"{self.target_acos:.1f}%" if self.target_acos else "Not Set"
            trs = f"{self.target_roas:.1f}x" if self.target_roas else "Not Set"
            return f"""
=================================================================
AMAZON PPC PERFORMANCE REPORT
Client: {self.client_name}
Date: {datetime.now().strftime('%B %d, %Y')}
=================================================================

Health Score: {h}/100 - {hs}
Target ACOS: {tas} | Target ROAS: {trs}

FINANCIAL PERFORMANCE
-----------------------------------------------------------------
Total Spend:     Rs {s['total_spend']:,.2f}
Total Sales:     Rs {s['total_sales']:,.2f}
Net Profit:      Rs {s['total_profit']:,.2f}
ROAS:            {s['roas']:.2f}x
ACOS:            {s['acos']:.1f}%

METRICS
-----------------------------------------------------------------
Orders:          {s['total_orders']:,}
Clicks:          {s['total_clicks']:,}
CVR:             {s['avg_cvr']:.2f}%
Wastage:         Rs {s['total_wastage']:,.2f}

KEYWORD BREAKDOWN
-----------------------------------------------------------------
High Potential:  {len(c['high_potential'])}
Opportunities:   {len(c['opportunities'])}
Future Watch:    {len(c['future_watch'])}
Low Potential:   {len(c['low_potential'])}
Wastage:         {len(c['wastage'])}

ACTION ITEMS
-----------------------------------------------------------------
1. Pause {len(c['wastage'])} wastage keywords
2. Scale {len(c['high_potential'])} winning keywords
3. Monitor {len(c['future_watch'])} future opportunities
4. Optimize match type strategy

=================================================================
Report by: Amazon Ads Dashboard Pro v2.0
=================================================================
"""
        except Exception as e: return f"Error: {e}"

def init_session_state():
    if 'clients' not in st.session_state: st.session_state.clients = {}
    if 'active_client' not in st.session_state: st.session_state.active_client = None
    if 'agency_name' not in st.session_state: st.session_state.agency_name = "Your Agency"

def render_agency_header():
    st.markdown(f'<div class="agency-header"><h1>🏢 {st.session_state.agency_name}</h1><p>Amazon Ads Dashboard Pro v2.0 - PERFECT Edition</p></div>', unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        with st.expander("⚙️ Settings"):
            nn = st.text_input("Agency Name", value=st.session_state.agency_name)
            if nn != st.session_state.agency_name: st.session_state.agency_name = nn; st.rerun()
        st.markdown("---\n### 👥 Clients")
        if st.session_state.clients:
            cn = list(st.session_state.clients.keys())
            sel = st.selectbox("Active Client", cn)
            st.session_state.active_client = sel
            if sel:
                cl = st.session_state.clients[sel]
                if cl.analyzer and cl.analyzer.df is not None:
                    try:
                        h = cl.analyzer.get_health_score()
                        em = "🟢" if h>=70 else "🟡" if h>=50 else "🔴"
                        st.info(f"{em} Health: {h}/100")
                    except: pass
        st.markdown("---")
        with st.expander("➕ Add Client"):
            nm = st.text_input("Name*")
            ind = st.selectbox("Industry",["E-commerce","Electronics","Fashion","Beauty","Home","Sports","Books","Health","Other"])
            bug = st.number_input("Budget (₹)", value=50000, step=5000)
            st.info("Goals (Optional - 0 = smart defaults)")
            c1,c2 = st.columns(2)
            with c1: tacos = st.number_input("ACOS %", value=0.0, step=5.0)
            with c2: troas = st.number_input("ROAS x", value=0.0, step=0.5)
            em = st.text_input("Email")
            up = st.file_uploader("Upload Report", type=["xlsx","xls"])
            if st.button("✅ Add", type="primary", use_container_width=True):
                if not nm: st.error("Enter name")
                elif not up: st.error("Upload file")
                else:
                    try:
                        with st.spinner(f"Analyzing {nm}..."):
                            df = pd.read_excel(up)
                            st.info(f"Found {len(df)} rows")
                            cd = ClientData(nm, ind, bug)
                            cd.contact_email = em
                            cd.target_acos = tacos if tacos>0 else None
                            cd.target_roas = troas if troas>0 else None
                            cd.analyzer = CompleteAnalyzer(df, nm, cd.target_acos, cd.target_roas)
                            st.session_state.clients[nm] = cd
                            st.session_state.active_client = nm
                            st.success(f"✅ Added {nm}!")
                            st.balloons()
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        with st.expander("Details"): st.code(traceback.format_exc())
        if st.session_state.clients:
            st.markdown("---\n### 📋 All")
            for cn in list(st.session_state.clients.keys()):
                c1,c2 = st.columns([4,1])
                with c1: st.text(f"📊 {cn}")
                with c2:
                    if st.button("❌", key=f"del_{cn}"):
                        del st.session_state.clients[cn]
                        if st.session_state.active_client == cn: st.session_state.active_client = None
                        st.rerun()

def render_dashboard_tab(cl, an):
    try:
        st.header(f"📊 {cl.name} Dashboard")
        s = an.get_client_summary()
        h = an.get_health_score()
        tad = f"{cl.target_acos:.1f}%" if cl.target_acos else "Smart Defaults (30%)"
        trd = f"{cl.target_roas:.1f}x" if cl.target_roas else "Smart Defaults (3.0x)"
        st.markdown(f'<div class="info-box"><h2>Health: {h}/100</h2><p>ACOS: {tad} | ROAS: {trd}</p></div>', unsafe_allow_html=True)
        st.markdown("---\n### 💰 Financial")
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("Spend", format_currency(s['total_spend']))
        with c2: st.metric("Sales", format_currency(s['total_sales']))
        with c3: st.metric("ROAS", f"{s['roas']:.2f}x")
        with c4: st.metric("Orders", format_number(s['total_orders']))
        with c5: st.metric("Profit", format_currency(s['total_profit']))
        st.markdown("---\n### 📈 Metrics")
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("CVR", f"{s['avg_cvr']:.2f}%")
        with c2: st.metric("ACOS", f"{s['acos']:.1f}%")
        with c3: st.metric("Clicks", format_number(s['total_clicks']))
        with c4:
            wp = (s['total_wastage']/s['total_spend']*100) if s['total_spend']>0 else 0
            st.metric("Wastage", f"{format_currency(s['total_wastage'])} ({wp:.1f}%)")
        st.markdown("---\n### 🎯 ROAS Plan")
        p = an.get_roas_improvement_plan()
        st.markdown(f'<div class="info-box"><strong>Current: {p["current_roas"]:.2f}x | Target: {p["target_roas"]:.2f}x | Gap: {p["gap"]:.2f}x</strong></div>', unsafe_allow_html=True)
        if p['immediate_actions']:
            st.markdown("#### 🚨 IMMEDIATE")
            for a in p['immediate_actions']:
                st.markdown(f'<div class="danger-box"><strong>{a["priority"]}: {a["action"]}</strong><br>Impact: {a["impact"]}<br>How: {a["how"]}</div>', unsafe_allow_html=True)
        if p['short_term']:
            st.markdown("#### ⚡ SHORT TERM")
            for a in p['short_term']:
                st.markdown(f'<div class="warning-box"><strong>{a["priority"]}: {a["action"]}</strong><br>Impact: {a["impact"]}<br>How: {a["how"]}</div>', unsafe_allow_html=True)
    except Exception as e: st.error(f"Error: {e}")

def render_keywords_tab(an):
    try:
        st.header("🎯 Keywords")
        c = an.classify_keywords_improved()
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("🏆 High", len(c['high_potential']))
        with c2: st.metric("⚡ Opps", len(c['opportunities']))
        with c3: st.metric("👀 Future", len(c['future_watch']))
        with c4: st.metric("⚠️ Low", len(c['low_potential']))
        with c5: st.metric("🚨 Waste", len(c['wastage']))
        st.markdown("---")
        tabs = st.tabs([f"🏆 Scale ({len(c['high_potential'])})", f"⚡ Test ({len(c['opportunities'])})", f"👀 Watch ({len(c['future_watch'])})", f"⚠️ Low ({len(c['low_potential'])})", f"🚨 Pause ({len(c['wastage'])})"])
        with tabs[0]:
            if c['high_potential']: st.success("✅ Scale 15-25%!"); st.dataframe(pd.DataFrame(c['high_potential']), use_container_width=True, hide_index=True)
            else: st.info("No champions yet")
        with tabs[1]:
            if c['opportunities']: st.info("⚡ Test 10-15%"); st.dataframe(pd.DataFrame(c['opportunities']), use_container_width=True, hide_index=True)
            else: st.info("None")
        with tabs[2]:
            if c['future_watch']:
                st.markdown('<div class="info-box"><strong>👀 Future Watch</strong><br>Relevant keywords need more data</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(c['future_watch']), use_container_width=True, hide_index=True)
                fsk = an.get_future_scale_keywords()
                if fsk: st.markdown("---\n### 🔮 Scale in Future"); st.dataframe(pd.DataFrame(fsk), use_container_width=True, hide_index=True)
            else: st.info("None watching")
        with tabs[3]:
            if c['low_potential']: st.warning("⚠️ Reduce 30%"); st.dataframe(pd.DataFrame(c['low_potential']), use_container_width=True, hide_index=True)
            else: st.success("✅ No low performers")
        with tabs[4]:
            if c['wastage']:
                tw = sum(float(k['Spend'].replace('₹','').replace(',','')) for k in c['wastage'])
                st.error(f"🚨 {format_currency(tw)} wasted on ZERO sales")
                st.markdown('<div class="danger-box"><strong>Wastage = Spend with ZERO sales</strong><br>Exports → Negatives → Upload</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(c['wastage']), use_container_width=True, hide_index=True)
            else: st.success("🎉 No wastage!")
    except Exception as e: st.error(f"Error: {e}")

def render_match_type_tab(an):
    try:
        st.header("📊 Match Type Strategy")
        mp = an.get_match_type_performance()
        if not mp.empty:
            st.subheader("Performance")
            dm = mp.copy()
            dm['Spend'] = dm['Spend'].apply(format_currency)
            dm['Sales'] = dm['Sales'].apply(format_currency)
            dm['ROAS'] = dm['ROAS'].apply(lambda x: f"{x:.2f}x")
            dm['ACOS'] = dm['ACOS'].apply(lambda x: f"{x:.1f}%")
            dm['CVR'] = dm['CVR'].apply(lambda x: f"{x:.2f}%")
            dm['CTR'] = dm['CTR'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(dm, use_container_width=True)
        st.markdown("---\n### 🎯 Strategy")
        s = an.get_match_type_strategy()
        if s.get('recommendations'):
            for r in s['recommendations']:
                bc = "danger-box" if r['priority']=='HIGH' else "warning-box" if r['priority']=='MEDIUM' else "info-box"
                st.markdown(f'<div class="{bc}"><strong>{r["match_type"]}:</strong> {r["action"]}<br>{r["reason"]}</div>', unsafe_allow_html=True)
        st.markdown("---\n### 📚 Guide")
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown('<div class="success-box"><h4>🎯 EXACT</h4>Winners<br>Aggressive bids<br>Example: "blue bottle"</div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="warning-box"><h4>📝 PHRASE</h4>Discovery<br>Moderate bids<br>Example: "water bottle"</div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="info-box"><h4>🌐 BROAD</h4>Research<br>Low bids<br>Example: "bottle"</div>', unsafe_allow_html=True)
    except Exception as e: st.error(f"Error: {e}")

def render_bid_tab(an):
    try:
        st.header("💡 Bid Optimization")
        tad = f"{an.target_acos:.1f}%" if an.target_acos else "30% (default)"
        trd = f"{an.target_roas:.1f}x" if an.target_roas else "3.0x (default)"
        st.markdown(f'<div class="info-box">ACOS: {tad} | ROAS: {trd}</div>', unsafe_allow_html=True)
        sug = an.get_bid_suggestions_improved()
        if sug:
            af = st.selectbox("Filter", ["All","INCREASE","REDUCE","PAUSE"])
            filt = sug if af=="All" else [s for s in sug if af in s['Action']]
            inc = len([s for s in sug if 'INCREASE' in s['Action']])
            red = len([s for s in sug if 'REDUCE' in s['Action']])
            pau = len([s for s in sug if 'PAUSE' in s['Action']])
            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("⬆️ Scale", inc)
            with c2: st.metric("⬇️ Reduce", red)
            with c3: st.metric("⏸️ Pause", pau)
            st.markdown(f"---\n**Showing {len(filt)} of {len(sug)}**")
            st.dataframe(pd.DataFrame(filt), use_container_width=True, hide_index=True, height=500)
        else: st.info("No suggestions")
    except Exception as e: st.error(f"Error: {e}")

def render_exports_tab(an, cn):
    try:
        st.header("📥 Exports")
        c = an.classify_keywords_improved()
        sug = an.get_bid_suggestions_improved()
        c1,c2,c3 = st.columns(3)
        with c1:
            st.subheader("🚫 Negatives")
            w = c['wastage']
            if w:
                nd = [{'Campaign':k['Campaign'],'Ad Group':'','Keyword':k['Keyword'],'Match Type':'Negative Exact','Status':'Enabled'} for k in w]
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr: pd.DataFrame(nd).to_excel(wr, index=False)
                out.seek(0)
                st.download_button(f"📥 Download ({len(nd)})", data=out, file_name=f"Negatives_{cn}_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                st.error(f"🚨 {len(nd)} to pause")
            else: st.success("✅ None needed")
        with c2:
            st.subheader("💰 Bids")
            if sug:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as wr: pd.DataFrame(sug).to_excel(wr, index=False)
                out.seek(0)
                st.download_button(f"📥 Download ({len(sug)})", data=out, file_name=f"Bids_{cn}_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                st.success(f"✅ {len(sug)} suggestions")
            else: st.info("No adjustments")
        with c3:
            st.subheader("📊 Full Data")
            if an.df is not None:
                csv = an.df.to_csv(index=False)
                st.download_button(f"📥 CSV ({len(an.df)} rows)", data=csv, file_name=f"Full_{cn}_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
                st.success("✅ Complete")
    except Exception as e: st.error(f"Error: {e}")

def render_report_tab(cl, an):
    try:
        st.header("📝 Report")
        rep = an.generate_client_report()
        st.text_area("Report", rep, height=600)
        c1,c2 = st.columns(2)
        with c1: st.download_button("📄 TXT", data=rep, file_name=f"Report_{cl.name}_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", use_container_width=True)
        with c2:
            s = an.get_client_summary()
            c = an.classify_keywords_improved()
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
                pd.DataFrame([s]).to_excel(wr, sheet_name='Summary', index=False)
                if c['high_potential']: pd.DataFrame(c['high_potential']).to_excel(wr, sheet_name='High Potential', index=False)
                if c['wastage']: pd.DataFrame(c['wastage']).to_excel(wr, sheet_name='Wastage', index=False)
            out.seek(0)
            st.download_button("📊 Excel", data=out, file_name=f"Report_{cl.name}_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    except Exception as e: st.error(f"Error: {e}")

def render_all_clients_tab():
    try:
        st.header("👥 All Clients")
        if not st.session_state.clients: st.info("No clients"); return
        data = []
        for n, c in st.session_state.clients.items():
            if c.analyzer and c.analyzer.df is not None:
                try:
                    s = c.analyzer.get_client_summary()
                    h = c.analyzer.get_health_score()
                    data.append({'Client':n,'Health':f"{h}/100",'Spend':format_currency(s['total_spend']),'Sales':format_currency(s['total_sales']),'ROAS':f"{s['roas']:.2f}x",'ACOS':f"{s['acos']:.1f}%",'Status':'🟢' if h>=70 else '🟡' if h>=50 else '🔴'})
                except: continue
        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            st.markdown("---\n### 📊 Overview")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Clients", len(data))
            with c2: st.metric("Healthy", len([d for d in data if '🟢' in d['Status']]))
            with c3: st.metric("Attention", len([d for d in data if '🔴' in d['Status']]))
    except Exception as e: st.error(f"Error: {e}")

def render_dashboard():
    render_agency_header()
    if not st.session_state.clients:
        st.markdown('<div class="info-box"><h3>👋 Welcome!</h3><strong>Features:</strong><ul><li>✅ Fixed wastage calculation</li><li>✅ Optional targets</li><li>✅ Future watch keywords</li><li>✅ Match type strategy</li><li>✅ ROAS improvement plan</li></ul><br><strong>Add client from sidebar!</strong></div>', unsafe_allow_html=True)
        return
    if not st.session_state.active_client: st.warning("Select client"); return
    cl = st.session_state.clients[st.session_state.active_client]
    if not cl.analyzer or cl.analyzer.df is None: st.error("No data"); return
    an = cl.analyzer
    tabs = st.tabs(["📊 Dashboard","🎯 Keywords","💡 Bids","📊 Match Types","📝 Report","👥 All","📥 Exports"])
    with tabs[0]: render_dashboard_tab(cl, an)
    with tabs[1]: render_keywords_tab(an)
    with tabs[2]: render_bid_tab(an)
    with tabs[3]: render_match_type_tab(an)
    with tabs[4]: render_report_tab(cl, an)
    with tabs[5]: render_all_clients_tab()
    with tabs[6]: render_exports_tab(an, cl.name)

def main():
    load_custom_css()
    init_session_state()
    render_sidebar()
    render_dashboard()
    st.markdown(f'<div style="text-align:center;color:#94a3b8;padding:1rem;"><strong>{st.session_state.agency_name}</strong><br>Amazon Ads Dashboard Pro v2.0 - PERFECT Edition<br><small>Fixed Wastage | Optional Targets | Future Insights</small></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
