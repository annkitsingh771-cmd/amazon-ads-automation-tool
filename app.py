# ============================================================================
# AMAZON PPC AUTOMATION TOOL - Complete Solution
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

class AmazonPPCAnalyzer:
    def __init__(self, file_path, target_acos=35):
        """
        Initialize the Amazon PPC Analyzer
        
        Args:
            file_path: Path to search term report Excel file
            target_acos: Target ACOS percentage (default 35%)
        """
        self.file_path = file_path
        self.target_acos = target_acos
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and clean the search term report"""
        self.df = pd.read_excel(self.file_path)
        self.df.columns = self.df.columns.str.strip()
        
        # Handle missing values
        self.df['7 Day Total Sales (₹)'] = self.df['7 Day Total Sales (₹)'].fillna(0)
        self.df['Total Advertising Cost of Sales (ACOS)'] = self.df['Total Advertising Cost of Sales (ACOS)'].fillna(0)
        self.df['Total Return on Advertising Spend (ROAS)'] = self.df['Total Return on Advertising Spend (ROAS)'].fillna(0)
        
        # Calculate additional metrics
        self.df['Wastage'] = self.df.apply(
            lambda x: x['Spend'] if x['7 Day Total Sales (₹)'] == 0 else 0, axis=1
        )
        
        print(f"✅ Loaded {len(self.df)} search terms from {self.df['Start Date'].min()} to {self.df['End Date'].max()}")
        
    def calculate_overall_metrics(self):
        """Calculate campaign-level metrics"""
        total_revenue = self.df['7 Day Total Sales (₹)'].sum()
        total_spend = self.df['Spend'].sum()
        total_clicks = self.df['Clicks'].sum()
        total_impressions = self.df['Impressions'].sum()
        total_orders = self.df['7 Day Total Orders (#)'].sum()
        total_wastage = self.df['Wastage'].sum()
        
        metrics = {
            'total_spend': total_spend,
            'total_sales': total_revenue,
            'overall_roas': total_revenue/total_spend if total_spend > 0 else 0,
            'overall_acos': (total_spend/total_revenue*100) if total_revenue > 0 else 0,
            'overall_tacos': (total_spend/total_revenue*100) if total_revenue > 0 else 0,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'avg_ctr': (total_clicks/total_impressions*100) if total_impressions > 0 else 0,
            'avg_cpc': total_spend/total_clicks if total_clicks > 0 else 0,
            'total_orders': total_orders,
            'conversion_rate': (total_orders/total_clicks*100) if total_clicks > 0 else 0,
            'total_wastage': total_wastage,
            'wastage_percentage': (total_wastage/total_spend*100) if total_spend > 0 else 0,
            'keywords_with_zero_sales': len(self.df[self.df['7 Day Total Sales (₹)'] == 0])
        }
        
        return metrics
        
    def classify_keywords(self):
        """Classify keywords into categories"""
        
        def classify(row):
            spend = row['Spend']
            sales = row['7 Day Total Sales (₹)']
            clicks = row['Clicks']
            roas = row['Total Return on Advertising Spend (ROAS)']
            acos = row['Total Advertising Cost of Sales (ACOS)']
            ctr = row['Click-Through Rate (CTR)']
            conversion_rate = row['7 Day Conversion Rate']
            orders = row['7 Day Total Orders (#)']
            
            # HIGH POTENTIAL - Great performers
            if (roas >= 2.5 or (acos > 0 and acos <= 40)) and orders >= 2 and clicks >= 5:
                return 'High Potential'
            
            # FUTURE POTENTIAL - Show promise
            elif (orders >= 1 and clicks <= 10) or (ctr >= 0.1 and sales == 0 and clicks <= 5):
                return 'Future Potential'
            
            # NEGATIVE - Clear waste
            elif (clicks >= 5 and sales == 0) or (clicks >= 10 and sales == 0) or (acos > 200 and orders == 0):
                return 'Negative'
            
            # LOW POTENTIAL - Underperformers
            elif (acos > 100 and orders > 0) or (clicks >= 3 and sales == 0):
                return 'Low Potential'
            
            else:
                return 'Monitor'
        
        self.df['Keyword_Category'] = self.df.apply(classify, axis=1)
        return self.df['Keyword_Category'].value_counts()
        
    def is_irrelevant_keyword(self, search_term):
        """Detect irrelevant keywords using pattern matching"""
        search_term_lower = str(search_term).lower()
        
        # Customize these patterns for your product category
        irrelevant_patterns = [
            r'badminton|racket|racquet|shuttlecock|sports',
            r'b0[0-9a-z]{8}',  # ASIN codes
            r'electric|battery|usb|rechargeable|sonic|oscillating',
            r'dispenser|holder|stand|rack|storage',
            r'travel case|cover|cap|pouch',
            r'whitening strips|mouthwash|floss|dental picks',
        ]
        
        for pattern in irrelevant_patterns:
            if re.search(pattern, search_term_lower):
                return True
        return False
    
    def generate_bid_suggestions(self):
        """Generate bid adjustment recommendations"""
        
        def suggest_bid(row):
            current_cpc = row['Cost Per Click (CPC)']
            acos = row['Total Advertising Cost of Sales (ACOS)']
            roas = row['Total Return on Advertising Spend (ROAS)']
            category = row['Keyword_Category']
            
            if category == 'High Potential':
                if roas >= 3:
                    bid_change = 25
                else:
                    bid_change = 15
                action = 'INCREASE'
                suggested_cpc = current_cpc * (1 + bid_change/100)
                reason = f"Strong performer (ROAS: {roas:.2f}x)"
                
            elif category == 'Future Potential':
                bid_change = 10
                action = 'INCREASE'
                suggested_cpc = current_cpc * 1.10
                reason = "Shows promise, increase for more data"
                
            elif category == 'Low Potential':
                bid_change = -30
                action = 'DECREASE'
                suggested_cpc = current_cpc * 0.70
                reason = f"Underperforming (ACOS: {acos:.1f}%)"
                
            elif category == 'Negative':
                bid_change = -100
                action = 'PAUSE/NEGATIVE'
                suggested_cpc = 0
                reason = "No conversions, add as negative"
                
            else:
                bid_change = 0
                action = 'MAINTAIN'
                suggested_cpc = current_cpc
                reason = "Monitor performance"
            
            return pd.Series({
                'Bid_Action': action,
                'Bid_Change_%': bid_change,
                'Suggested_CPC': suggested_cpc,
                'Reason': reason
            })
        
        bid_suggestions = self.df.apply(suggest_bid, axis=1)
        self.df = pd.concat([self.df, bid_suggestions], axis=1)
        
    def get_negative_keywords(self):
        """Identify negative keyword candidates"""
        self.df['Is_Irrelevant'] = self.df['Customer Search Term'].apply(self.is_irrelevant_keyword)
        
        negative_candidates = self.df[
            ((self.df['Keyword_Category'] == 'Negative') | 
             (self.df['Keyword_Category'] == 'Low Potential') | 
             (self.df['Is_Irrelevant'] == True)) &
            (self.df['Clicks'] >= 2)
        ].copy()
        
        return negative_candidates.sort_values('Spend', ascending=False)
    
    def export_negative_keywords_bulk_file(self, output_file='amazon_negative_keywords_bulk.csv'):
        """
        Generate Amazon Ads bulk upload file for negative keywords
        Ready to upload directly to Amazon without editing
        """
        negative_keywords = self.get_negative_keywords()
        
        bulk_data = []
        for idx, row in negative_keywords.iterrows():
            # Determine match type for negative
            if row['Match Type'] in ['BROAD', 'PHRASE']:
                negative_match = 'NEGATIVE_PHRASE'
            else:
                negative_match = 'NEGATIVE_EXACT'
            
            bulk_data.append({
                'Campaign Name': row['Campaign Name'],
                'Ad Group Name': row['Ad Group Name'],
                'Keyword Text': row['Customer Search Term'],
                'Match Type': negative_match,
                'Status': 'ENABLED'
            })
        
        bulk_df = pd.DataFrame(bulk_data)
        bulk_df.to_csv(output_file, index=False)
        print(f"✅ Negative keywords bulk file created: {output_file}")
        print(f"   Total negative keywords: {len(bulk_df)}")
        print(f"   Potential savings: ₹{negative_keywords['Spend'].sum():,.2f}")
        print(f"\n📤 Upload this file directly to Amazon Ads > Bulk Operations")
        
        return bulk_df
    
    def export_high_potential_keywords(self, output_file='high_potential_keywords.csv'):
        """Export high-performing keywords for new campaigns"""
        high_potential = self.df[self.df['Keyword_Category'] == 'High Potential'].copy()
        
        export_df = high_potential[[
            'Customer Search Term', 'Campaign Name', 'Ad Group Name',
            'Impressions', 'Clicks', 'Click-Through Rate (CTR)', 'Spend',
            '7 Day Total Sales (₹)', 'Total Return on Advertising Spend (ROAS)',
            'Total Advertising Cost of Sales (ACOS)', '7 Day Total Orders (#)',
            'Cost Per Click (CPC)', 'Suggested_CPC', 'Bid_Action', 'Reason'
        ]].sort_values('Total Return on Advertising Spend (ROAS)', ascending=False)
        
        export_df.to_csv(output_file, index=False)
        print(f"✅ High potential keywords exported: {output_file}")
        return export_df
    
    def export_future_potential_keywords(self, output_file='future_potential_keywords.csv'):
        """Export keywords that show promise"""
        future_potential = self.df[self.df['Keyword_Category'] == 'Future Potential'].copy()
        
        export_df = future_potential[[
            'Customer Search Term', 'Campaign Name', 'Ad Group Name',
            'Impressions', 'Clicks', 'Spend', '7 Day Total Sales (₹)',
            '7 Day Total Orders (#)', 'Suggested_CPC', 'Reason'
        ]].sort_values('7 Day Total Sales (₹)', ascending=False)
        
        export_df.to_csv(output_file, index=False)
        print(f"✅ Future potential keywords exported: {output_file}")
        return export_df
    
    def generate_campaign_recommendations(self):
        """Generate new campaign structure recommendations"""
        recommendations = []
        
        # Exact match campaigns for high performers
        high_potential = self.df[self.df['Keyword_Category'] == 'High Potential']
        if len(high_potential) > 0:
            recommendations.append({
                'Campaign Type': 'Exact Match - High Performers',
                'Keywords': len(high_potential),
                'Avg ROAS': high_potential['Total Return on Advertising Spend (ROAS)'].mean(),
                'Suggested Budget': high_potential['Spend'].sum() * 1.5,
                'Action': 'Create new exact match campaign with these proven converters'
            })
        
        # Broad match for discovery
        future_potential = self.df[self.df['Keyword_Category'] == 'Future Potential']
        if len(future_potential) > 5:
            recommendations.append({
                'Campaign Type': 'Broad Match - Discovery',
                'Keywords': len(future_potential),
                'Avg ROAS': future_potential['Total Return on Advertising Spend (ROAS)'].mean(),
                'Suggested Budget': future_potential['Spend'].sum() * 0.8,
                'Action': 'Test with lower bids to discover new opportunities'
            })
        
        return pd.DataFrame(recommendations)
    
    def generate_full_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("AMAZON PPC AUTOMATION TOOL - FULL ANALYSIS REPORT")
        print("="*80)
        
        # Overall Metrics
        metrics = self.calculate_overall_metrics()
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Total Spend: ₹{metrics['total_spend']:,.2f}")
        print(f"   Total Sales: ₹{metrics['total_sales']:,.2f}")
        print(f"   Overall ROAS: {metrics['overall_roas']:.2f}x")
        print(f"   Overall ACOS: {metrics['overall_acos']:.2f}%")
        print(f"   Overall TACOS: {metrics['overall_tacos']:.2f}%")
        print(f"   Total Orders: {int(metrics['total_orders'])}")
        print(f"   Conversion Rate: {metrics['conversion_rate']:.2f}%")
        print(f"   Average CPC: ₹{metrics['avg_cpc']:.2f}")
        print(f"   Average CTR: {metrics['avg_ctr']:.2f}%")
        
        print(f"\n🔴 WASTAGE ALERT:")
        print(f"   Wasted Spend: ₹{metrics['total_wastage']:,.2f} ({metrics['wastage_percentage']:.1f}%)")
        print(f"   Keywords with Zero Sales: {metrics['keywords_with_zero_sales']}")
        
        # Classification
        self.classify_keywords()
        category_counts = self.df['Keyword_Category'].value_counts()
        category_spend = self.df.groupby('Keyword_Category')['Spend'].sum()
        category_sales = self.df.groupby('Keyword_Category')['7 Day Total Sales (₹)'].sum()
        
        print(f"\n🎯 KEYWORD CLASSIFICATION:")
        for category in ['High Potential', 'Future Potential', 'Monitor', 'Low Potential', 'Negative']:
            if category in category_counts.index:
                count = category_counts[category]
                spend = category_spend[category]
                sales = category_sales[category]
                roas = sales/spend if spend > 0 else 0
                print(f"\n   {category.upper()}:")
                print(f"      Keywords: {count}")
                print(f"      Spend: ₹{spend:,.2f}")
                print(f"      Sales: ₹{sales:,.2f}")
                print(f"      ROAS: {roas:.2f}x")
        
        # Bid Suggestions
        self.generate_bid_suggestions()
        
        # Top performers
        print(f"\n🟢 TOP 5 HIGH PERFORMERS (Increase Bids):")
        high_potential = self.df[self.df['Keyword_Category'] == 'High Potential'].sort_values('Total Return on Advertising Spend (ROAS)', ascending=False).head(5)
        for idx, row in high_potential.iterrows():
            print(f"\n   '{row['Customer Search Term']}'")
            print(f"      ROAS: {row['Total Return on Advertising Spend (ROAS)']:.2f}x | Orders: {int(row['7 Day Total Orders (#)'])}")
            print(f"      Current CPC: ₹{row['Cost Per Click (CPC)']:.2f} → Suggested: ₹{row['Suggested_CPC']:.2f}")
        
        # Negative keywords
        negative_keywords = self.get_negative_keywords()
        print(f"\n🔴 TOP 5 NEGATIVE KEYWORD CANDIDATES:")
        for idx, row in negative_keywords.head(5).iterrows():
            print(f"\n   '{row['Customer Search Term']}'")
            print(f"      Wasted: ₹{row['Spend']:.2f} | Clicks: {int(row['Clicks'])} | Sales: ₹0.00")
            if row['Is_Irrelevant']:
                print(f"      ⚠️  IRRELEVANT TERM")
        
        print(f"\n💾 GENERATING EXPORT FILES...")
        
        # Export files
        self.export_negative_keywords_bulk_file()
        self.export_high_potential_keywords()
        self.export_future_potential_keywords()
        
        # Campaign recommendations
        recommendations = self.generate_campaign_recommendations()
        if len(recommendations) > 0:
            print(f"\n💡 CAMPAIGN RECOMMENDATIONS:")
            for idx, rec in recommendations.iterrows():
                print(f"\n   {rec['Campaign Type']}:")
                print(f"      Keywords: {rec['Keywords']}")
                print(f"      Avg ROAS: {rec['Avg ROAS']:.2f}x")
                print(f"      Suggested Budget: ₹{rec['Suggested Budget']:,.2f}")
                print(f"      Action: {rec['Action']}")
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AmazonPPCAnalyzer(
        file_path='Sponsored_Products_Search_term_report-8686.xlsx',
        target_acos=35  # Adjust your target ACOS
    )
    
    # Generate full report and export files
    analyzer.generate_full_report()
    
    # Access data for custom analysis
    df_with_analysis = analyzer.df
    print(f"\n📁 Full data with analysis saved in 'df_with_analysis' variable")
