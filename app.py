class MultiAccountPPCManager:
    def __init__(self):
        self.accounts = {}
    
    def add_account(self, account_name, file_path, target_acos=35):
        """Add account for analysis"""
        self.accounts[account_name] = AmazonPPCAnalyzer(file_path, target_acos)
    
    def analyze_all_accounts(self):
        """Run analysis for all accounts"""
        for account_name, analyzer in self.accounts.items():
            print(f"\n{'='*80}")
            print(f"ANALYZING ACCOUNT: {account_name}")
            print(f"{'='*80}")
            analyzer.generate_full_report()
    
    def export_combined_negatives(self, output_file='all_accounts_negatives.csv'):
        """Combine negative keywords from all accounts"""
        all_negatives = []
        for account_name, analyzer in self.accounts.items():
            negatives = analyzer.get_negative_keywords()
            negatives['Account'] = account_name
            all_negatives.append(negatives)
        
        combined = pd.concat(all_negatives, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"✅ Combined negative keywords exported: {output_file}")

# Usage for multiple accounts
manager = MultiAccountPPCManager()
manager.add_account('Account_1', 'account1_report.xlsx', target_acos=30)
manager.add_account('Account_2', 'account2_report.xlsx', target_acos=40)
manager.analyze_all_accounts()
