import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import glob
from pathlib import Path

class LEAPSAnalyzer:
    def __init__(self, csv_data_path):
        """
        Initialize LEAPS analyzer for January LEAPS contracts (20240119 expiry only)
        
        Args:
            csv_data_path: Path to options_leaps_csv folder
        """
        self.csv_data_path = csv_data_path
        self.target_opp = 5.0  # Target 5% OPP
        self.target_expiry = '20240119'  # Fixed expiry date - January 19, 2024
        self.target_dte_range = (360, 370)  # Look for contracts around 365 DTE
        
        # Assets to analyze
        self.assets = ['AAPL', 'AMZN', 'WMT', 'COST', 'KO', 'PG', 'SPY']
        
        # Data storage
        self.selected_contracts = {}
        self.portfolio_data = None
        
        print(f"LEAPS Analyzer initialized")
        print(f"Data path: {self.csv_data_path}")
        print(f"Target expiry: {self.target_expiry} (January 19, 2024)")
        print(f"Target OPP: {self.target_opp}%")
        print(f"Target DTE range: {self.target_dte_range}")
        print(f"Assets to analyze: {self.assets}")
    
    def validate_data_path(self):
        """Validate that the data path exists and contains expected structure"""
        if not os.path.exists(self.csv_data_path):
            print(f"❌ Error: Data path does not exist: {self.csv_data_path}")
            return False
        
        # Check for at least one asset folder with the target expiry
        asset_found = False
        for asset in self.assets:
            asset_path = os.path.join(self.csv_data_path, asset)
            expiry_path = os.path.join(asset_path, self.target_expiry)
            if os.path.exists(expiry_path):
                asset_found = True
                break
        
        if not asset_found:
            print(f"❌ Error: No asset folders with {self.target_expiry} expiry found in {self.csv_data_path}")
            return False
        
        print(f"✅ Data path validation passed")
        return True
    
    def find_available_strikes_for_expiry(self, asset):
        """
        Find all available strikes for an asset with the target expiry
        Returns list of strikes
        """
        expiry_path = os.path.join(self.csv_data_path, asset, self.target_expiry)
        if not os.path.exists(expiry_path):
            return []
        
        strikes = []
        
        # Find all CSV files in the expiry folder
        csv_files = glob.glob(os.path.join(expiry_path, f"{asset}_{self.target_expiry}_C_*.csv"))
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            try:
                # Extract strike from filename: ASSET_EXPIRY_C_STRIKE.csv
                strike_str = filename.split('_')[-1].replace('.csv', '')
                strike = float(strike_str)
                strikes.append(strike)
            except (ValueError, IndexError):
                continue
        
        return sorted(strikes)
    
    def load_contract_data(self, asset, strike):
        """Load a single contract's data from CSV for the target expiry"""
        filename = f"{asset}_{self.target_expiry}_C_{int(strike)}.csv"
        file_path = os.path.join(self.csv_data_path, asset, self.target_expiry, filename)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate DTE if not present
            expiry_datetime = pd.to_datetime(self.target_expiry, format='%Y%m%d')
            df['dte'] = (expiry_datetime - df['date']).dt.days
            
            # Clean up OPP column - use pre-calculated OPP if available
            if 'OPP' in df.columns:
                df['opp'] = df['OPP']
            elif 'opp' not in df.columns:
                print(f"⚠️ Warning: No OPP column found in {filename}")
                return None
            
            # Remove rows with invalid OPP or DTE
            df = df.dropna(subset=['opp', 'dte'])
            df = df[df['opp'] > 0]  # OPP should be positive
            df = df[df['dte'] >= 0]  # DTE should be non-negative
            
            # Add metadata
            df['asset'] = asset
            df['expiry'] = self.target_expiry
            df['strike'] = strike
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def evaluate_contract_at_target_dte(self, contract_df, target_dte_range):
        """
        Evaluate a contract's OPP at the target DTE range
        Returns dict with evaluation metrics or None if not suitable
        """
        if contract_df is None or len(contract_df) == 0:
            return None
        
        # Filter to target DTE range
        target_data = contract_df[
            (contract_df['dte'] >= target_dte_range[0]) & 
            (contract_df['dte'] <= target_dte_range[1])
        ]
        
        if len(target_data) == 0:
            return None
        
        # Find the point closest to 365 DTE
        target_dte = 365
        closest_idx = (target_data['dte'] - target_dte).abs().idxmin()
        closest_point = target_data.loc[closest_idx]
        
        evaluation = {
            'opp_at_target': closest_point['opp'],
            'dte_at_target': closest_point['dte'],
            'date_at_target': closest_point['date'],
            'diff_from_target_opp': abs(closest_point['opp'] - self.target_opp),
            'data_points': len(contract_df),
            'target_range_points': len(target_data)
        }
        
        return evaluation
    
    def find_best_contract_for_asset(self, asset):
        """
        Find the best contract for an asset based on target OPP criteria (20240119 expiry only)
        """
        print(f"\n🔍 Analyzing {asset} ({self.target_expiry} expiry)...")
        
        # Find all available strikes for the target expiry
        available_strikes = self.find_available_strikes_for_expiry(asset)
        
        if not available_strikes:
            print(f"  ❌ No contracts found for {asset} with {self.target_expiry} expiry")
            return None
        
        print(f"  📊 Found {len(available_strikes)} strikes for {self.target_expiry} expiry")
        
        best_contract = None
        best_score = float('inf')
        total_contracts_evaluated = 0
        
        # Evaluate each strike
        for strike in available_strikes:
            total_contracts_evaluated += 1
            
            # Load contract data
            contract_df = self.load_contract_data(asset, strike)
            if contract_df is None:
                continue
            
            # Evaluate contract at target DTE
            evaluation = self.evaluate_contract_at_target_dte(contract_df, self.target_dte_range)
            if evaluation is None:
                continue
            
            # Check if this is the best contract so far
            score = evaluation['diff_from_target_opp']
            if score < best_score:
                best_score = score
                best_contract = {
                    'asset': asset,
                    'expiry': self.target_expiry,
                    'strike': strike,
                    'evaluation': evaluation,
                    'data': contract_df
                }
            
            # Print evaluation for this contract
            print(f"    💰 Strike ${strike:>6.0f}: OPP={evaluation['opp_at_target']:>5.2f}% "
                  f"at {evaluation['dte_at_target']:>3.0f} DTE, "
                  f"diff={evaluation['diff_from_target_opp']:>5.2f}%")
        
        # Report results
        if best_contract:
            eval_data = best_contract['evaluation']
            print(f"  ✅ SELECTED: ${best_contract['strike']:.0f} strike, {self.target_expiry} expiry")
            print(f"     📈 OPP: {eval_data['opp_at_target']:.2f}% at {eval_data['dte_at_target']:.0f} DTE")
            print(f"     📊 Data points: {eval_data['data_points']} total, {eval_data['target_range_points']} in target range")
            print(f"     🎯 Diff from target: {eval_data['diff_from_target_opp']:.2f}%")
        else:
            print(f"  ❌ No suitable contract found (evaluated {total_contracts_evaluated} contracts)")
        
        return best_contract
    
    def select_all_contracts(self):
        """Select the best contract for each asset (20240119 expiry only)"""
        print("\n" + "="*80)
        print("🔍 CONTRACT SELECTION PROCESS")
        print(f"Target Expiry: {self.target_expiry} (January 19, 2024)")
        print("="*80)
        
        if not self.validate_data_path():
            return False
        
        for asset in self.assets:
            contract = self.find_best_contract_for_asset(asset)
            if contract:
                self.selected_contracts[asset] = contract
        
        print(f"\n📋 SELECTION SUMMARY:")
        print(f"   Assets analyzed: {len(self.assets)}")
        print(f"   Contracts selected: {len(self.selected_contracts)}")
        print(f"   Success rate: {len(self.selected_contracts)/len(self.assets)*100:.1f}%")
        
        if self.selected_contracts:
            print(f"\n✅ Selected contracts (all {self.target_expiry}):")
            for asset, contract in self.selected_contracts.items():
                eval_data = contract['evaluation']
                print(f"   {asset}: ${contract['strike']:.0f} - "
                      f"OPP={eval_data['opp_at_target']:.2f}%")
            return True
        else:
            print("❌ No contracts selected. Cannot proceed with analysis.")
            return False

    def filter_to_last_365_days(self):
        """Filter each selected contract to last 365 days (DTE <= 365)"""
        print("\n" + "="*80)
        print("📅 FILTERING TO LAST 365 DAYS")
        print("="*80)
        
        filtered_contracts = {}
        
        for asset, contract in self.selected_contracts.items():
            full_data = contract['data']
            
            # Filter to DTE <= 365 (last 365 days before expiry)
            last_365 = full_data[full_data['dte'] <= 365].copy()
            last_365 = last_365.sort_values('date').reset_index(drop=True)
            
            if len(last_365) > 0:
                contract['filtered_data'] = last_365
                filtered_contracts[asset] = contract
                
                print(f"✅ {asset}: {len(last_365):>3} data points "
                      f"({last_365['dte'].max():>3.0f} to {last_365['dte'].min():>3.0f} DTE)")
            else:
                print(f"❌ {asset}: No data in last 365 days")
        
        self.selected_contracts = filtered_contracts
        print(f"\n📊 Kept {len(self.selected_contracts)} contracts with valid 365-day data")
    
    def create_portfolio_timeline(self):
        """Create unified timeline and calculate portfolio returns"""
        print("\n" + "="*80)
        print("💼 PORTFOLIO CONSTRUCTION")
        print("="*80)
        
        if not self.selected_contracts:
            print("❌ No contracts available for portfolio construction")
            return None
        
        # Find union of all dates
        all_dates = set()
        for asset, contract in self.selected_contracts.items():
            if 'filtered_data' not in contract:
                continue
            asset_dates = set(contract['filtered_data']['date'])
            all_dates = all_dates.union(asset_dates)
        
        all_dates = sorted(list(all_dates))
        print(f"📅 Portfolio timeline: {len(all_dates)} days")
        print(f"   From: {min(all_dates).strftime('%Y-%m-%d')}")
        print(f"   To:   {max(all_dates).strftime('%Y-%m-%d')}")
        print(f"   All contracts expire: {self.target_expiry}")
        
        # Create base portfolio DataFrame
        portfolio_df = pd.DataFrame({'date': all_dates})
        
        # Add each asset's data
        initial_investment_per_asset = 10000  # $10k per asset
        positions = {}
        total_initial_investment = 0
        
        print(f"\n💰 POSITION SIZING (${initial_investment_per_asset:,} per asset):")
        
        for asset, contract in self.selected_contracts.items():
            if 'filtered_data' not in contract:
                continue
            
            asset_data = contract['filtered_data'].set_index('date')
            
            # Create series for all dates (forward fill missing data)
            asset_timeline = pd.DataFrame(index=pd.to_datetime(all_dates))
            
            # Find the price column
            price_col = None
            for col in ['close', 'real_price', 'interpolated_price', 'options_price']:
                if col in asset_data.columns:
                    price_col = col
                    break
            
            if price_col is None:
                print(f"❌ {asset}: No price column found")
                continue
            
            # Reindex and forward fill
            for col in [price_col, 'opp', 'dte']:
                if col in asset_data.columns:
                    series = asset_data[col].reindex(asset_timeline.index)
                    series = series.fillna(method='ffill').fillna(method='bfill')
                    
                    if col == price_col:
                        asset_timeline['price'] = series
                    else:
                        asset_timeline[col] = series
            
            # Calculate position sizing
            first_price = asset_timeline['price'].dropna().iloc[0]
            if pd.isna(first_price) or first_price <= 0:
                print(f"❌ {asset}: Invalid first price")
                continue
            
            # Each option contract represents 100 shares
            num_contracts = initial_investment_per_asset / (first_price * 100)
            position_value = num_contracts * first_price * 100
            
            positions[asset] = {
                'contracts': num_contracts,
                'initial_price': first_price,
                'initial_investment': position_value,
                'strike': contract['strike'],
                'expiry': self.target_expiry
            }
            total_initial_investment += position_value
            
            # Add to portfolio DataFrame
            portfolio_df[f'{asset}_price'] = asset_timeline['price'].values
            portfolio_df[f'{asset}_opp'] = asset_timeline['opp'].values
            portfolio_df[f'{asset}_dte'] = asset_timeline['dte'].values
            
            # Calculate position values and returns
            position_values = asset_timeline['price'] * num_contracts * 100
            returns = ((asset_timeline['price'] / first_price) - 1) * 100
            
            portfolio_df[f'{asset}_value'] = position_values.values
            portfolio_df[f'{asset}_return'] = returns.values
            
            print(f"   {asset}: {num_contracts:.3f} contracts @ ${first_price:.2f} = ${position_value:>8,.0f}")
        
        # Calculate portfolio totals
        value_columns = [col for col in portfolio_df.columns if col.endswith('_value')]
        portfolio_df['total_value'] = portfolio_df[value_columns].sum(axis=1)
        portfolio_df['total_return'] = ((portfolio_df['total_value'] / total_initial_investment) - 1) * 100
        
        # Add DTE (use first asset's DTE as reference)
        dte_columns = [col for col in portfolio_df.columns if col.endswith('_dte')]
        if dte_columns:
            portfolio_df['dte'] = portfolio_df[dte_columns[0]]
        
        self.portfolio_data = {
            'df': portfolio_df,
            'positions': positions,
            'initial_investment': total_initial_investment
        }
        
        print(f"\n📊 Portfolio Summary:")
        print(f"   Total initial investment: ${total_initial_investment:,.0f}")
        print(f"   Number of positions: {len(positions)}")
        print(f"   Portfolio data points: {len(portfolio_df)}")
        print(f"   Common expiry: {self.target_expiry}")
        
        return portfolio_df
    
    def create_output_directory(self):
        """Create output directory for plots"""
        output_dir = f'leaps_analysis_output_{self.target_expiry}'
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def plot_individual_assets(self):
        """Create individual OPP plots for each asset"""
        print("\n📊 Creating individual asset plots...")
        output_dir = self.create_output_directory()
        
        for asset in self.selected_contracts.keys():
            plt.figure(figsize=(12, 7))
            
            contract = self.selected_contracts[asset]
            if 'filtered_data' not in contract:
                continue
            
            data = contract['filtered_data']
            
            # Plot OPP over time (DTE on x-axis, reversed)
            plt.plot(data['dte'], data['opp'], linewidth=2, color='blue', alpha=0.8)
            plt.axhline(y=self.target_opp, color='red', linestyle='--', alpha=0.7, 
                       label=f'{self.target_opp}% Target')
            
            # Mark entry point
            eval_data = contract['evaluation']
            plt.scatter([eval_data['dte_at_target']], [eval_data['opp_at_target']], 
                       color='green', s=100, label='Entry Point', zorder=5)
            
            # Statistics box
            stats = {
                'Asset': asset,
                'Expiry': self.target_expiry,
                'Strike': f"${contract['strike']:.0f}",
                'Entry OPP': f"{eval_data['opp_at_target']:.2f}%",
                'Entry DTE': f"{eval_data['dte_at_target']:.0f}",
                'OPP Range': f"{data['opp'].min():.2f}% - {data['opp'].max():.2f}%",
                'Data Points': f"{len(data)}"
            }
            
            stats_text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.title(f'{asset} - Options Price Percentage (OPP) - {self.target_expiry} Expiry')
            plt.xlabel('Days to Expiry')
            plt.ylabel('OPP (%)')
            plt.xlim(data['dte'].max(), data['dte'].min())  # Reverse x-axis
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            filename = os.path.join(output_dir, f'{asset}_opp_analysis.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close to save memory
            
        print(f"   ✅ Saved individual plots to {output_dir}/")
    
    def plot_portfolio_overlay(self):
        """Create overlay plot of all assets"""
        print("📊 Creating portfolio overlay plot...")
        output_dir = self.create_output_directory()
        
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.selected_contracts)))
        
        for i, (asset, contract) in enumerate(self.selected_contracts.items()):
            if 'filtered_data' not in contract:
                continue
            
            data = contract['filtered_data']
            plt.plot(data['dte'], data['opp'], linewidth=2, 
                    label=f'{asset} (${contract["strike"]:.0f})', alpha=0.8, color=colors[i])
        
        plt.axhline(y=self.target_opp, color='red', linestyle='--', 
                   alpha=0.7, label=f'{self.target_opp}% Target')
        
        plt.title(f'Options Price Percentage (OPP) - All Assets ({self.target_expiry} Expiry)')
        plt.xlabel('Days to Expiry')
        plt.ylabel('OPP (%)')
        plt.xlim(365, 0)  # Reverse x-axis
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'portfolio_opp_overlay.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved overlay plot")
    
    def plot_portfolio_returns(self):
        """Create portfolio returns plots"""
        print("📊 Creating portfolio returns plots...")
        output_dir = self.create_output_directory()
        
        if self.portfolio_data is None:
            print("   ❌ No portfolio data available")
            return
        
        portfolio_df = self.portfolio_data['df']
        
        # Portfolio performance plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Portfolio value over time
        ax1.plot(portfolio_df['dte'], portfolio_df['total_value'], 
                linewidth=2, color='navy', label='Portfolio Value')
        ax1.axhline(y=self.portfolio_data['initial_investment'], 
                   color='red', linestyle='--', alpha=0.7, label='Initial Investment')
        ax1.set_title(f'Portfolio Value Over Time ({self.target_expiry} Expiry)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Portfolio returns
        ax2.plot(portfolio_df['dte'], portfolio_df['total_return'], 
                linewidth=2, color='green', label='Portfolio Return')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
        ax2.set_title('Portfolio Returns (%)')
        ax2.set_xlabel('Days to Expiry')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add final return annotation
        final_return = portfolio_df['total_return'].iloc[-1]
        ax2.text(0.02, 0.98, f'Final Return: {final_return:.1f}%',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Set x-axis limits (reverse)
        for ax in [ax1, ax2]:
            ax.set_xlim(portfolio_df['dte'].max(), portfolio_df['dte'].min())
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'portfolio_returns.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved portfolio returns plot")
    
    def plot_individual_returns(self):
        """Plot individual asset returns"""
        print("📊 Creating individual returns plot...")
        output_dir = self.create_output_directory()
        
        if self.portfolio_data is None:
            return
        
        portfolio_df = self.portfolio_data['df']
        
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.selected_contracts)))
        
        for i, asset in enumerate(self.selected_contracts.keys()):
            return_col = f'{asset}_return'
            if return_col in portfolio_df.columns:
                contract = self.selected_contracts[asset]
                plt.plot(portfolio_df['dte'], portfolio_df[return_col], 
                        linewidth=2, label=f'{asset} (${contract["strike"]:.0f})', 
                        alpha=0.8, color=colors[i])
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
        plt.title(f'Individual Asset Returns ({self.target_expiry} Expiry)')
        plt.xlabel('Days to Expiry')
        plt.ylabel('Return (%)')
        plt.xlim(portfolio_df['dte'].max(), portfolio_df['dte'].min())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, 'individual_returns.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved individual returns plot")
    
    def print_final_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("📋 FINAL ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n🎯 STRATEGY PARAMETERS:")
        print(f"   Target expiry: {self.target_expiry} (January 19, 2024)")
        print(f"   Target OPP: {self.target_opp}%")
        print(f"   Investment per asset: $10,000")
        print(f"   Analysis period: Last 365 days to expiry")
        
        print(f"\n📊 SELECTED CONTRACTS:")
        for asset, contract in self.selected_contracts.items():
            eval_data = contract['evaluation']
            print(f"   {asset}: ${contract['strike']:.0f} strike, {self.target_expiry} expiry")
            print(f"        Entry OPP: {eval_data['opp_at_target']:.2f}% at {eval_data['dte_at_target']:.0f} DTE")
        
        if self.portfolio_data:
            portfolio_df = self.portfolio_data['df']
            initial_investment = self.portfolio_data['initial_investment']
            final_value = portfolio_df['total_value'].iloc[-1]
            final_return = portfolio_df['total_return'].iloc[-1]
            max_return = portfolio_df['total_return'].max()
            min_return = portfolio_df['total_return'].min()
            
            print(f"\n💼 PORTFOLIO PERFORMANCE:")
            print(f"   Initial investment: ${initial_investment:,.0f}")
            print(f"   Final value: ${final_value:,.0f}")
            print(f"   Total return: {final_return:.1f}%")
            print(f"   Best return: {max_return:.1f}%")
            print(f"   Worst return: {min_return:.1f}%")
            print(f"   Number of data points: {len(portfolio_df):,}")
            print(f"   Common expiry: {self.target_expiry}")
    
    def run_complete_analysis(self):
        """Run the complete LEAPS analysis for 20240119 expiry"""
        print("🚀 STARTING COMPLETE LEAPS ANALYSIS")
        print(f"Target Expiry: {self.target_expiry} (January 19, 2024)")
        print("="*80)
        
        # Part 1: Contract Selection
        if not self.select_all_contracts():
            return None
        
        # Part 2: Portfolio Analysis
        self.filter_to_last_365_days()
        
        if not self.selected_contracts:
            print("❌ No contracts remain after filtering. Analysis stopped.")
            return None
        
        # Create portfolio
        portfolio_df = self.create_portfolio_timeline()
        if portfolio_df is None:
            return None
        
        # Generate all plots
        self.plot_individual_assets()
        self.plot_portfolio_overlay()
        self.plot_portfolio_returns()
        self.plot_individual_returns()
        
        # Print summary
        self.print_final_summary()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"   All plots saved to 'leaps_analysis_output_{self.target_expiry}/' directory")
        print(f"   Portfolio data available in analyzer.portfolio_data")
        
        return self.portfolio_data


# Usage example
if __name__ == "__main__":
    # Initialize and run complete analysis for 20240119 expiry only
    analyzer = LEAPSAnalyzer("/Users/tim/CODE_PROJECTS/OPP/options_leaps_csv")
    results = analyzer.run_complete_analysis()