import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import glob
from pathlib import Path

class LEAPSAnalyzerIWLS2026:
    def __init__(self, csv_data_path, iwls_csv_path):
        """
        Initialize LEAPS analyzer with IWLS-based asset selection (20260116 expiry only)
        Entry date: May 9, 2024 (365 days before dataset cutoff)
        """
        self.csv_data_path = csv_data_path
        self.iwls_csv_path = iwls_csv_path
        self.target_opp = 10.0  # Target 10% OPP
        self.target_expiry = '20260116'  # Fixed expiry date - January 16, 2026
        self.entry_date = '2024-05-09'  # Entry date (365 days before dataset cutoff)
        self.target_dte_range = (620, 630)  # Look for contracts around 625 DTE (from entry date to expiry)
        
        # Asset selection parameters
        self.top_n_assets = 10  # Still get assets from range
        self.range_start = 25   # Start from 5th ranked asset
        self.range_end = 35     # End at 25th ranked asset
        self.rolling_range_column = 'rolling_range_pct_6_month'
        
        # Data storage
        self.iwls_data = None
        self.candidate_assets = []
        self.available_assets = []
        self.selected_contracts = {}
        self.portfolio_data = None
        
        print(f"LEAPS Analyzer with IWLS Asset Selection for 2026 Chain initialized")
        print(f"Options data path: {self.csv_data_path}")
        print(f"IWLS data path: {self.iwls_csv_path}")
        print(f"Target expiry: {self.target_expiry} (January 16, 2026)")
        print(f"Entry date: {self.entry_date}")
        print(f"Target OPP: {self.target_opp}%")
        print(f"Target DTE range at entry: {self.target_dte_range}")
        print(f"Asset selection: Ranked {self.range_start}-{self.range_end} by {self.rolling_range_column}")
    
    def load_iwls_data(self):
        """Load and validate IWLS data"""
        print(f"\n🔍 LOADING IWLS DATA")
        print("="*50)
        
        if not os.path.exists(self.iwls_csv_path):
            print(f"❌ Error: IWLS file does not exist: {self.iwls_csv_path}")
            return False
        
        try:
            self.iwls_data = pd.read_csv(self.iwls_csv_path)
            print(f"✅ Loaded IWLS data: {len(self.iwls_data):,} rows")
            
            # Convert date column
            self.iwls_data['date'] = pd.to_datetime(self.iwls_data['date'])
            
            # Check required columns
            required_cols = ['date', 'asset', self.rolling_range_column]
            missing_cols = [col for col in required_cols if col not in self.iwls_data.columns]
            if missing_cols:
                print(f"❌ Error: Missing required columns: {missing_cols}")
                return False
            
            # Calculate DTE from entry date to expiry for asset selection
            entry_datetime = pd.to_datetime(self.entry_date)
            expiry_datetime = pd.to_datetime(self.target_expiry, format='%Y%m%d')
            self.iwls_data['dte_from_entry'] = (expiry_datetime - self.iwls_data['date']).dt.days
            
            # Also calculate DTE from actual expiry date for later filtering
            self.iwls_data['dte'] = (expiry_datetime - self.iwls_data['date']).dt.days
            
            print(f"📅 Date range: {self.iwls_data['date'].min().date()} to {self.iwls_data['date'].max().date()}")
            print(f"📊 Unique assets: {self.iwls_data['asset'].nunique()}")
            print(f"🎯 Entry date: {entry_datetime.date()}")
            print(f"🎯 Expiry date: {expiry_datetime.date()}")
            print(f"📏 DTE at entry: {(expiry_datetime - entry_datetime).days} days")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading IWLS data: {e}")
            return False
    
    def select_candidate_assets(self):
        """Select assets ranked 5-25 by rolling range at entry date"""
        print(f"\n🎯 SELECTING CANDIDATE ASSETS (Ranked {self.range_start}-{self.range_end})")
        print("="*60)
        
        # Filter to data around entry date
        entry_datetime = pd.to_datetime(self.entry_date)
        
        # Look for data within 5 days of entry date
        entry_window = 5
        entry_data = self.iwls_data[
            (self.iwls_data['date'] >= entry_datetime - timedelta(days=entry_window)) &
            (self.iwls_data['date'] <= entry_datetime + timedelta(days=entry_window))
        ].copy()
        
        if len(entry_data) == 0:
            print(f"❌ No IWLS data found around entry date {self.entry_date}")
            return False
        
        print(f"📊 Found {len(entry_data)} data points around entry date")
        
        # For each asset, get the rolling range value closest to entry date
        asset_rolling_ranges = []
        
        for asset in entry_data['asset'].unique():
            asset_data = entry_data[entry_data['asset'] == asset]
            
            # Find point closest to entry date
            closest_idx = (asset_data['date'] - entry_datetime).abs().idxmin()
            closest_point = asset_data.loc[closest_idx]
            
            rolling_range = closest_point[self.rolling_range_column]
            
            # Skip if rolling range is NaN or invalid
            if pd.isna(rolling_range) or rolling_range <= 0:
                continue
            
            asset_rolling_ranges.append({
                'asset': asset,
                'rolling_range': rolling_range,
                'date_at_measurement': closest_point['date'],
                'dte_at_measurement': closest_point['dte_from_entry']
            })
        
        # Sort by rolling range (descending - higher volatility preferred)
        asset_rolling_ranges.sort(key=lambda x: x['rolling_range'], reverse=True)
        
        # Take assets ranked 5-25 instead of top assets
        selected_assets = asset_rolling_ranges[self.range_start-1:self.range_end]
        self.candidate_assets = [asset['asset'] for asset in selected_assets]
        
        print(f"🏆 ASSETS RANKED {self.range_start}-{self.range_end} BY {self.rolling_range_column}:")
        for i, asset_info in enumerate(selected_assets, self.range_start):
            print(f"   {i:>2}. {asset_info['asset']:>6}: {asset_info['rolling_range']:>6.2f}% "
                  f"(on {asset_info['date_at_measurement'].strftime('%Y-%m-%d')})")
        
        return True
    
    def check_options_data_availability(self):
        """Check which candidate assets have options data available"""
        print(f"\n🔍 CHECKING OPTIONS DATA AVAILABILITY")
        print("="*50)
        
        if not os.path.exists(self.csv_data_path):
            print(f"❌ Error: Options data path does not exist: {self.csv_data_path}")
            return False
        
        available_assets = []
        
        for asset in self.candidate_assets:
            asset_path = os.path.join(self.csv_data_path, asset)
            expiry_path = os.path.join(asset_path, self.target_expiry)
            
            if os.path.exists(expiry_path):
                # Check if there are any CSV files
                csv_files = glob.glob(os.path.join(expiry_path, f"{asset}_{self.target_expiry}_C_*.csv"))
                if csv_files:
                    available_assets.append(asset)
                    print(f"   ✅ {asset}: {len(csv_files)} option contracts found")
                else:
                    print(f"   ❌ {asset}: Folder exists but no CSV files found")
            else:
                print(f"   ❌ {asset}: No {self.target_expiry} expiry folder found")
        
        self.available_assets = available_assets
        
        print(f"\n📊 AVAILABILITY SUMMARY:")
        print(f"   Candidate assets: {len(self.candidate_assets)}")
        print(f"   Available for analysis: {len(self.available_assets)}")
        if len(self.candidate_assets) > 0:
            print(f"   Success rate: {len(self.available_assets)/len(self.candidate_assets)*100:.1f}%")
        
        if self.available_assets:
            print(f"\n✅ Assets with options data: {', '.join(self.available_assets)}")
            return True
        else:
            print("❌ No assets have options data available")
            return False
    
    def find_available_strikes_for_expiry(self, asset):
        """Find all available strikes for an asset with the target expiry"""
        expiry_path = os.path.join(self.csv_data_path, asset, self.target_expiry)
        if not os.path.exists(expiry_path):
            return []
        
        strikes = []
        csv_files = glob.glob(os.path.join(expiry_path, f"{asset}_{self.target_expiry}_C_*.csv"))
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            try:
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
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate DTE if not present
            expiry_datetime = pd.to_datetime(self.target_expiry, format='%Y%m%d')
            df['dte'] = (expiry_datetime - df['date']).dt.days
            
            # Clean up OPP column
            if 'OPP' in df.columns:
                df['opp'] = df['OPP']
            elif 'opp' not in df.columns:
                print(f"⚠️ Warning: No OPP column found in {filename}")
                return None
            
            # Remove invalid data
            df = df.dropna(subset=['opp', 'dte'])
            df = df[df['opp'] > 0]
            df = df[df['dte'] >= 0]
            
            # Add metadata
            df['asset'] = asset
            df['expiry'] = self.target_expiry
            df['strike'] = strike
            
            return df.sort_values('date').reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def evaluate_contract_at_entry_date(self, contract_df):
        """Evaluate a contract's OPP at the entry date"""
        if contract_df is None or len(contract_df) == 0:
            return None
        
        entry_datetime = pd.to_datetime(self.entry_date)
        
        # Look for data within 5 days of entry date
        entry_window = 5
        target_data = contract_df[
            (contract_df['date'] >= entry_datetime - timedelta(days=entry_window)) &
            (contract_df['date'] <= entry_datetime + timedelta(days=entry_window))
        ]
        
        if len(target_data) == 0:
            return None
        
        # Find point closest to entry date
        closest_idx = (target_data['date'] - entry_datetime).abs().idxmin()
        closest_point = target_data.loc[closest_idx]
        
        return {
            'opp_at_target': closest_point['opp'],
            'dte_at_target': closest_point['dte'],
            'date_at_target': closest_point['date'],
            'diff_from_target_opp': abs(closest_point['opp'] - self.target_opp),
            'data_points': len(contract_df),
            'entry_data_points': len(target_data)
        }
    
    def find_best_contract_for_asset(self, asset):
        """Find the best contract for an asset based on target OPP criteria"""
        print(f"\n🔍 Analyzing {asset} ({self.target_expiry} expiry)...")
        
        available_strikes = self.find_available_strikes_for_expiry(asset)
        if not available_strikes:
            print(f"  ❌ No contracts found for {asset}")
            return None
        
        print(f"  📊 Found {len(available_strikes)} strikes")
        
        best_contract = None
        best_score = float('inf')
        
        for strike in available_strikes:
            contract_df = self.load_contract_data(asset, strike)
            if contract_df is None:
                continue
            
            evaluation = self.evaluate_contract_at_entry_date(contract_df)
            if evaluation is None:
                continue
            
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
            
            print(f"    💰 Strike ${strike:>6.0f}: OPP={evaluation['opp_at_target']:>5.2f}% "
                  f"at {evaluation['dte_at_target']:>3.0f} DTE, "
                  f"diff={evaluation['diff_from_target_opp']:>5.2f}%")
        
        if best_contract:
            eval_data = best_contract['evaluation']
            print(f"  ✅ SELECTED: ${best_contract['strike']:.0f} strike")
            print(f"     📈 OPP: {eval_data['opp_at_target']:.2f}% at {eval_data['dte_at_target']:.0f} DTE")
            print(f"     📅 Entry date: {eval_data['date_at_target'].strftime('%Y-%m-%d')}")
            print(f"     🎯 Diff from target: {eval_data['diff_from_target_opp']:.2f}%")
        else:
            print(f"  ❌ No suitable contract found")
        
        return best_contract
    
    def select_all_contracts(self):
        """Select the best contract for each available asset"""
        print("\n" + "="*80)
        print("🔍 CONTRACT SELECTION PROCESS")
        print(f"Target Expiry: {self.target_expiry} (January 16, 2026)")
        print(f"Entry Date: {self.entry_date}")
        print("="*80)
        
        for asset in self.available_assets:
            contract = self.find_best_contract_for_asset(asset)
            if contract:
                self.selected_contracts[asset] = contract
        
        print(f"\n📋 SELECTION SUMMARY:")
        print(f"   Available assets: {len(self.available_assets)}")
        print(f"   Contracts selected: {len(self.selected_contracts)}")
        if len(self.available_assets) > 0:
            print(f"   Success rate: {len(self.selected_contracts)/len(self.available_assets)*100:.1f}%")
        
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

    def filter_from_entry_to_cutoff(self):
        """Filter each selected contract from entry date to dataset cutoff"""
        print("\n" + "="*80)
        print("📅 FILTERING FROM ENTRY DATE TO DATASET CUTOFF")
        print("="*80)
        
        entry_datetime = pd.to_datetime(self.entry_date)
        cutoff_datetime = pd.to_datetime('2025-05-09')  # Dataset cutoff
        
        filtered_contracts = {}
        
        for asset, contract in self.selected_contracts.items():
            full_data = contract['data']
            
            # Filter from entry date to cutoff date
            filtered_data = full_data[
                (full_data['date'] >= entry_datetime) &
                (full_data['date'] <= cutoff_datetime)
            ].copy()
            filtered_data = filtered_data.sort_values('date').reset_index(drop=True)
            
            if len(filtered_data) > 0:
                contract['filtered_data'] = filtered_data
                filtered_contracts[asset] = contract
                
                print(f"✅ {asset}: {len(filtered_data):>3} data points "
                      f"({filtered_data['dte'].max():>3.0f} to {filtered_data['dte'].min():>3.0f} DTE)")
                print(f"   📅 From: {filtered_data['date'].min().strftime('%Y-%m-%d')} "
                      f"to: {filtered_data['date'].max().strftime('%Y-%m-%d')}")
            else:
                print(f"❌ {asset}: No data in specified date range")
        
        self.selected_contracts = filtered_contracts
        print(f"\n📊 Kept {len(self.selected_contracts)} contracts with valid data in range")
    
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
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame({'date': all_dates})
        initial_investment_per_asset = 10000
        positions = {}
        total_initial_investment = 0
        
        print(f"\n💰 POSITION SIZING (${initial_investment_per_asset:,} per asset):")
        
        for asset, contract in self.selected_contracts.items():
            if 'filtered_data' not in contract:
                continue
            
            asset_data = contract['filtered_data'].set_index('date')
            asset_timeline = pd.DataFrame(index=pd.to_datetime(all_dates))
            
            # Find price column
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
            
            # Calculate position sizing based on first price
            first_price = asset_timeline['price'].dropna().iloc[0]
            if pd.isna(first_price) or first_price <= 0:
                print(f"❌ {asset}: Invalid first price")
                continue
            
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
            
            # Calculate returns
            position_values = asset_timeline['price'] * num_contracts * 100
            returns = ((asset_timeline['price'] / first_price) - 1) * 100
            
            portfolio_df[f'{asset}_value'] = position_values.values
            portfolio_df[f'{asset}_return'] = returns.values
            
            print(f"   {asset}: {num_contracts:.3f} contracts @ ${first_price:.2f} = ${position_value:>8,.0f}")
        
        # Calculate portfolio totals
        value_columns = [col for col in portfolio_df.columns if col.endswith('_value')]
        portfolio_df['total_value'] = portfolio_df[value_columns].sum(axis=1)
        portfolio_df['total_return'] = ((portfolio_df['total_value'] / total_initial_investment) - 1) * 100
        
        # Add DTE reference
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
        
        return portfolio_df
    
    def create_output_directory(self):
        """Create output directory for plots"""
        output_dir = f'leaps_analysis_output_iwls_{self.target_expiry}'
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
                'Entry Date': eval_data['date_at_target'].strftime('%Y-%m-%d'),
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
            plt.xlim(data['dte'].max(), data['dte'].min())
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            filename = os.path.join(output_dir, f'{asset}_opp_analysis.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
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
        plt.xlim(data['dte'].max() if 'data' in locals() else 650, 
                data['dte'].min() if 'data' in locals() else 250)
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
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Portfolio value
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
    
    def print_asset_selection_summary(self):
        """Print summary of asset selection process"""
        print("\n" + "="*80)
        print("📊 ASSET SELECTION SUMMARY")
        print("="*80)
        
        print(f"\n🎯 SELECTION CRITERIA:")
        print(f"   Rolling range metric: {self.rolling_range_column}")
        print(f"   Evaluation at entry date: {self.entry_date}")
        print(f"   Candidates ranked: {self.range_start}-{self.range_end}")
        
        if hasattr(self, 'candidate_assets') and self.candidate_assets:
            print(f"\n🏆 ASSETS RANKED {self.range_start}-{self.range_end} BY ROLLING RANGE:")
            
            # Get rolling range values for display
            entry_datetime = pd.to_datetime(self.entry_date)
            entry_window = 5
            iwls_entry_data = self.iwls_data[
                (self.iwls_data['date'] >= entry_datetime - timedelta(days=entry_window)) &
                (self.iwls_data['date'] <= entry_datetime + timedelta(days=entry_window))
            ]
            
            for i, asset in enumerate(self.candidate_assets, self.range_start):
                asset_data = iwls_entry_data[iwls_entry_data['asset'] == asset]
                if len(asset_data) > 0:
                    # Get closest to entry date
                    closest_idx = (asset_data['date'] - entry_datetime).abs().idxmin()
                    rolling_range = asset_data.loc[closest_idx, self.rolling_range_column]
                    
                    status = "✅ SELECTED" if asset in self.selected_contracts else "❌ NO OPTIONS DATA"
                    print(f"   {i:>2}. {asset:>6}: {rolling_range:>6.2f}% - {status}")
        
        if hasattr(self, 'available_assets') and self.available_assets:
            print(f"\n📈 FINAL PORTFOLIO ASSETS:")
            for asset in self.available_assets:
                if asset in self.selected_contracts:
                    contract = self.selected_contracts[asset]
                    eval_data = contract['evaluation']
                    print(f"   {asset}: ${contract['strike']:.0f} strike, "
                          f"OPP={eval_data['opp_at_target']:.2f}%")
    
    def print_final_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("📋 FINAL ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n🎯 STRATEGY PARAMETERS:")
        print(f"   Target expiry: {self.target_expiry} (January 16, 2026)")
        print(f"   Entry date: {self.entry_date}")
        print(f"   Dataset cutoff: 2025-05-09")
        print(f"   Target OPP: {self.target_opp}%")
        print(f"   Investment per asset: $10,000")
        print(f"   Asset selection: Ranked {self.range_start}-{self.range_end} by {self.rolling_range_column}")
        
        self.print_asset_selection_summary()
        
        if self.portfolio_data:
            portfolio_df = self.portfolio_data['df']
            initial_investment = self.portfolio_data['initial_investment']
            final_value = portfolio_df['total_value'].iloc[-1]
            final_return = portfolio_df['total_return'].iloc[-1]
            max_return = portfolio_df['total_return'].max()
            min_return = portfolio_df['total_return'].min()
            
            # Calculate days held
            days_held = len(portfolio_df)
            entry_dte = portfolio_df['dte'].iloc[0] if len(portfolio_df) > 0 else 0
            final_dte = portfolio_df['dte'].iloc[-1] if len(portfolio_df) > 0 else 0
            
            print(f"\n💼 PORTFOLIO PERFORMANCE:")
            print(f"   Initial investment: ${initial_investment:,.0f}")
            print(f"   Final value: ${final_value:,.0f}")
            print(f"   Total return: {final_return:.1f}%")
            print(f"   Best return: {max_return:.1f}%")
            print(f"   Worst return: {min_return:.1f}%")
            print(f"   Days held: {days_held}")
            print(f"   DTE range: {entry_dte:.0f} to {final_dte:.0f}")
            print(f"   Common expiry: {self.target_expiry}")
    
    def run_complete_analysis(self):
        """Run the complete LEAPS analysis with IWLS asset selection for 2026 chain"""
        print("🚀 STARTING COMPLETE LEAPS ANALYSIS WITH IWLS ASSET SELECTION")
        print(f"Target Expiry: {self.target_expiry} (January 16, 2026)")
        print(f"Entry Date: {self.entry_date}")
        print("="*80)
        
        # Step 1: Load IWLS data
        if not self.load_iwls_data():
            return None
        
        # Step 2: Select candidate assets by rolling range at entry date
        if not self.select_candidate_assets():
            return None
        
        # Step 3: Check options data availability
        if not self.check_options_data_availability():
            return None
        
        # Step 4: Contract selection
        if not self.select_all_contracts():
            return None
        
        # Step 5: Portfolio analysis - filter from entry to cutoff
        self.filter_from_entry_to_cutoff()
        
        if not self.selected_contracts:
            print("❌ No contracts remain after filtering. Analysis stopped.")
            return None
        
        # Step 6: Create portfolio
        portfolio_df = self.create_portfolio_timeline()
        if portfolio_df is None:
            return None
        
        # Step 7: Generate plots
        self.plot_individual_assets()
        self.plot_portfolio_overlay()
        self.plot_portfolio_returns()
        self.plot_individual_returns()
        
        # Step 8: Print summary
        self.print_final_summary()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"   All plots saved to 'leaps_analysis_output_iwls_{self.target_expiry}/' directory")
        print(f"   Portfolio data available in analyzer.portfolio_data")
        
        return self.portfolio_data


# Usage example
if __name__ == "__main__":
    # Initialize and run complete analysis with IWLS asset selection for 2026 chain
    analyzer = LEAPSAnalyzerIWLS2026(
        csv_data_path="/Users/tim/CODE_PROJECTS/OPP/options_leaps_csv",
        iwls_csv_path="/Users/tim/CODE_PROJECTS/OPP/IWLS_WITH_WAVE_ANALYSIS.csv"
    )
    results = analyzer.run_complete_analysis()