"""
AAPL-Only Test Version of Contract Processor
Test the date parsing fix on one asset before processing all data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re

class AAPLTestProcessor:
    def __init__(self, daily_data_dir, contracts_output_dir, spot_prices_file):
        self.daily_data_dir = Path(daily_data_dir)
        self.contracts_dir = Path(contracts_output_dir)
        self.spot_prices_file = spot_prices_file
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spot prices data
        self.spot_prices = self.load_spot_prices()
        
        print(f"🍎 AAPL TEST CONTRACT PROCESSOR")
        print(f"Daily data: {daily_data_dir}")
        print(f"Output dir: {contracts_output_dir}")
        print(f"Spot prices: {spot_prices_file}")
        print(f"🎯 PROCESSING AAPL ONLY FOR TESTING")
        print("=" * 60)
    
    def load_spot_prices(self):
        """Load and prepare spot prices data"""
        print(f"\n📈 LOADING SPOT PRICES")
        print("-" * 30)
        
        try:
            spot_df = pd.read_csv(self.spot_prices_file)
            spot_df['Date'] = pd.to_datetime(spot_df['Date'])
            
            print(f"✅ Loaded spot prices: {len(spot_df):,} rows")
            print(f"📅 Date range: {spot_df['Date'].min().date()} to {spot_df['Date'].max().date()}")
            print(f"🏢 Unique tickers: {spot_df['Ticker'].nunique()}")
            
            # Check if AAPL is available
            if 'AAPL' in spot_df['Ticker'].values:
                aapl_count = len(spot_df[spot_df['Ticker'] == 'AAPL'])
                print(f"🍎 AAPL spot prices: {aapl_count} data points")
            else:
                print(f"❌ AAPL not found in spot prices!")
                return {}
            
            # Create lookup dictionary for faster access
            spot_lookup = {}
            for ticker in spot_df['Ticker'].unique():
                ticker_data = spot_df[spot_df['Ticker'] == ticker].set_index('Date')['Close']
                spot_lookup[ticker] = ticker_data
            
            return spot_lookup
            
        except Exception as e:
            print(f"❌ Error loading spot prices: {e}")
            return {}
    
    def parse_option_ticker(self, ticker):
        """Parse Polygon option ticker format - FIXED VERSION"""
        if ticker.startswith('O:'):
            ticker = ticker[2:]
        
        # Pattern: SYMBOL + YYMMDD + C/P + STRIKE (8 digits)
        pattern = r'^([A-Z]+)(\d{6})([CP])(\d{8})$'
        match = re.match(pattern, ticker)
        
        if not match:
            return None
        
        symbol, date_str, option_type, strike_str = match.groups()
        
        # FIXED DATE CONVERSION
        year = int(date_str[:2])
        if year <= 50:     # 00-50 = 2000-2050  
            year = 2000 + year
        else:              # 51-99 = 1951-1999
            year = 1900 + year
        
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiry_date = f"{year:04d}{month:02d}{day:02d}"
        
        # Convert strike
        strike = float(strike_str) / 1000
        
        return {
            'underlying': symbol,
            'expiry': expiry_date,
            'option_type': option_type,
            'strike': strike,
            'contract_id': f"{symbol}_{expiry_date}_{option_type}_{int(strike)}"
        }
    
    def test_date_parsing(self):
        """Test the date parsing with some examples"""
        print(f"\n🧪 TESTING DATE PARSING")
        print("-" * 30)
        
        test_cases = [
            "O:AAPL260116C00250000",  # Should be 2026-01-16
            "O:AAPL250117C00200000",  # Should be 2025-01-17  
            "O:AAPL240119C00180000",  # Should be 2024-01-19
            "O:AAPL460115C00300000",  # Should be 2046-01-15 (far future)
        ]
        
        for test_ticker in test_cases:
            parsed = self.parse_option_ticker(test_ticker)
            if parsed:
                print(f"✅ {test_ticker} → Expiry: {parsed['expiry']} ({parsed['underlying']} {parsed['option_type']} ${parsed['strike']})")
            else:
                print(f"❌ {test_ticker} → Failed to parse")
    
    def calculate_essential_metrics(self, df):
        """Calculate only the essential metrics you specified"""
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        # Get underlying asset symbol
        underlying = df['underlying'].iloc[0]
        
        # Merge with spot prices
        df = self.add_spot_prices(df, underlying)
        
        # Essential attributes
        df['date'] = pd.to_datetime(df['date'])
        df['price'] = df['close']
        df['interpolated_price'] = self.calculate_interpolated_price(df)
        
        # OPP calculation
        df['opp'] = np.where(
            df['underlying_close'] > 0,
            (df['price'] / df['underlying_close']) * 100,
            np.nan
        )
        
        # Moneyness
        df['moneyness'] = df['underlying_close'] / df['strike']
        df['underlying_spot_price'] = df['underlying_close']
        
        # DTE calculation
        expiry_date = pd.to_datetime(df['expiry'].iloc[0], format='%Y%m%d')
        df['dte'] = (expiry_date - df['date']).dt.days
        
        # DTE category
        df['dte_category'] = pd.cut(df['dte'], 
                                   bins=[0, 30, 90, 180, 365, float('inf')], 
                                   labels=['<30d', '30-90d', '90-180d', '180-365d', '>365d'])
        
        # Rolling Range
        df = self.calculate_rolling_range(df, window=90)
        
        return df
    
    def calculate_interpolated_price(self, df):
        """Create interpolated price series to fill gaps"""
        interpolated = df['price'].copy()
        interpolated = interpolated.interpolate(method='linear')
        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
        return interpolated
    
    def calculate_rolling_range(self, df, window=90):
        """Calculate Rolling Range (RR)"""
        df[f'rolling_high_{window}d'] = df['price'].rolling(window=window).max()
        df[f'rolling_low_{window}d'] = df['price'].rolling(window=window).min()
        
        df[f'rolling_range_{window}d'] = np.where(
            df[f'rolling_high_{window}d'] > 0,
            ((df[f'rolling_high_{window}d'] - df[f'rolling_low_{window}d']) / df[f'rolling_high_{window}d']) * 100,
            np.nan
        )
        
        df['rr'] = df[f'rolling_range_{window}d']
        
        df[f'price_percentile_{window}d'] = np.where(
            (df[f'rolling_high_{window}d'] > df[f'rolling_low_{window}d']) & 
            (df[f'rolling_high_{window}d'] > 0),
            ((df['price'] - df[f'rolling_low_{window}d']) / 
             (df[f'rolling_high_{window}d'] - df[f'rolling_low_{window}d'])) * 100,
            np.nan
        )
        
        return df
    
    def add_spot_prices(self, df, underlying):
        """Add underlying spot prices to options data"""
        if underlying not in self.spot_prices:
            print(f"⚠️ Warning: No spot prices found for {underlying}")
            df['underlying_close'] = np.nan
            return df
        
        df['date'] = pd.to_datetime(df['date'])
        spot_data = self.spot_prices[underlying]
        
        merged_data = []
        for _, row in df.iterrows():
            date = row['date']
            
            if date in spot_data.index:
                spot_price = spot_data[date]
            else:
                available_dates = spot_data.index[spot_data.index <= date]
                if len(available_dates) > 0:
                    spot_price = spot_data[available_dates[-1]]
                else:
                    spot_price = np.nan
            
            row_dict = row.to_dict()
            row_dict['underlying_close'] = spot_price
            merged_data.append(row_dict)
        
        return pd.DataFrame(merged_data)
    
    def process_contract(self, contract_data, contract_info):
        """Process a single contract"""
        for key, value in contract_info.items():
            contract_data[key] = value
        
        simple_data = self.calculate_essential_metrics(contract_data)
        
        final_columns = [
            'date', 'price', 'interpolated_price', 'opp', 'moneyness',
            'underlying_spot_price', 'dte', 'option_type', 'strike', 'expiry',
            'dte_category', 'rr', 'rolling_high_90d', 'rolling_low_90d', 'price_percentile_90d'
        ]
        
        available_columns = [col for col in final_columns if col in simple_data.columns]
        final_data = simple_data[available_columns]
        
        contract_dir = self.contracts_dir / contract_info['underlying'] / contract_info['expiry']
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{contract_info['contract_id']}.csv"
        filepath = contract_dir / filename
        
        final_data.to_csv(filepath, index=False)
        return filepath
    
    def process_aapl_only(self, min_dte=300):
        """Process AAPL contracts only"""
        
        print(f"\n🍎 PROCESSING AAPL CONTRACTS ONLY")
        print(f"Minimum DTE: {min_dte}")
        print("-" * 50)
        
        contracts = {}
        daily_files = sorted(self.daily_data_dir.glob("*.csv"))
        
        print(f"📁 Found {len(daily_files)} daily files to process")
        
        for i, daily_file in enumerate(daily_files):
            if i % 50 == 0:
                print(f"   Processing file {i+1}/{len(daily_files)}: {daily_file.name}")
            
            try:
                daily_df = pd.read_csv(daily_file)
                daily_df['date'] = daily_file.stem
                
                # Filter to AAPL options only
                aapl_options = daily_df[daily_df['ticker'].str.contains('AAPL', na=False)]
                
                for _, row in aapl_options.iterrows():
                    ticker = row['ticker']
                    parsed = self.parse_option_ticker(ticker)
                    
                    if not parsed or parsed['underlying'] != 'AAPL':
                        continue
                    
                    contract_id = parsed['contract_id']
                    
                    if contract_id not in contracts:
                        contracts[contract_id] = {
                            'data': [],
                            'info': parsed
                        }
                    
                    contracts[contract_id]['data'].append(row.to_dict())
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {daily_file.name}: {e}")
                continue
        
        print(f"📊 Found {len(contracts)} unique AAPL contracts")
        
        # Show sample of found contracts
        if contracts:
            print(f"\n📋 SAMPLE AAPL CONTRACTS FOUND:")
            sample_contracts = list(contracts.keys())[:5]
            for contract_id in sample_contracts:
                info = contracts[contract_id]['info']
                data_points = len(contracts[contract_id]['data'])
                print(f"   {contract_id}: {data_points} data points, expiry {info['expiry']}")
        
        # Process contracts
        processed_count = 0
        skipped_count = 0
        
        for contract_id, contract_info in contracts.items():
            try:
                contract_df = pd.DataFrame(contract_info['data'])
                contract_df['date'] = pd.to_datetime(contract_df['date'])
                
                # Filter by minimum DTE
                if min_dte:
                    expiry_date = pd.to_datetime(contract_info['info']['expiry'], format='%Y%m%d')
                    max_date = expiry_date - timedelta(days=min_dte)
                    contract_df = contract_df[contract_df['date'] <= max_date]
                
                if len(contract_df) < 90:
                    skipped_count += 1
                    continue
                
                # Show processing info for first few contracts
                if processed_count < 3:
                    print(f"\n🔍 Processing: {contract_id}")
                    print(f"   Expiry: {contract_info['info']['expiry']}")
                    print(f"   Data points: {len(contract_df)}")
                    if len(contract_df) > 0:
                        first_date = contract_df['date'].min().date()
                        last_date = contract_df['date'].max().date()
                        expiry_date = pd.to_datetime(contract_info['info']['expiry'], format='%Y%m%d').date()
                        sample_dte = (expiry_date - first_date).days
                        print(f"   Date range: {first_date} to {last_date}")
                        print(f"   Sample DTE: {sample_dte} days")
                
                self.process_contract(contract_df, contract_info['info'])
                processed_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Error processing {contract_id}: {e}")
                skipped_count += 1
                continue
        
        print(f"\n🎉 AAPL PROCESSING COMPLETE!")
        print(f"   ✅ Successfully processed: {processed_count} contracts")
        print(f"   ⏭️ Skipped (insufficient data): {skipped_count} contracts")
        print(f"   📁 Output directory: {self.contracts_dir}")
        
        return processed_count


# Test the AAPL processor
if __name__ == "__main__":
    # Initialize AAPL test processor
    processor = AAPLTestProcessor(
        daily_data_dir="options_data/processed",
        contracts_output_dir="options_data/contracts_aapl_test",
        spot_prices_file="/Users/tim/CODE_PROJECTS/OPP/spot_prices_clean.csv"
    )
    
    # Test date parsing first
    processor.test_date_parsing()
    
    # Process AAPL contracts
    print(f"\n🚀 STARTING AAPL TEST PROCESSING...")
    processor.process_aapl_only(min_dte=300)