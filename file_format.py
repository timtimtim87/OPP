"""
Enhanced Contract File Structure for LEAPS Analysis
Each contract gets its own time-series file with calculated metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re

class ContractProcessor:
    def __init__(self, daily_data_dir, contracts_output_dir):
        self.daily_data_dir = Path(daily_data_dir)
        self.contracts_dir = Path(contracts_output_dir)
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_option_ticker(self, ticker):
        """Parse Polygon option ticker format"""
        if ticker.startswith('O:'):
            ticker = ticker[2:]
        
        # Pattern: SYMBOL + YYMMDD + C/P + STRIKE (8 digits)
        pattern = r'^([A-Z]+)(\d{6})([CP])(\d{8})$'
        match = re.match(pattern, ticker)
        
        if not match:
            return None
        
        symbol, date_str, option_type, strike_str = match.groups()
        
        # Convert date
        year = int(date_str[:2])
        year = 2000 + year if year > 50 else 2020 + year
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
    
    def calculate_enhanced_metrics(self, df):
        """Calculate enhanced metrics for a contract time series"""
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        # Basic derived metrics
        df['mid_price'] = (df['high'] + df['low']) / 2
        df['daily_range'] = df['high'] - df['low']
        df['daily_return'] = df['close'].pct_change()
        
        # Calculate DTE (Days to Expiry)
        expiry_date = pd.to_datetime(df['expiry'].iloc[0], format='%Y%m%d')
        df['dte'] = (expiry_date - pd.to_datetime(df['date'])).dt.days
        
        # OPP (Options Price Percentage) - assuming we have underlying price
        # For now, we'll calculate relative to first price as placeholder
        df['opp'] = (df['close'] / df['close'].iloc[0]) * 100
        
        # Rolling metrics (various windows)
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'volatility_{window}d'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)
            df[f'volume_avg_{window}d'] = df['volume'].rolling(window=window).mean()
            
        # Drawdown calculations
        df['running_max'] = df['close'].expanding().max()
        df['drawdown'] = (df['close'] - df['running_max']) / df['running_max'] * 100
        df['drawdown_50d'] = df['close'].rolling(50).max()
        df['drawdown_50d'] = (df['close'] - df['drawdown_50d']) / df['drawdown_50d'] * 100
        
        # Price momentum
        df['momentum_5d'] = df['close'].pct_change(5) * 100
        df['momentum_20d'] = df['close'].pct_change(20) * 100
        
        # Volume metrics
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['high_volume_day'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        
        # Support/Resistance levels (simple approach)
        df['support_20d'] = df['low'].rolling(20).min()
        df['resistance_20d'] = df['high'].rolling(20).max()
        
        return df
    
    def process_contract(self, contract_data, contract_info):
        """Process a single contract and create enhanced file"""
        
        # Add contract metadata
        for key, value in contract_info.items():
            contract_data[key] = value
        
        # Calculate enhanced metrics
        enhanced_data = self.calculate_enhanced_metrics(contract_data)
        
        # Create directory structure
        contract_dir = self.contracts_dir / contract_info['underlying'] / contract_info['expiry']
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        # Save enhanced contract file
        filename = f"{contract_info['contract_id']}.csv"
        filepath = contract_dir / filename
        
        enhanced_data.to_csv(filepath, index=False)
        return filepath
    
    def build_contracts_from_daily_files(self, target_underlying=None, min_dte=300):
        """Convert daily files to contract-specific enhanced files"""
        
        print(f"🔄 Processing daily files into contract-specific files...")
        
        # Dictionary to accumulate contract data
        contracts = {}
        
        # Process each daily file
        daily_files = sorted(self.daily_data_dir.glob("*.csv"))
        
        for i, daily_file in enumerate(daily_files):
            if i % 50 == 0:
                print(f"   Processing file {i+1}/{len(daily_files)}: {daily_file.name}")
            
            try:
                daily_df = pd.read_csv(daily_file)
                daily_df['date'] = daily_file.stem  # Use filename as date
                
                for _, row in daily_df.iterrows():
                    ticker = row['ticker']
                    parsed = self.parse_option_ticker(ticker)
                    
                    if not parsed:
                        continue
                    
                    # Filter by underlying if specified
                    if target_underlying and parsed['underlying'] != target_underlying:
                        continue
                    
                    contract_id = parsed['contract_id']
                    
                    # Initialize contract if first time seeing it
                    if contract_id not in contracts:
                        contracts[contract_id] = {
                            'data': [],
                            'info': parsed
                        }
                    
                    # Add this day's data
                    row_dict = row.to_dict()
                    contracts[contract_id]['data'].append(row_dict)
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {daily_file.name}: {e}")
                continue
        
        print(f"📊 Found {len(contracts)} unique contracts")
        
        # Process each contract
        processed_count = 0
        for contract_id, contract_info in contracts.items():
            try:
                # Convert to DataFrame
                contract_df = pd.DataFrame(contract_info['data'])
                contract_df['date'] = pd.to_datetime(contract_df['date'])
                
                # Filter by minimum DTE if specified
                if min_dte:
                    expiry_date = pd.to_datetime(contract_info['info']['expiry'], format='%Y%m%d')
                    max_date = expiry_date - timedelta(days=min_dte)
                    contract_df = contract_df[contract_df['date'] <= max_date]
                
                if len(contract_df) < 10:  # Skip contracts with insufficient data
                    continue
                
                # Process and save
                self.process_contract(contract_df, contract_info['info'])
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"   ✅ Processed {processed_count} contracts...")
                    
            except Exception as e:
                print(f"   ⚠️ Error processing contract {contract_id}: {e}")
                continue
        
        print(f"🎉 Successfully processed {processed_count} contracts")
        return processed_count

# Example usage
if __name__ == "__main__":
    processor = ContractProcessor(
        daily_data_dir="options_data/processed",
        contracts_output_dir="options_data/contracts"
    )
    
    # Process all contracts
    processor.build_contracts_from_daily_files(min_dte=300)
    
    # Or process specific underlying
    # processor.build_contracts_from_daily_files(target_underlying="AAPL", min_dte=300)