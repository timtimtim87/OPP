import pandas as pd
import os
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

class DataStructureAnalyzer:
    def __init__(self, data_dir="options_data"):
        """
        Analyze options data structure and create documentation
        
        Args:
            data_dir: Root directory containing options data
        """
        self.data_dir = Path(data_dir)
        self.analysis = {}
        
    def parse_option_ticker(self, ticker):
        """Parse option ticker to extract components"""
        try:
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
            expiry_date = f"{year:04d}-{month:02d}-{day:02d}"
            
            # Convert strike
            strike = float(strike_str) / 1000
            
            return {
                'underlying': symbol,
                'expiry_date': expiry_date,
                'option_type': option_type,
                'strike': strike,
                'expiry_yyyymmdd': f"{year:04d}{month:02d}{day:02d}"
            }
        except:
            return None
    
    def analyze_directory_structure(self):
        """Analyze the overall directory structure"""
        print("📁 Analyzing directory structure...")
        
        structure = {}
        
        # Get all subdirectories and files
        for root, dirs, files in os.walk(self.data_dir):
            rel_path = Path(root).relative_to(self.data_dir)
            
            # Count files by extension
            file_stats = defaultdict(int)
            total_size = 0
            
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                file_stats[ext] += 1
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass
            
            if file_stats:  # Only include directories with files
                structure[str(rel_path)] = {
                    'file_count_by_type': dict(file_stats),
                    'total_files': sum(file_stats.values()),
                    'total_size_mb': round(total_size / (1024 * 1024), 1)
                }
        
        self.analysis['directory_structure'] = structure
        return structure
    
    def analyze_sample_files(self, max_files=10):
        """Analyze sample data files to understand structure"""
        print("📊 Analyzing sample data files...")
        
        # Find CSV files
        csv_files = list(self.data_dir.rglob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV files found")
            return {}
        
        print(f"Found {len(csv_files)} CSV files, analyzing {min(max_files, len(csv_files))}...")
        
        file_analysis = []
        
        # Analyze a sample of files
        sample_files = csv_files[:max_files] if len(csv_files) > max_files else csv_files
        
        for i, file_path in enumerate(sample_files):
            print(f"  [{i+1}/{len(sample_files)}] Analyzing {file_path.name}")
            
            try:
                # Read sample of file
                df = pd.read_csv(file_path, nrows=10000)
                
                file_info = {
                    'filename': file_path.name,
                    'file_size_mb': round(file_path.stat().st_size / (1024 * 1024), 1),
                    'total_rows': len(df),
                    'columns': list(df.columns),
                    'column_types': df.dtypes.astype(str).to_dict(),
                    'sample_data': df.head(3).to_dict('records')
                }
                
                # Analyze tickers if present
                if 'ticker' in df.columns:
                    # Parse tickers
                    parsed_tickers = []
                    for ticker in df['ticker'].dropna().unique()[:100]:  # Sample 100 tickers
                        parsed = self.parse_option_ticker(ticker)
                        if parsed:
                            parsed_tickers.append(parsed)
                    
                    if parsed_tickers:
                        # Get unique underlyings, expiries, types
                        underlyings = list(set(t['underlying'] for t in parsed_tickers))
                        expiries = list(set(t['expiry_date'] for t in parsed_tickers))
                        types = list(set(t['option_type'] for t in parsed_tickers))
                        
                        file_info['ticker_analysis'] = {
                            'total_unique_tickers': df['ticker'].nunique(),
                            'sample_unique_underlyings': sorted(underlyings),
                            'sample_expiry_dates': sorted(expiries),
                            'option_types': sorted(types),
                            'sample_parsed_tickers': parsed_tickers[:5]  # Just first 5
                        }
                
                file_analysis.append(file_info)
                
            except Exception as e:
                print(f"    ❌ Error analyzing {file_path.name}: {e}")
                continue
        
        self.analysis['file_analysis'] = file_analysis
        return file_analysis
    
    def analyze_data_coverage(self):
        """Analyze what data we have - date ranges, underlyings, etc."""
        print("📅 Analyzing data coverage...")
        
        csv_files = list(self.data_dir.rglob("*.csv"))
        
        if not csv_files:
            return {}
        
        # Extract dates from filenames (assuming YYYY-MM-DD.csv format)
        file_dates = []
        for file_path in csv_files:
            # Try to extract date from filename
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            match = re.search(date_pattern, file_path.name)
            if match:
                file_dates.append(match.group(1))
        
        # Overall statistics
        all_underlyings = set()
        all_expiries = set()
        all_option_types = set()
        volume_stats = []
        
        # Analyze a sample of files for content statistics
        sample_files = csv_files[:20] if len(csv_files) > 20 else csv_files
        
        for file_path in sample_files:
            try:
                df = pd.read_csv(file_path, nrows=5000)  # Sample rows
                
                if 'ticker' in df.columns:
                    for ticker in df['ticker'].dropna().sample(min(100, len(df))):
                        parsed = self.parse_option_ticker(ticker)
                        if parsed:
                            all_underlyings.add(parsed['underlying'])
                            all_expiries.add(parsed['expiry_date'])
                            all_option_types.add(parsed['option_type'])
                
                if 'volume' in df.columns:
                    volume_stats.extend(df['volume'].dropna().tolist())
                    
            except Exception as e:
                continue
        
        coverage = {
            'date_range': {
                'files_with_dates': len(file_dates),
                'date_range': f"{min(file_dates)} to {max(file_dates)}" if file_dates else "No dates found",
                'sample_dates': sorted(file_dates)[:10] + (["..."] if len(file_dates) > 10 else [])
            },
            'content_coverage': {
                'unique_underlyings_sample': sorted(list(all_underlyings))[:50],
                'total_underlyings_found': len(all_underlyings),
                'unique_expiries_sample': sorted(list(all_expiries))[:20],
                'total_expiries_found': len(all_expiries),
                'option_types': sorted(list(all_option_types)),
            }
        }
        
        if volume_stats:
            coverage['volume_statistics'] = {
                'total_volume_samples': len(volume_stats),
                'avg_volume': round(np.mean(volume_stats), 2),
                'median_volume': round(np.median(volume_stats), 2),
                'max_volume': max(volume_stats),
                'min_volume': min(volume_stats)
            }
        
        self.analysis['data_coverage'] = coverage
        return coverage
    
    def create_readme(self):
        """Create a README.md file documenting the data structure"""
        readme_content = f"""# Options Data Structure

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📁 Directory Structure

```
{self.data_dir}/
"""
        
        # Add directory tree
        for path, info in self.analysis.get('directory_structure', {}).items():
            if path == '.':
                continue
            indent = "  " * (len(Path(path).parts) - 1)
            readme_content += f"{indent}├── {Path(path).name}/\n"
            readme_content += f"{indent}    📊 {info['total_files']} files ({info['total_size_mb']} MB)\n"
            
            for ext, count in info['file_count_by_type'].items():
                readme_content += f"{indent}    - {count} {ext} files\n"
        
        readme_content += "```\n\n"
        
        # Add data coverage
        if 'data_coverage' in self.analysis:
            coverage = self.analysis['data_coverage']
            readme_content += f"""## 📅 Data Coverage

- **Date Range**: {coverage['date_range']['date_range']}
- **Files with dates**: {coverage['date_range']['files_with_dates']}
- **Unique underlyings found**: {coverage['content_coverage']['total_underlyings_found']}
- **Unique expiry dates found**: {coverage['content_coverage']['total_expiries_found']}
- **Option types**: {', '.join(coverage['content_coverage']['option_types'])}

### Sample Underlyings
{', '.join(coverage['content_coverage']['unique_underlyings_sample'])}

### Sample Expiry Dates  
{', '.join(coverage['content_coverage']['unique_expiries_sample'])}

"""
        
        # Add file format information
        if 'file_analysis' in self.analysis and self.analysis['file_analysis']:
            sample_file = self.analysis['file_analysis'][0]
            
            readme_content += f"""## 📋 File Format

Each daily CSV file contains options data with the following structure:

### Columns
{', '.join(f'`{col}`' for col in sample_file['columns'])}

### Data Types
```
{chr(10).join(f'{col}: {dtype}' for col, dtype in sample_file['column_types'].items())}
```

### Sample Data
```csv
{','.join(sample_file['columns'])}
"""
            
            for row in sample_file['sample_data']:
                readme_content += ','.join(str(row.get(col, '')) for col in sample_file['columns']) + '\n'
            
            readme_content += "```\n\n"
            
            # Add ticker format explanation
            if 'ticker_analysis' in sample_file:
                ta = sample_file['ticker_analysis']
                readme_content += f"""### Option Ticker Format

Tickers follow the format: `O:SYMBOL[YYMMDD][C/P][STRIKE]`

- **O:** Options prefix
- **SYMBOL**: Underlying asset (e.g., AAPL, SPY)  
- **YYMMDD**: Expiry date
- **C/P**: Call or Put
- **STRIKE**: Strike price * 1000 (e.g., 00150000 = $150.00)

#### Examples from data:
"""
                for ticker in ta['sample_parsed_tickers']:
                    readme_content += f"- `{ticker['underlying']}{ticker['expiry_yyyymmdd']}{ticker['option_type']}{int(ticker['strike']*1000):08d}`: {ticker['underlying']} {ticker['option_type']}all expiring {ticker['expiry_date']} at ${ticker['strike']}\n"
        
        # Add usage notes
        readme_content += f"""

## 💡 Usage Notes

### For LEAPS Analysis
This data structure is ideal for:
- Analyzing specific option contracts over time
- Building time series of option prices
- Calculating OPP (Options Price Percentage)
- Backtesting LEAPS strategies

### Data Size Considerations  
- **Total files**: {sum(info['total_files'] for info in self.analysis.get('directory_structure', {}).values())}
- **Total size**: {sum(info['total_size_mb'] for info in self.analysis.get('directory_structure', {}).values())} MB
- **Recommendation**: Consider filtering to specific underlyings or date ranges for analysis

### File Organization
- Raw files: `{self.data_dir}/raw/` (compressed .gz files)
- Processed files: `{self.data_dir}/processed/` (extracted .csv files)
- Daily structure: One file per trading day with all options

## 🔧 Next Steps

1. **Filter data** by specific underlyings (e.g., AAPL, SPY, QQQ)
2. **Reorganize by contract** for time series analysis
3. **Calculate derived metrics** (OPP, Greeks, etc.)
4. **Build LEAPS analysis pipeline**

---
*This documentation was auto-generated. The actual data files are excluded from the repository due to size.*
"""
        
        return readme_content
    
    def save_analysis_json(self):
        """Save detailed analysis as JSON"""
        output_file = self.data_dir / "data_structure_analysis.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)
        
        print(f"✅ Saved detailed analysis to: {output_file}")
        return output_file
    
    def run_complete_analysis(self):
        """Run complete data structure analysis"""
        print("🔍 STARTING DATA STRUCTURE ANALYSIS")
        print("="*50)
        
        start_time = datetime.now()
        
        # Run all analysis steps
        self.analyze_directory_structure()
        self.analyze_sample_files()
        self.analyze_data_coverage()
        
        # Create documentation
        readme_content = self.create_readme()
        readme_file = self.data_dir / "README.md"
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Save JSON analysis
        json_file = self.save_analysis_json()
        
        # Summary
        duration = datetime.now() - start_time
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⏱️  Duration: {duration}")
        print(f"📄 README created: {readme_file}")
        print(f"📋 JSON analysis: {json_file}")
        
        return {
            'readme_file': readme_file,
            'json_file': json_file,
            'analysis': self.analysis
        }


if __name__ == "__main__":
    # Analyze the options data structure
    analyzer = DataStructureAnalyzer("options_data")
    results = analyzer.run_complete_analysis()
    
    print(f"\n📚 Documentation files created:")
    print(f"   README.md - Human-readable documentation")
    print(f"   data_structure_analysis.json - Detailed analysis")
    print(f"\n💡 Add these to your GitHub repo to document your data!")