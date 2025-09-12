# Options Data Structure

*Generated on: 2025-09-12 10:30:50*

## 📁 Directory Structure

```
options_data/
├── processed/
    📊 501 files (7356.5 MB)
    - 501 .csv files
├── raw/
    📊 501 files (1288.4 MB)
    - 501 .gz files
```

## 📅 Data Coverage

- **Date Range**: 2023-09-11 to 2025-09-09
- **Files with dates**: 501
- **Unique underlyings found**: 176
- **Unique expiry dates found**: 93
- **Option types**: C, P

### Sample Underlyings
A, AA, AAAU, AADI, AAL, AAN, AAOI, AAON, AAP, AAPB, AAPD, AAPL, AAPU, AAPX, AB, ABAT, ABBV, ABCL, ABEO, ABEV, ABG, ABM, ABNB, ABOS, ABR, ABSI, ABT, ABUS, ACA, ACAD, ACB, ACCD, ACCO, ACDC, ACGL, ACHC, ACHR, ACI, ACIC, ACIU, ACIW, ACLS, ACLX, ACM, ACMR, ACN, ACNB, ACRS, ACVA, ACWI

### Sample Expiry Dates  
2043-09-15, 2043-09-22, 2043-09-29, 2043-10-06, 2043-10-13, 2043-10-20, 2043-10-27, 2043-11-03, 2043-11-10, 2043-11-17, 2043-11-24, 2043-12-01, 2043-12-08, 2043-12-15, 2043-12-22, 2043-12-29, 2044-01-05, 2044-01-12, 2044-01-19, 2044-01-26

## 📋 File Format

Each daily CSV file contains options data with the following structure:

### Columns
`ticker`, `volume`, `open`, `close`, `high`, `low`, `window_start`, `transactions`

### Data Types
```
ticker: object
volume: int64
open: float64
close: float64
high: float64
low: float64
window_start: int64
transactions: int64
```

### Sample Data
```csv
ticker,volume,open,close,high,low,window_start,transactions
O:A250718C00110000,3,13.0,13.3,13.3,13.0,1752206400000000000,3
O:A250718C00120000,12,3.75,3.96,4.28,3.75,1752206400000000000,9
O:A250718C00125000,5,2.1,1.24,2.1,0.9,1752206400000000000,3
```

### Option Ticker Format

Tickers follow the format: `O:SYMBOL[YYMMDD][C/P][STRIKE]`

- **O:** Options prefix
- **SYMBOL**: Underlying asset (e.g., AAPL, SPY)  
- **YYMMDD**: Expiry date
- **C/P**: Call or Put
- **STRIKE**: Strike price * 1000 (e.g., 00150000 = $150.00)

#### Examples from data:
- `A20450718C00110000`: A Call expiring 2045-07-18 at $110.0
- `A20450718C00120000`: A Call expiring 2045-07-18 at $120.0
- `A20450718C00125000`: A Call expiring 2045-07-18 at $125.0
- `A20450718C00130000`: A Call expiring 2045-07-18 at $130.0
- `A20450718P00100000`: A Pall expiring 2045-07-18 at $100.0


## 💡 Usage Notes

### For LEAPS Analysis
This data structure is ideal for:
- Analyzing specific option contracts over time
- Building time series of option prices
- Calculating OPP (Options Price Percentage)
- Backtesting LEAPS strategies

### Data Size Considerations  
- **Total files**: 1003
- **Total size**: 8644.9 MB
- **Recommendation**: Consider filtering to specific underlyings or date ranges for analysis

### File Organization
- Raw files: `options_data/raw/` (compressed .gz files)
- Processed files: `options_data/processed/` (extracted .csv files)
- Daily structure: One file per trading day with all options

## 🔧 Next Steps

1. **Filter data** by specific underlyings (e.g., AAPL, SPY, QQQ)
2. **Reorganize by contract** for time series analysis
3. **Calculate derived metrics** (OPP, Greeks, etc.)
4. **Build LEAPS analysis pipeline**

---
*This documentation was auto-generated. The actual data files are excluded from the repository due to size.*
