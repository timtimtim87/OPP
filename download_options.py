import boto3
from botocore.config import Config
import pandas as pd
import os
import gzip
import shutil
import json
from datetime import datetime, timedelta, date
from pathlib import Path
import logging

class OptionsDownloader:
    def __init__(self):
        """Simple options data downloader using S3 direct access"""
        # Load config
        self.config = self.load_config()
        
        # Setup directories
        self.base_dir = Path("options_data")
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        for dir_path in [self.base_dir, self.raw_dir, self.processed_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup S3 client
        self.s3_client = self.setup_s3_client()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        print(f"✅ Options downloader initialized")
        print(f"📁 Data will be saved to: {self.base_dir}")
    
    def load_config(self):
        """Load credentials from config.json"""
        config_file = Path("config.json")
        
        if not config_file.exists():
            print("❌ config.json not found!")
            print("📝 Create config.json with your S3 credentials:")
            print(json.dumps({
                "s3_access_key_id": "YOUR_S3_ACCESS_KEY_ID_HERE", 
                "s3_secret_access_key": "YOUR_S3_SECRET_ACCESS_KEY_HERE",
                "s3_endpoint": "https://files.polygon.io",
                "s3_bucket": "flatfiles"
            }, indent=2))
            raise FileNotFoundError("Please create config.json with your credentials")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check for placeholder values
        placeholders = ["YOUR_S3_ACCESS_KEY_ID_HERE", "YOUR_S3_SECRET_ACCESS_KEY_HERE"]
        if any(config.get(key, "") in placeholders for key in config):
            raise ValueError("❌ Please replace placeholder values in config.json with your actual S3 credentials")
        
        required_keys = ['s3_access_key_id', 's3_secret_access_key']
        missing_keys = [key for key in required_keys if not config.get(key)]
        if missing_keys:
            raise ValueError(f"❌ Missing required config keys: {missing_keys}")
        
        return config
    
    def setup_s3_client(self):
        """Setup S3 client for Polygon flat files"""
        try:
            session = boto3.Session(
                aws_access_key_id=self.config['s3_access_key_id'],
                aws_secret_access_key=self.config['s3_secret_access_key']
            )
            
            s3_client = session.client(
                's3',
                endpoint_url=self.config['s3_endpoint'],
                config=Config(signature_version='s3v4')
            )
            
            print("✅ S3 client configured successfully")
            return s3_client
            
        except Exception as e:
            print(f"❌ Failed to setup S3 client: {e}")
            raise
    
    def list_available_options_data(self, data_type='day_aggs_v1', max_files=20):
        """List available options files to see the structure"""
        print(f"🔍 Exploring available options data...")
        
        # US Options OPRA data structure
        prefix = f'us_options_opra/{data_type}/'
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            files_found = []
            for page in paginator.paginate(Bucket=self.config['s3_bucket'], Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files_found.append(obj)
                        if len(files_found) >= max_files:
                            break
                if len(files_found) >= max_files:
                    break
            
            print(f"📊 Found {len(files_found)} files with prefix: {prefix}")
            
            if files_found:
                print(f"📋 Sample files:")
                for i, obj in enumerate(files_found[:10]):
                    size_mb = obj['Size'] / (1024 * 1024)
                    date_str = obj['LastModified'].strftime('%Y-%m-%d')
                    print(f"   {i+1}. {obj['Key']} ({size_mb:.1f} MB, {date_str})")
            
            return [obj['Key'] for obj in files_found]
            
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []
    
    def get_files_for_date_range(self, data_type='day_aggs_v1', days_back=5):
        """Get options files for a specific date range"""
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days_back)
        
        print(f"🔍 Looking for {data_type} files from {start_date} to {end_date}")
        
        # Generate expected file paths
        target_files = []
        current_date = start_date
        
        while current_date <= end_date:
            # Format: us_options_opra/day_aggs_v1/2024/09/2024-09-11.csv.gz
            year = current_date.year
            month = current_date.month
            date_str = current_date.strftime('%Y-%m-%d')
            
            file_key = f'us_options_opra/{data_type}/{year}/{month:02d}/{date_str}.csv.gz'
            target_files.append(file_key)
            
            current_date += timedelta(days=1)
        
        # Check which files actually exist
        existing_files = []
        
        for file_key in target_files:
            try:
                self.s3_client.head_object(Bucket=self.config['s3_bucket'], Key=file_key)
                existing_files.append(file_key)
                print(f"✅ Found: {file_key}")
            except Exception:
                print(f"❌ Missing: {file_key}")
        
        print(f"📊 Found {len(existing_files)}/{len(target_files)} files")
        return existing_files
    
    def download_file(self, file_key):
        """Download a single file from S3"""
        filename = Path(file_key).name
        local_path = self.raw_dir / filename
        
        # Skip if already exists
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"⏭️  Already have: {filename} ({size_mb:.1f} MB)")
            return True
        
        print(f"⬇️  Downloading: {filename}")
        
        try:
            # Get file size first
            response = self.s3_client.head_object(Bucket=self.config['s3_bucket'], Key=file_key)
            file_size = response['ContentLength']
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"   File size: {file_size_mb:.1f} MB")
            
            # Download file
            self.s3_client.download_file(
                Bucket=self.config['s3_bucket'],
                Key=file_key,
                Filename=str(local_path)
            )
            
            # Verify download
            actual_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✅ Downloaded: {filename} ({actual_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Download failed for {filename}: {e}")
            if local_path.exists():
                local_path.unlink()
            return False
    
    def extract_file(self, gz_file):
        """Extract a gzipped file"""
        csv_file = self.processed_dir / gz_file.stem
        
        if csv_file.exists():
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"⏭️  Already extracted: {csv_file.name} ({size_mb:.1f} MB)")
            return True
        
        print(f"📦 Extracting: {gz_file.name}")
        
        try:
            with gzip.open(gz_file, 'rb') as f_in:
                with open(csv_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"✅ Extracted: {csv_file.name} ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Extraction failed for {gz_file.name}: {e}")
            return False
    
    def analyze_sample_file(self):
        """Look at the structure of downloaded data"""
        csv_files = list(self.processed_dir.glob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV files found to analyze")
            return
        
        # Use most recent file
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        print(f"📊 Analyzing: {latest_file.name}")
        
        try:
            # Read just a few rows to see structure
            df = pd.read_csv(latest_file, nrows=1000)
            
            print(f"📋 Columns: {list(df.columns)}")
            print(f"📏 Sample has {len(df)} rows")
            
            if 'T' in df.columns:  # Ticker column
                unique_tickers = df['T'].nunique()
                print(f"🎯 Unique option contracts: {unique_tickers}")
                print(f"📝 Sample tickers: {df['T'].head().tolist()}")
            
            print(f"💾 File size: {latest_file.stat().st_size / (1024*1024):.1f} MB")
            
            # Show sample data
            print(f"\n📋 Sample data:")
            print(df.head(3).to_string())
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    def download_options_data(self, days_back=5, data_type='day_aggs_v1', extract=True):
        """Main function to download options data"""
        print(f"🚀 Starting options data download")
        print(f"📅 Getting last {days_back} days of {data_type} data")
        
        start_time = datetime.now()
        
        # Step 1: Get available files
        files = self.get_files_for_date_range(data_type, days_back)
        if not files:
            print("❌ No files available for the specified date range")
            print("\n💡 Let's explore what's available...")
            self.list_available_options_data(data_type)
            return
        
        # Step 2: Download files
        print(f"\n📥 DOWNLOADING {len(files)} FILES")
        print("="*50)
        
        success_count = 0
        for i, file_key in enumerate(files, 1):
            print(f"[{i}/{len(files)}] ", end="")
            if self.download_file(file_key):
                success_count += 1
        
        print(f"\n📊 Download Summary: {success_count}/{len(files)} successful")
        
        # Step 3: Extract files
        if extract:
            print(f"\n📦 EXTRACTING FILES")
            print("="*30)
            
            gz_files = list(self.raw_dir.glob("*.gz"))
            extract_count = 0
            
            for i, gz_file in enumerate(gz_files, 1):
                print(f"[{i}/{len(gz_files)}] ", end="")
                if self.extract_file(gz_file):
                    extract_count += 1
            
            print(f"\n📊 Extraction Summary: {extract_count}/{len(gz_files)} successful")
        
        # Step 4: Quick analysis
        if extract:
            print(f"\n🔍 ANALYZING DATA")
            print("="*20)
            self.analyze_sample_file()
        
        # Final summary
        duration = datetime.now() - start_time
        print(f"\n🎉 COMPLETE!")
        print(f"⏱️  Total time: {duration}")
        print(f"📁 Raw files: {self.raw_dir}")
        if extract:
            print(f"📁 CSV files: {self.processed_dir}")


if __name__ == "__main__":
    # Download 2 years of options data
    downloader = OptionsDownloader()
    
    print("🚀 DOWNLOADING 2 YEARS OF OPTIONS DATA")
    print("="*50)
    
    # Download 2 years of day aggregates
    downloader.download_options_data(
        days_back=730,                  # 2 years of data
        data_type='day_aggs_v1',        # Day aggregates 
        extract=True                    # Extract to CSV
    )