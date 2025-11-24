import os
import pandas as pd #type:ignore
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

def upload_air_quality_data():
    """
    Upload air quality data from CSV to MongoDB
    
    Reads raw data from data/raw/global_air_quality_data_10000.csv
    and uploads it to MongoDB training_data collection
    """
    
    # Get MongoDB URI from environment
    mongodb_uri = os.getenv("MONGODB_URI")
    
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in .env file")
    
    # Path to CSV file
    csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / "global_air_quality_data_10000.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    print(f"📂 Reading CSV from: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded {len(df)} records from CSV")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Connect to MongoDB
    try:
        client = MongoClient(mongodb_uri)
        db = client.air_quality_db
        collection = db.training_data
        
        print(f"✓ Connected to MongoDB")
        
        # Convert DataFrame to dictionary records
        records = df.to_dict(orient='records')
        
        # Convert datetime objects to be MongoDB-compatible
        for record in records:
            if 'Date' in record:
                # Convert to datetime if it's a string
                if isinstance(record['Date'], str):
                    record['Date'] = pd.to_datetime(record['Date']).to_pydatetime()
        
        # Clear existing data (optional - comment out if you want to keep existing data)
        # collection.delete_many({})
        
        # Insert all records
        result = collection.insert_many(records)
        
        print(f"✅ Successfully uploaded {len(result.inserted_ids)} records to MongoDB")
        print(f"📊 Database: air_quality_db")
        print(f"📋 Collection: training_data")
        
        # Print sample record
        sample = collection.find_one()
        print(f"\n📌 Sample record:")
        for key, value in sample.items():
            if key != '_id':
                print(f"   {key}: {value}")
        
        client.close()
        print("\n✓ Connection closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to MongoDB: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Air Quality Data Upload...\n")
    success = upload_air_quality_data()
    exit(0 if success else 1)
