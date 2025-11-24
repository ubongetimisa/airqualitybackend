#!/usr/bin/env python3
"""
Clear old predictions from database to test with fresh data
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("❌ MONGODB_URI not found in .env file")
    exit(1)

try:
    print("Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client.air_quality_db
    
    # Show current count
    count_before = db.predictions.count_documents({})
    print(f"📊 Current predictions in database: {count_before}")
    
    if count_before > 0:
        # Delete all predictions
        result = db.predictions.delete_many({})
        print(f"✓ Deleted {result.deleted_count} predictions")
        
        # Verify deletion
        count_after = db.predictions.count_documents({})
        print(f"✓ Predictions remaining: {count_after}")
    else:
        print("No predictions to delete")
    
    client.close()
    print("✓ Done!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
