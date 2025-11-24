#!/usr/bin/env python3
"""
Check what's actually in the database
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("❌ MONGODB_URI not found in .env file")
    exit(1)

try:
    print("Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    db = client.air_quality_db
    
    # Check predictions
    count = db.predictions.count_documents({})
    print(f"\n📊 Total predictions in database: {count}")
    
    if count > 0:
        print("\n📋 First prediction record:")
        pred = db.predictions.find_one()
        
        # Print all fields
        for key, value in pred.items():
            if key == "_id":
                print(f"  {key}: {value}")
            elif isinstance(value, datetime):
                print(f"  {key}: {value.isoformat()}")
            else:
                print(f"  {key}: {value}")
        
        print("\n🔍 Full JSON:")
        print(json.dumps({
            **{k: (v.isoformat() if isinstance(v, datetime) else str(v)) for k, v in pred.items()}
        }, indent=2))
    
    client.close()
    print("\n✓ Done!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
