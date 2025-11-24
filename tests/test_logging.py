#!/usr/bin/env python3
"""
Quick test script to verify logging is working properly.
"""

import sys
import logging
from utils.logger_config import get_logger, app_logger, error_logger, debug_logger, api_logger
from pathlib import Path

print("=" * 80)
print("LOGGING TEST")
print("=" * 80)

# Test the main logger
logger = get_logger("air_quality_api")

print("\n✓ Testing logger output to console and files...")
logger.info("Test message from main logger")
logger.warning("Test warning message")
logger.error("Test error message")
logger.debug("Test debug message")

print("\n✓ Testing app_logger...")
app_logger.info("Test from app_logger")

print("\n✓ Testing error_logger...")
error_logger.error("Test from error_logger")

print("\n✓ Testing debug_logger...")
debug_logger.debug("Test from debug_logger")

print("\n✓ Testing api_logger...")
api_logger.info("Test from api_logger")

# Check if files exist and have content
print("\n" + "=" * 80)
print("CHECKING LOG FILES")
print("=" * 80)

log_files = [
    Path("logs/app.log"),
    Path("logs/error.log"),
    Path("logs/debug.log"),
    Path("logs/api.log"),
    Path("../logs/backend/app.log"),
    Path("../logs/backend/error.log"),
    Path("../logs/backend/debug.log"),
    Path("../logs/backend/api.log"),
]

for log_file in log_files:
    abs_path = log_file.resolve()
    if abs_path.exists():
        size = abs_path.stat().st_size
        print(f"✅ {log_file} - {size} bytes")
    else:
        print(f"❌ {log_file} - NOT FOUND")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
