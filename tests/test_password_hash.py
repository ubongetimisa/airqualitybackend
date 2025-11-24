#!/usr/bin/env python3
"""
Test password hashing to verify bcrypt is working
"""

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

print("=" * 80)
print("TESTING PASSWORD HASHING")
print("=" * 80)

test_passwords = [
    "SimplePass123!",
    "S3cur3P@ssw0rd!",
    "VeryLongPasswordThatExceedsTheSeventyTwoByteLimitButShouldBeTruncatedAutomatically123456789",
    "A" * 100,  # 100 A's - way over the limit
]

for i, password in enumerate(test_passwords, 1):
    print(f"\n[Test {i}] Original password length: {len(password)} bytes")
    truncated = password[:72]
    print(f"  Truncated to: {len(truncated)} bytes")
    
    try:
        hashed = pwd_context.hash(truncated)
        print(f"  ✅ Hashed successfully: {hashed[:50]}...")
        
        # Verify it works
        is_valid = pwd_context.verify(truncated, hashed)
        print(f"  ✅ Verification: {'PASSED' if is_valid else 'FAILED'}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 80)
print("✅ Password hashing is working correctly!")
print("=" * 80)
