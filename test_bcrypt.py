"""
bcrypt 테스트
"""

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

passwords = ["test123", "alice123", "bob123"]

for pwd in passwords:
    print(f"Testing password: {pwd} (length: {len(pwd)}, bytes: {len(pwd.encode('utf-8'))})")
    try:
        hashed = pwd_context.hash(pwd)
        print(f"  Success! Hash: {hashed[:50]}...")
    except Exception as e:
        print(f"  Error: {e}")

