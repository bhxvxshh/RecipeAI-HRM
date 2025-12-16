"""
Hugging Face Login Script
"""
from huggingface_hub import login, whoami

print("="*60)
print("Hugging Face Login")
print("="*60)
print("\nGet your token from: https://huggingface.co/settings/tokens")
print("Create a 'Write' token and paste it below\n")

token = input("Enter your HuggingFace token: ").strip()

if not token:
    print("❌ No token provided")
    exit(1)

try:
    login(token=token)
    user_info = whoami()
    print(f"\n✅ Successfully logged in as: {user_info['name']}")
    print(f"✓ Token saved to ~/.cache/huggingface/token")
    print("\nYou can now run: python scripts/upload_to_hf.py")
except Exception as e:
    print(f"\n❌ Login failed: {e}")
    print("\nMake sure:")
    print("  1. Token is correct")
    print("  2. Token has 'Write' access")
    exit(1)
