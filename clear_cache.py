"""
Clear Python cache and verify IC generation
"""
import sys
import os
import shutil

# Clear all __pycache__ directories
print("Clearing Python cache...")
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        cache_dir = os.path.join(root, '__pycache__')
        print(f"  Removing: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

print("\nCache cleared. Please restart Python and run main.py again.")
print("\nTo verify IC is correct, check that u0 contains positive AND negative values")
print("(multi-mode superposition), not all-positive values (Gaussian).")
