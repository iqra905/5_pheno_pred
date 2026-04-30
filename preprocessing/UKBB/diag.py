#!/usr/bin/env python3
"""
Diagnostic script to check bgen-reader API availability
"""

from bgen_reader import open_bgen, read_bgen
import os

# Your BGEN file path
bgen_path = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra/ukb_imp_chr1_maf0.05.bgen"

print("="*80)
print("BGEN-READER API DIAGNOSTIC")
print("="*80)
print()

# Check if index file exists
index_path = bgen_path + ".bgi"
print(f"BGEN file: {os.path.exists(bgen_path)}")
print(f"Index file (.bgi): {os.path.exists(index_path)}")
print()

# Open the file and check available methods
print("Opening BGEN file...")
try:
    with open_bgen(bgen_path, verbose=False) as bgen:
        print(f"✓ File opened successfully")
        print(f"  Variants: {bgen.nvariants:,}")
        print(f"  Samples: {bgen.nsamples:,}")
        print()
        
        # Check available methods
        print("Available methods and attributes:")
        methods = [m for m in dir(bgen) if not m.startswith('_')]
        for method in methods:
            print(f"  - {method}")
        print()
        
        # Test variants() method
        print("Testing variants() method...")
        try:
            variants_df = bgen.variants()
            print(f"✓ variants() works! Returns DataFrame with {len(variants_df)} rows")
            print("\nFirst 3 rows:")
            print(variants_df.head(3))
        except AttributeError as e:
            print(f"✗ variants() not available: {e}")
            print("\nTrying alternative: iterating with read()")
            variant = bgen.read(0)
            print(f"✓ read(0) works!")
            print(f"  Variant attributes: {[a for a in dir(variant) if not a.startswith('_')]}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        
except Exception as e:
    print(f"✗ Error opening file: {e}")

print()
print("="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

# Try with read_bgen function
print("\nTrying read_bgen() function (alternative API)...")
try:
    from bgen_reader import read_bgen
    bgen_data = read_bgen(bgen_path, verbose=False)
    print(f"✓ read_bgen() works!")
    print(f"  Type: {type(bgen_data)}")
    print(f"  Attributes: {list(bgen_data.keys()) if hasattr(bgen_data, 'keys') else dir(bgen_data)}")
except Exception as e:
    print(f"✗ read_bgen() failed: {e}")