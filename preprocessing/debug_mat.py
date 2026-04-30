import os
import struct
import numpy as np
from scipy import io
import h5py
import argparse

def check_file_signature(file_path):
    """Check the file signature to determine the format"""
    print(f"\n=== File Signature Analysis for {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return None
    
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    with open(file_path, 'rb') as f:
        # Read first 16 bytes to check signature
        header = f.read(16)
        print(f"First 16 bytes (hex): {header.hex()}")
        print(f"First 16 bytes (ascii): {header}")
        
        # Check for common file signatures
        if header.startswith(b'MATLAB'):
            print("✓ MATLAB file signature detected")
            return "matlab_old"
        elif header.startswith(b'\x89HDF'):
            print("✓ HDF5 file signature detected (MATLAB v7.3)")
            return "matlab_v73"
        elif header[:8] == b'\x00\x00\x00\x00\x00\x00\x00\x00':
            print("⚠ File appears to be empty or corrupted (all zeros)")
            return "corrupted"
        else:
            print("❌ Unknown file signature")
            return "unknown"

def try_matlab_formats(file_path):
    """Try different methods to read the MATLAB file"""
    print(f"\n=== Attempting to read {file_path} ===")
    
    # Method 1: Standard scipy.io.loadmat
    print("\n1. Trying scipy.io.loadmat...")
    try:
        # First, check what variables are in the file
        mat_info = io.whosmat(file_path)
        print(f"Variables found: {mat_info}")
        
        # Try to load the file
        mat_data = io.loadmat(file_path)
        print(f"✓ Successfully loaded with scipy! Keys: {list(mat_data.keys())}")
        return mat_data, "scipy"
    except Exception as e:
        print(f"❌ scipy.io.loadmat failed: {e}")
    
    # Method 2: Try h5py (for v7.3 files)
    print("\n2. Trying h5py...")
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"✓ Successfully opened with h5py! Keys: {keys}")
            return f, "h5py"
    except Exception as e:
        print(f"❌ h5py failed: {e}")
    
    # Method 3: Try loading specific variables only
    print("\n3. Trying to load specific variables...")
    try:
        mat_data = io.loadmat(file_path, variable_names=['combined_genotypes'])
        print(f"✓ Successfully loaded 'combined_genotypes' variable!")
        return mat_data, "scipy_specific"
    except Exception as e:
        print(f"❌ Loading specific variable failed: {e}")
    
    return None, None

def analyze_file_content(file_path):
    """Analyze the content of the file to understand its structure"""
    print(f"\n=== Content Analysis ===")
    
    try:
        # Try to read as binary and look for patterns
        with open(file_path, 'rb') as f:
            # Read first 1KB
            chunk = f.read(1024)
            
            # Look for MATLAB variable names
            if b'combined_genotypes' in chunk:
                print("✓ Found 'combined_genotypes' variable name in file")
            else:
                # Search further into the file
                f.seek(0)
                first_mb = f.read(1024 * 1024)  # Read first MB
                if b'combined_genotypes' in first_mb:
                    print("✓ Found 'combined_genotypes' variable name in first MB")
                else:
                    print("❌ 'combined_genotypes' variable name not found")
            
            # Check for text patterns that might indicate data format
            if b'ATCG' in first_mb or b'AUCG' in first_mb:
                print("✓ Found nucleotide sequences in file")
            
    except Exception as e:
        print(f"❌ Content analysis failed: {e}")

def create_test_file(output_path):
    """Create a test file to verify our processing works"""
    print(f"\n=== Creating test file: {output_path} ===")
    
    # Create sample genetic data
    n_subjects = 100
    n_snps = 50
    
    # Generate random genotypes
    alleles = ['A', 'T', 'C', 'G']
    genotypes = []
    
    for i in range(n_subjects):
        subject_genotypes = []
        for j in range(n_snps):
            # Random genotype (could be homozygous or heterozygous)
            if np.random.random() < 0.3:  # 30% homozygous
                allele = np.random.choice(alleles)
                genotype = allele + allele
            else:  # 70% heterozygous
                genotype = ''.join(np.random.choice(alleles, 2))
            subject_genotypes.append(genotype)
        genotypes.append(subject_genotypes)
    
    # Convert to numpy array
    combined_genotypes = np.array(genotypes, dtype=object)
    
    # Save as .mat file
    io.savemat(output_path, {'combined_genotypes': combined_genotypes})
    print(f"✓ Test file created with shape: {combined_genotypes.shape}")
    
    return output_path

def recover_data_alternative(file_path):
    """Try alternative data recovery methods"""
    print(f"\n=== Alternative Recovery Methods ===")
    
    # Method 1: Try reading with different scipy options
    print("1. Trying scipy with different options...")
    try:
        # Try without squeeze_me
        mat_data = io.loadmat(file_path, squeeze_me=False)
        print("✓ Loaded with squeeze_me=False")
        return mat_data
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    try:
        # Try with struct_as_record=False
        mat_data = io.loadmat(file_path, struct_as_record=False)
        print("✓ Loaded with struct_as_record=False")
        return mat_data
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Method 2: Try reading raw binary data
    print("\n2. Trying raw binary analysis...")
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            print(f"Read {len(data)} bytes of raw data")
            # This is just for diagnostic purposes
            return None
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Diagnose and recover MATLAB files")
    parser.add_argument('-file', type=str, required=True, help='Path to the .mat file to diagnose')
    parser.add_argument('-create_test', action='store_true', help='Create a test file for verification')
    args = parser.parse_args()
    
    file_path = args.file
    
    print("=== MATLAB File Diagnostic Tool ===")
    
    # Step 1: Check file signature
    signature = check_file_signature(file_path)
    
    # Step 2: Try different reading methods
    data, method = try_matlab_formats(file_path)
    
    # Step 3: Analyze content
    analyze_file_content(file_path)
    
    # Step 4: Try recovery methods
    if data is None:
        print("\n=== File appears corrupted. Trying recovery methods ===")
        recovered_data = recover_data_alternative(file_path)
        if recovered_data is not None:
            data = recovered_data
            method = "recovered"
    
    # Step 5: Create test file if requested
    if args.create_test:
        test_file = file_path.replace('.mat', '_test.mat')
        create_test_file(test_file)
        
        # Test our processing on the test file
        print(f"\n=== Testing processing on test file ===")
        test_data, test_method = try_matlab_formats(test_file)
        if test_data is not None:
            print("✓ Test file processing successful!")
    
    # Summary
    print(f"\n=== DIAGNOSIS SUMMARY ===")
    print(f"File: {file_path}")
    print(f"Signature: {signature}")
    print(f"Readable: {'Yes' if data is not None else 'No'}")
    print(f"Method: {method if data is not None else 'None'}")
    
    if data is None:
        print(f"\n❌ RECOMMENDATIONS:")
        print(f"1. Check if the file is corrupted")
        print(f"2. Try copying the file to a different location")
        print(f"3. Check available disk space and memory")
        print(f"4. Contact the data provider for a fresh copy")
        print(f"5. Consider using the test file generation feature")
    else:
        print(f"\n✓ File is readable! You can proceed with processing.")

if __name__ == "__main__":
    main()