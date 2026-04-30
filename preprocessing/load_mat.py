import numpy as np
from scipy import io
import gc
import os

def test_load_small_chunk(file_path):
    """Test loading just a tiny piece of the data"""
    print(f"Testing file: {file_path}")
    
    # Get file info
    try:
        mat_info = io.whosmat(file_path)
        print(f"File info: {mat_info}")
        
        for var_name, shape, dtype in mat_info:
            if var_name == 'combined_genotypes':
                n_subjects, n_snps = shape
                print(f"Found data: {n_subjects} subjects, {n_snps} SNPs, type: {dtype}")
                break
        else:
            print("ERROR: combined_genotypes not found")
            return False
    
    except Exception as e:
        print(f"ERROR reading file info: {e}")
        return False
    
    # Try to load just the first 10 subjects
    print("\nTrying to load first 10 subjects...")
    try:
        # Load entire file (risky but let's try)
        mat_data = io.loadmat(file_path)
        combined_genotypes = mat_data['combined_genotypes']
        
        print(f"SUCCESS! Loaded full data: {combined_genotypes.shape}")
        print(f"Data type: {combined_genotypes.dtype}")
        
        # Extract small sample
        sample = combined_genotypes[:10, :10]
        print(f"Sample data (first 10x10):")
        print(sample)
        
        # Check data types
        print(f"\nSample element types:")
        for i in range(min(3, sample.shape[0])):
            for j in range(min(3, sample.shape[1])):
                element = sample[i, j]
                print(f"  [{i},{j}]: '{element}' (type: {type(element)})")
        
        # Clean up
        del mat_data, combined_genotypes
        gc.collect()
        
        return True
        
    except MemoryError:
        print("MEMORY ERROR: File too large to load completely")
        return False
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return False

def create_tiny_test_file(output_path):
    """Create a tiny test file to verify our processing works"""
    print(f"\nCreating test file: {output_path}")
    
    # Create small dataset
    n_subjects = 100
    n_snps = 50
    
    # Generate realistic genotype data
    genotypes = []
    alleles = ['A', 'T', 'C', 'G']
    
    for i in range(n_subjects):
        subject_genotypes = []
        for j in range(n_snps):
            if np.random.random() < 0.3:  # 30% homozygous
                allele = np.random.choice(alleles)
                genotype = allele  # Single character for homozygous
            else:  # 70% heterozygous
                genotype = ''.join(np.random.choice(alleles, 2))
            subject_genotypes.append(genotype)
        genotypes.append(subject_genotypes)
    
    # Convert to object array (like your original data)
    combined_genotypes = np.array(genotypes, dtype=object)
    
    # Save as .mat file
    io.savemat(output_path, {'combined_genotypes': combined_genotypes})
    
    print(f"✓ Created test file with shape: {combined_genotypes.shape}")
    print(f"Sample data:")
    print(combined_genotypes[:5, :5])
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_test.py <path_to_mat_file>")
        print("   or: python simple_test.py create_test <output_path>")
        sys.exit(1)
    
    if sys.argv[1] == "create_test":
        if len(sys.argv) < 3:
            output_path = "test_genetic_data.mat"
        else:
            output_path = sys.argv[2]
        
        test_file = create_tiny_test_file(output_path)
        print(f"\nTesting the created file...")
        test_load_small_chunk(test_file)
    else:
        file_path = sys.argv[1]
        success = test_load_small_chunk(file_path)
        
        if success:
            print("\n✅ SUCCESS! File can be loaded. Your data format is compatible.")
        else:
            print("\n❌ FAILED! File cannot be loaded due to memory or corruption issues.")
            
            # Offer to create test file
            print("\nWould you like to create a test file? Run:")
            print(f"python {sys.argv[0]} create_test test_data.mat")