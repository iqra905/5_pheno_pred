import os
import gzip
import glob
from multiprocessing import Pool
import numpy as np

def encode_genotype_probs(aa_prob, ab_prob, bb_prob):
    """
    Encode three genotype probabilities into a single numeric code.
    Returns 0 for AA, 1 for AB, 2 for BB based on highest probability.
    """
    probs = [aa_prob, ab_prob, bb_prob]
    return probs.index(max(probs))

def process_file(input_file):
    """
    Process a single .gen.gz file and save the encoded version
    """
    try:
        # Create output path
        output_dir = "/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_encoded"
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Read and process the file
        encoded_data = []
        with gzip.open(input_file, 'rt') as f:
            for line in f:
                # Convert line to floats
                probs = [np.float16(x) for x in line.strip().split()]
                if len(probs) != 3:
                    raise ValueError(f"Expected 3 probabilities per line, got {len(probs)}")
                
                # Encode genotype
                encoded = encode_genotype_probs(*probs)
                encoded_data.append(encoded)
        
        # Save encoded data
        with gzip.open(output_file, 'wt') as f:
            for encoded in encoded_data:
                f.write(f"{encoded}\n")
                
        return f"Successfully processed {input_file}"
    
    except Exception as e:
        return f"Error processing {input_file}: {str(e)}"

def main():
    # Input directory
    input_dir = "/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d"
    
    # Get all sample_*.gen.gz files
    input_files = glob.glob(os.path.join(input_dir, "sample_*.gen.gz"))
    
    if not input_files:
        print("No input files found!")
        return
    
    # Number of CPU cores to use (leave one core free)
    n_cores = max(1, os.cpu_count() - 1)
    
    print(f"Found {len(input_files)} files to process")
    print(f"Using {n_cores} CPU cores")
    
    # Process files in parallel
    with Pool(n_cores) as pool:
        results = pool.map(process_file, input_files)
    
    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()