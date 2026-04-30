import os
import gzip
import numpy as np
from scipy import io
import multiprocessing as mp
from multiprocessing import shared_memory
import argparse
import pickle
import tempfile

parser = argparse.ArgumentParser(description="Merge and filter .gen.gz files")
parser.add_argument('-reference_file', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen', help='Path to the folder containing .mat sample files')
parser.add_argument('-sample_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/pros', help='Path to the output folder')
parser.add_argument('-output_base_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_disease_wise/pros_seq_2', help='Path to the metadata file')
parser.add_argument('-chunk_size', type=int, default=1000, help='Number of samples to process in each chunk')
parser.add_argument('-max_processes', type=int, default=None, help='Maximum number of processes to use (default: CPU count - 1)')

args = parser.parse_args()

def load_reference_data(reference_file):
    """
    Load reference data efficiently with proper chromosome handling.
    Modified to detect all chromosomes present in the file.
    """
    print(f"Loading reference data from {reference_file}...")
    
    # Use memory-efficient approach with numpy
    import numpy as np
    
    # Pre-define all possible chromosomes (1-22, X, Y, MT)
    # This ensures we catch all chromosomes even if they're not in the sample
    all_possible_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    chromosome_set = set(all_possible_chromosomes)
    
    # First count total lines
    line_count = 0
    with open(reference_file, 'r') as f:
        for _ in f:
            line_count += 1
    
    print(f"Total lines in reference file: {line_count}")
    print(f"Pre-allocating for all possible chromosomes (1-22, X, Y, MT)")
    
    # Initialize data structures with pre-allocated capacity
    # Allocate conservatively - we'll resize if needed
    initial_allocation = max(10000, line_count // len(chromosome_set) * 2)
    chr_to_line_map = {chr: np.zeros(initial_allocation, dtype=np.int32) for chr in chromosome_set}
    chr_filled = {chr: 0 for chr in chromosome_set}
    chr_to_alleles = {chr: [] for chr in chromosome_set}
    
    # Process in chunks to avoid memory issues
    chunk_size = 1000000  # Read 1M lines at a time
    
    # Track chromosomes actually found in the file
    chromosomes_found = set()
    
    with open(reference_file, 'r') as f:
        for chunk_start in range(0, line_count, chunk_size):
            print(f"Processing reference data chunk {chunk_start//chunk_size + 1}/{(line_count + chunk_size - 1)//chunk_size}")
            
            # Read chunk of lines
            chunk_end = min(chunk_start + chunk_size, line_count)
            lines_to_read = chunk_end - chunk_start
            
            for i in range(lines_to_read):
                line = next(f)
                parts = line.strip().split()
                chromosome = parts[0]
                chromosomes_found.add(chromosome)
                
                # If we encounter a chromosome not in our predefined list, add it
                if chromosome not in chr_to_line_map:
                    print(f"Found new chromosome {chromosome} not in predefined list")
                    chr_to_line_map[chromosome] = np.zeros(initial_allocation, dtype=np.int32)
                    chr_filled[chromosome] = 0
                    chr_to_alleles[chromosome] = []
                
                # Check if we need to resize the array
                if chr_filled[chromosome] >= len(chr_to_line_map[chromosome]):
                    # Double the size of the array
                    old_size = len(chr_to_line_map[chromosome])
                    new_size = old_size * 2
                    print(f"Resizing array for chromosome {chromosome} from {old_size} to {new_size}")
                    new_array = np.zeros(new_size, dtype=np.int32)
                    new_array[:old_size] = chr_to_line_map[chromosome]
                    chr_to_line_map[chromosome] = new_array
                
                # Store line index
                current_idx = chr_filled[chromosome]
                chr_to_line_map[chromosome][current_idx] = chunk_start + i
                
                # Store alleles
                chr_to_alleles[chromosome].append((parts[3], parts[4]))  # (Ref Allele, Alt Allele)
                
                # Update count
                chr_filled[chromosome] += 1
    
    # Trim arrays to actual size and remove chromosomes not found
    for chromosome in list(chr_to_line_map.keys()):
        if chromosome in chromosomes_found:
            actual_count = chr_filled[chromosome]
            if actual_count > 0:
                chr_to_line_map[chromosome] = chr_to_line_map[chromosome][:actual_count]
            else:
                # Remove chromosomes with no entries
                del chr_to_line_map[chromosome]
                del chr_to_alleles[chromosome]
        else:
            # Remove chromosomes not found in the file
            del chr_to_line_map[chromosome]
            del chr_to_alleles[chromosome]
    
    # Create shared reference data
    ref_data = {
        'chr_to_line_map': chr_to_line_map,
        'chr_to_alleles': chr_to_alleles
    }
    
    # Print memory usage statistics
    import sys
    memory_usage = sum(sys.getsizeof(arr) for arr in chr_to_line_map.values())
    print(f"Reference data loaded. Found data for {len(chr_to_line_map)} chromosomes: {', '.join(sorted(chr_to_line_map.keys()))}")
    print(f"Estimated memory usage for indices: {memory_usage / (1024*1024):.2f} MB")
    
    return ref_data

def save_reference_to_temp(ref_data):
    """Save reference data to a temporary file and return the filename."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp_file.name, 'wb') as f:
        pickle.dump(ref_data, f)
    return temp_file.name

def load_reference_from_temp(temp_file):
    """Load reference data from a temporary file."""
    with open(temp_file, 'rb') as f:
        return pickle.load(f)

def process_sample_file_chunked(args):
    """Process a sample file in a memory-efficient way using numpy arrays for faster lookups."""
    sample_file, ref_temp_file, output_base_folder = args
    
    # Load reference data from temporary file
    ref_data = load_reference_from_temp(ref_temp_file)
    chr_to_line_map = ref_data['chr_to_line_map']
    chr_to_alleles = ref_data['chr_to_alleles']
    
    base_name = os.path.basename(sample_file)
    results = []
    
    # Create a dictionary to store genotypes by chromosome
    chr_genotypes = {chrom: [] for chrom in chr_to_line_map.keys()}
    
    # Create a combined line index to chromosome mapping for faster lookups
    line_to_chr = {}
    line_to_chr_idx = {}
    
    # This is memory-intensive but saves significant processing time
    print(f"Building index lookup table for {sample_file}...")
    for chromosome, indices in chr_to_line_map.items():
        for chr_idx, line_idx in enumerate(indices):
            line_to_chr[line_idx] = chromosome
            line_to_chr_idx[line_idx] = chr_idx
    
    # Process the file using buffered IO for better performance
    total_lines = 0
    processed_lines = 0
    
    # Count lines first (optional - for progress reporting)
    try:
        with gzip.open(sample_file, 'rt') as in_f:
            for _ in in_f:
                total_lines += 1
    except Exception as e:
        print(f"Warning: Could not count lines in {sample_file}: {e}")
        total_lines = -1  # Unknown line count
    
    # Process in chunks for memory efficiency and better reporting
    try:
        with gzip.open(sample_file, 'rt') as in_f:
            # Use a buffer to read larger chunks at once
            chunk_size = 100000  # Adjust based on your memory constraints
            
            lines_read = 0
            while True:
                # Read chunk of lines
                lines = []
                for _ in range(chunk_size):
                    line = next(in_f, None)
                    if line is None:
                        break
                    lines.append(line)
                
                if not lines:
                    break  # End of file
                
                # Process each line in the chunk
                for i, line in enumerate(lines):
                    line_idx = lines_read + i
                    processed_lines += 1
                    
                    # Use direct lookup instead of iteration
                    if line_idx in line_to_chr:
                        chromosome = line_to_chr[line_idx]
                        chr_idx = line_to_chr_idx[line_idx]
                        
                        try:
                            # Parse probabilities from the line
                            probs = list(map(float, line.strip().split()))
                            
                            # Get reference and alternate alleles
                            ref_allele, alt_allele = chr_to_alleles[chromosome][chr_idx]
                            
                            # Determine genotype
                            if max(probs) == probs[0]:  # Homozygous dominant
                                genotype = ref_allele + ref_allele
                            elif max(probs) == probs[1]:  # Heterozygous
                                genotype = ref_allele + alt_allele
                            else:  # Homozygous recessive
                                genotype = alt_allele + alt_allele
                            
                            # Store genotype for this chromosome
                            chr_genotypes[chromosome].append(genotype)
                        except Exception as e:
                            print(f"Warning: Error processing line {line_idx} for chromosome {chromosome}: {e}")
                
                lines_read += len(lines)
                
                # Report progress
                if total_lines > 0:
                    print(f"\rProcessing {sample_file}: {processed_lines}/{total_lines} lines "
                        f"({processed_lines/total_lines*100:.1f}%)", end="")
                else:
                    print(f"\rProcessing {sample_file}: {processed_lines} lines", end="")
        
        print()  # New line after progress reporting
    
    except Exception as e:
        print(f"Error processing file {sample_file}: {e}")
        return [f"Error processing file {sample_file}: {e}"]
    
    # Save genotypes for each chromosome and count SNPs
    print(f"\nSNP counts for {base_name}:")
    total_snps = 0
    chr_snp_counts = []  # For sorted output
    
    for chromosome, genotypes in sorted(chr_genotypes.items(), key=lambda x: (not x[0].isdigit(), int(x[0]) if x[0].isdigit() else x[0])):
        snp_count = len(genotypes)
        total_snps += snp_count
        chr_snp_counts.append((chromosome, snp_count))

        if genotypes:
            # Convert genotypes to a single row of space-separated genotypes
            genotypes_row = ' '.join(genotypes)
            
            # Create chromosome-specific output folder
            output_folder = os.path.join(output_base_folder, f"chromosome_{chromosome}")
            os.makedirs(output_folder, exist_ok=True)
            
            # Save as .mat file
            output_file = os.path.join(output_folder, base_name.replace('.gen.gz', f'_chr{chromosome}.mat'))
            io.savemat(output_file, {'genotypes': genotypes_row})
            
            results.append(f"Processed and saved chromosome {chromosome} SNPs as .mat: {output_file}")
        else:
            results.append(f"No SNPs found for chromosome {chromosome} in {base_name}")
    
    # Print SNP counts in chromosome order
    max_chr_len = max(len(chr) for chr, _ in chr_snp_counts) if chr_snp_counts else 0
    for chromosome, count in chr_snp_counts:
        print(f"  Chromosome {chromosome:{max_chr_len}}: {count:,} SNPs")
    print(f"  Total: {total_snps:,} SNPs")
    
    # Clean up large temporary data structures
    del line_to_chr
    del line_to_chr_idx
    
    return results

def process_files_in_chunks(sample_files, ref_temp_file, output_base_folder, chunk_size, max_processes):
    """Process files in chunks to control memory usage."""
    # Determine number of processes to use
    n_processes = min(max_processes if max_processes else mp.cpu_count() - 1, mp.cpu_count() - 1)
    n_processes = max(1, n_processes)  # Ensure at least one process
    
    print(f"Processing with {n_processes} processes, chunk size: {chunk_size}")
    
     # Summary stats
    total_files = len(sample_files)
    files_processed = 0
    file_stats = []
    
    # Process files in chunks
    for i in range(0, len(sample_files), chunk_size):
        chunk = sample_files[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(sample_files) + chunk_size - 1)//chunk_size} " 
              f"({len(chunk)} files, {i+1}-{min(i+chunk_size, len(sample_files))} of {len(sample_files)})")
        
        # Prepare arguments for multiprocessing
        args_list = [(file, ref_temp_file, output_base_folder) for file in chunk]
        
        # Use multiprocessing to process files in this chunk
        with mp.Pool(processes=n_processes) as pool:
            chunk_results = pool.map(process_sample_file_chunked, args_list)
        
        # Update counts
        files_processed += len(chunk)
        
        # Print summary header if this is the first chunk
        if i == 0:
            print("\n" + "=" * 80)
            print("SAMPLE PROCESSING SUMMARY")
            print("=" * 80)
        
        # Print progress summary
        print(f"\nProgress: {files_processed}/{total_files} files ({files_processed/total_files*100:.1f}%)")
        
        # Explicitly call garbage collection to free memory
        import gc
        gc.collect()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {files_processed}")
    print(f"All files processed and saved as .mat files for each chromosome.")
    print("=" * 80)

def main(reference_file, sample_folder, output_base_folder, chunk_size=1000, max_processes=None):
    # Load reference data
    print("Loading reference data and identifying SNPs for all chromosomes...")
    ref_data = load_reference_data(reference_file)
    
    # Save reference data to temporary file to avoid duplicating it in each process
    ref_temp_file = save_reference_to_temp(ref_data)
    print(f"Reference data saved to temporary file: {ref_temp_file}")
    
    try:
        # Get full paths of sample files
        sample_files = [os.path.join(sample_folder, f) 
                       for f in os.listdir(sample_folder) 
                       if f.endswith('.gen.gz') and f.startswith('sample_')]
        
        print(f"Found {len(sample_files)} sample files to process")
        
        # Process files in chunks
        process_files_in_chunks(sample_files, ref_temp_file, output_base_folder, chunk_size, max_processes)
        
        print("All files processed and saved as .mat files for each chromosome.")
    
    finally:
        # Clean up the temporary file
        try:
            os.unlink(ref_temp_file)
            print(f"Temporary reference data file removed: {ref_temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {ref_temp_file}: {e}")

if __name__ == "__main__":
    main(args.reference_file, 
         args.sample_folder, 
         args.output_base_folder, 
         args.chunk_size,
         args.max_processes)