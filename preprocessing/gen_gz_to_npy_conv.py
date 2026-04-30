#!/usr/bin/env python3
"""
Convert .gen.gz genomic files to .npy format for faster loading
"""

import numpy as np
import pandas as pd
import os
import gzip
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def analyze_genotype_values(input_file, sample_size=10000):
    """Analyze the unique values in genotype probability data"""
    df = pd.read_csv(input_file, compression='gzip', header=None, nrows=sample_size)
    
    unique_vals = set()
    for col in df.columns:
        unique_vals.update(df[col].unique())
    
    unique_vals = sorted(list(unique_vals))
    print(f"Unique values found (sample of {sample_size} rows): {unique_vals}")
    print(f"Number of unique values: {len(unique_vals)}")
    
    # Check if rows sum to 1 (probability constraint)
    row_sums = df.sum(axis=1)
    #print(f"Row sums range: {row_sums.min():.6f} to {row_sums.max():.6f}")
    
    return unique_vals

def convert_single_file(input_file, output_dir, dtype='float16', compress=False):
    """Convert a single .gen.gz file to .npy format"""
    try:
        # Read the compressed file
        df = pd.read_csv(input_file, compression='gzip', header=None, 
                        dtype=np.float32, sep=' ')
        
        # Convert to numpy array with specified dtype
        data = df.values.astype(dtype)
        
        # Verify probability constraints (optional quality check)
        row_sums = np.sum(data, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            print(f"Warning: Some rows in {input_file} don't sum to 1.0")
        
        # Create output filename
        input_path = Path(input_file)
        if compress:
            output_file = Path(output_dir) / f"{input_path.stem.replace('.gen', '')}.npz"
            np.savez_compressed(output_file, data=data)
        else:
            output_file = Path(output_dir) / f"{input_path.stem.replace('.gen', '')}.npy"
            np.save(output_file, data)
        
        # Calculate compression ratio
        original_size = Path(input_file).stat().st_size / (1024**2)  # MB
        new_size = output_file.stat().st_size / (1024**2)  # MB
        ratio = original_size / new_size if new_size > 0 else 0
        
        return f"✓ {input_path.name} -> {output_file.name} ({ratio:.1f}x size reduction)"
        
    except Exception as e:
        return f"✗ Error converting {input_file}: {str(e)}"

def batch_convert_parallel(input_dir, output_dir, dtype='float16', num_workers=4, compress=False):
    """Convert all .gen.gz files in parallel"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .gen.gz files
    input_files = list(Path(input_dir).glob("*.gen.gz"))
    print(f"Found {len(input_files)} .gen.gz files to convert")
    
    if not input_files:
        print("No .gen.gz files found!")
        return
    
    # Test conversion on first file to estimate sizes
    print("Testing conversion on first file...")
    test_result = convert_single_file(input_files[0], output_dir, dtype, compress)
    print(test_result)
    
    # Analyze the data characteristics
    print("\nAnalyzing genotype probability data...")
    analyze_genotype_values(input_files[0])
    
    # Get file sizes for estimation
    ext = '.npz' if compress else '.npy'
    test_file = Path(output_dir) / f"{Path(input_files[0]).stem.replace('.gen', '')}{ext}"
    if test_file.exists():
        npy_size = test_file.stat().st_size / (1024**2)  # MB
        total_estimated = npy_size * len(input_files) / 1024  # GB
        print(f"Estimated total size: {total_estimated:.1f} GB")
        
        # proceed = input(f"Continue with conversion of {len(input_files)} files? (y/n): ")
        # if proceed.lower() != 'y':
        #     return
    
    # Convert remaining files in parallel
    convert_func = partial(convert_single_file, output_dir=output_dir, dtype=dtype, compress=compress)
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_func, input_files[1:]), 
            total=len(input_files)-1,
            desc="Converting files"
        ))
    
    # Print results
    success_count = sum(1 for r in results if r.startswith("✓"))
    print(f"\nConversion complete: {success_count+1}/{len(input_files)} files successful")

def verify_conversion(original_file, npy_file):
    """Verify that conversion preserved data correctly"""
    # Load original
    original = pd.read_csv(original_file, compression='gzip', header=None, sep=' ').values
    
    # Load converted (handle both .npy and .npz)
    if npy_file.suffix == '.npz':
        converted = np.load(npy_file)['data']
    else:
        converted = np.load(npy_file)
    
    # Compare with tolerance for floating point precision
    if np.allclose(original, converted, rtol=1e-5, atol=1e-6):
        print("✓ Conversion verified - data matches within tolerance")
        return True
    else:
        print("✗ Conversion error - data mismatch!")
        print(f"Max difference: {np.max(np.abs(original - converted))}")
        return False

def compare_loading_speed(gen_file, npy_file, iterations=5):
    """Compare loading speeds between .gen.gz and .npy/.npz files"""
    import time
    
    print(f"Comparing loading speeds ({iterations} iterations each)...")
    
    # Time .gen.gz loading
    gen_times = []
    for _ in range(iterations):
        start = time.time()
        df = pd.read_csv(gen_file, compression='gzip', header=None, sep=' ')
        gen_times.append(time.time() - start)
    
    # Time .npy/.npz loading  
    npy_times = []
    for _ in range(iterations):
        start = time.time()
        if npy_file.suffix == '.npz':
            data = np.load(npy_file)['data']
        else:
            data = np.load(npy_file)
        npy_times.append(time.time() - start)
    
    gen_avg = np.mean(gen_times)
    npy_avg = np.mean(npy_times)
    speedup = gen_avg / npy_avg
    
    print(f".gen.gz average: {gen_avg:.3f}s")
    print(f"{npy_file.suffix} average: {npy_avg:.3f}s") 
    print(f"Speedup: {speedup:.1f}x faster")
    
    # File size comparison
    gen_size = Path(gen_file).stat().st_size / (1024**2)
    npy_size = Path(npy_file).stat().st_size / (1024**2)
    size_ratio = gen_size / npy_size
    
    print(f"\nFile sizes:")
    print(f".gen.gz: {gen_size:.1f} MB")
    print(f"{npy_file.suffix}: {npy_size:.1f} MB")
    print(f"Size ratio: {size_ratio:.1f}x {'smaller' if size_ratio < 1 else 'larger'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert genomic .gen.gz files to .npy format")
    parser.add_argument("-input_dir", type=str, default="/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq", help="Directory containing .gen.gz files")
    parser.add_argument("-output_dir", type=str, default="/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy", help="Directory to save .npy files")
    
    # parser.add_argument("-input_dir", type=str, default="/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq", help="Directory containing .gen.gz files")
    # parser.add_argument("-output_dir", type=str, default="/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy", help="Directory to save .npy files")
    
    parser.add_argument("-dtype", default="float16", choices=["float16", "float32"], help="Data type for numpy arrays (default: float16)")
    parser.add_argument("-workers", type=int, default=64, help="Number of parallel workers")
    parser.add_argument("-compress", type=int, choices=[0, 1], default=0, help="Use .npz compression (0=False, 1=True, default: 0)")
    parser.add_argument("-verify", type=int, choices=[0, 1], default=1, help="Verify conversion on first file (0=False, 1=True, default: 0)")
    parser.add_argument("-benchmark", type=int, choices=[0, 1], default=1, help="Benchmark loading speeds (0=False, 1=True, default: 0)")
    parser.add_argument("-analyze", type=int, choices=[0, 1], default=1, help="Analyze data characteristics only (0=False, 1=True, default: 0)")
    
    args = parser.parse_args()
    
    # Convert arguments from 0/1 to boolean for internal use
    compress = bool(args.compress)
    verify = bool(args.verify)
    benchmark = bool(args.benchmark)
    analyze = bool(args.analyze)
    
    # Just analyze data if requested
    if analyze:
        input_files = list(Path(args.input_dir).glob("*.gen.gz"))
        if input_files:
            print("Analyzing first file...")
            analyze_genotype_values(input_files[0])
        #return
    
    # Convert files
    batch_convert_parallel(args.input_dir, args.output_dir, args.dtype, args.workers, compress)
    
    # Optional verification
    if verify:
        input_files = list(Path(args.input_dir).glob("*.gen.gz"))
        if input_files:
            first_file = input_files[0]
            ext = '.npz' if compress else '.npy'
            npy_file = Path(args.output_dir) / f"{first_file.stem.replace('.gen', '')}{ext}"
            if npy_file.exists():
                verify_conversion(first_file, npy_file)
    
    # Optional benchmarking
    if benchmark:
        input_files = list(Path(args.input_dir).glob("*.gen.gz"))
        if input_files:
            first_file = input_files[0]
            ext = '.npz' if compress else '.npy'
            npy_file = Path(args.output_dir) / f"{first_file.stem.replace('.gen', '')}{ext}"
            if npy_file.exists():
                compare_loading_speed(first_file, npy_file)