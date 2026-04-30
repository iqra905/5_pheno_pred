#!/usr/bin/env python3
"""
Convert existing HDF5 genetic data files to 3D format
Reads chr*.h5 files in chunks and converts to chr*_3d.h5
"""

import h5py
import numpy as np
import os
import time
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser(description="Convert HDF5 genetic data to 3D format")
parser.add_argument('-input_folder', type=str, required=True, help='Folder with chr*.h5 files')
parser.add_argument('-output_folder', type=str, required=True, help='Output folder for chr*_3d.h5 files')
parser.add_argument('-chunk_size', type=int, default=100, help='Number of subjects to process at once')
parser.add_argument('-chromosomes', type=str, default='1-22', help='Chromosomes to process (e.g., 1-22, 1,2,3)')
parser.add_argument('-data_key', type=str, default='combined_genotypes', help='Key name in HDF5 file for genetic data')
parser.add_argument('-output_format', type=str, choices=['string', 'numeric'], default='string', 
                   help='Output format: string (A,T,C,G) or numeric (0,1,2,3)')
args = parser.parse_args()

def parse_chromosome_range(chrom_str):
    """Parse chromosome specification like '1-22' or '1,2,3'"""
    chromosomes = []
    for part in chrom_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            chromosomes.extend(range(start, end + 1))
        else:
            chromosomes.append(int(part))
    return sorted(set(chromosomes))

def inspect_h5_file(filename):
    """Inspect HDF5 file structure and return dataset info"""
    print(f"Inspecting {filename}...")
    
    try:
        with h5py.File(filename, 'r') as f:
            print(f"  File keys: {list(f.keys())}")
            
            # Look for the genetic data
            data_key = None
            possible_keys = ['combined_genotypes', 'genotype_data', 'data', 'genotypes']
            
            for key in possible_keys:
                if key in f:
                    data_key = key
                    break
            
            if data_key is None:
                # Try the first non-metadata key
                for key in f.keys():
                    if not key.startswith('__'):
                        data_key = key
                        break
            
            if data_key is None:
                raise ValueError("No suitable data key found")
            
            dataset = f[data_key]
            print(f"  Found data key: '{data_key}'")
            print(f"  Shape: {dataset.shape}")
            print(f"  Data type: {dataset.dtype}")
            
            # Get sample data to understand format
            if len(dataset.shape) == 2:
                sample = dataset[0:min(3, dataset.shape[0]), 0:min(5, dataset.shape[1])]
                print(f"  Sample data:")
                for i, row in enumerate(sample):
                    print(f"    Subject {i}: {row}")
            
            return {
                'data_key': data_key,
                'shape': dataset.shape,
                'dtype': dataset.dtype,
                'n_subjects': dataset.shape[0],
                'n_snps': dataset.shape[1] if len(dataset.shape) > 1 else 0
            }
    
    except Exception as e:
        print(f"  ❌ Error inspecting file: {e}")
        return None

def convert_genotype_chunk_to_3d(chunk_2d, output_format='string'):
    """Convert a 2D chunk to 3D format"""
    n_subjects, n_snps = chunk_2d.shape
    
    if output_format == 'string':
        output_3d = np.full((n_subjects, n_snps, 2), 'N', dtype='U1')
    else:  # numeric
        output_3d = np.full((n_subjects, n_snps, 2), 4, dtype=np.uint8)  # 4 = 'N'
        allele_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    
    for i in range(n_subjects):
        for j in range(n_snps):
            # Get genotype and handle different data types
            genotype = chunk_2d[i, j]
            
            if isinstance(genotype, (np.bytes_, bytes)):
                genotype = genotype.decode('utf-8')
            elif isinstance(genotype, np.ndarray):
                if genotype.size == 1:
                    genotype = str(genotype.item())
                else:
                    genotype = str(genotype)
            else:
                genotype = str(genotype)
            
            # Clean up genotype
            genotype = genotype.strip().upper()
            
            # Convert to 3D format
            if len(genotype) == 1:
                # Homozygous
                if output_format == 'string':
                    output_3d[i, j, 0] = genotype
                    output_3d[i, j, 1] = genotype
                else:
                    code = allele_map.get(genotype, 4)
                    output_3d[i, j, 0] = code
                    output_3d[i, j, 1] = code
            elif len(genotype) == 2:
                # Heterozygous
                if output_format == 'string':
                    output_3d[i, j, 0] = genotype[0]
                    output_3d[i, j, 1] = genotype[1]
                else:
                    code1 = allele_map.get(genotype[0], 4)
                    code2 = allele_map.get(genotype[1], 4)
                    output_3d[i, j, 0] = code1
                    output_3d[i, j, 1] = code2
            # else: remains 'N' or 4 for missing data
    
    return output_3d

def convert_h5_to_3d(input_file, output_file, chunk_size=100, data_key='combined_genotypes', output_format='string'):
    """Convert HDF5 genetic data to 3D format using chunked processing"""
    
    print(f"\n{'='*60}")
    print(f"Converting {os.path.basename(input_file)} to 3D format")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Chunk size: {chunk_size} subjects")
    print(f"Output format: {output_format}")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    try:
        # Open input file and get dataset info
        with h5py.File(input_file, 'r') as input_f:
            
            # Check if data_key exists, if not try to find it
            if data_key not in input_f:
                print(f"Key '{data_key}' not found. Available keys: {list(input_f.keys())}")
                # Try common alternatives
                for alt_key in ['genotype_data', 'data', 'genotypes', 'combined_genotypes']:
                    if alt_key in input_f:
                        data_key = alt_key
                        print(f"Using alternative key: '{data_key}'")
                        break
                else:
                    # Use first non-metadata key
                    for key in input_f.keys():
                        if not key.startswith('__'):
                            data_key = key
                            print(f"Using first available key: '{data_key}'")
                            break
                    else:
                        raise ValueError("No suitable data key found")
            
            input_dataset = input_f[data_key]
            n_subjects, n_snps = input_dataset.shape
            
            print(f"Input data shape: ({n_subjects:,}, {n_snps:,})")
            print(f"Input data type: {input_dataset.dtype}")
            
            # Create output file
            with h5py.File(output_file, 'w') as output_f:
                
                # Create output dataset with appropriate dtype
                if output_format == 'string':
                    output_dtype = 'S1'  # Single byte string
                    fill_value = b'N'
                else:
                    output_dtype = np.uint8
                    fill_value = 4  # N = 4
                
                output_dataset = output_f.create_dataset(
                    'genotype_data',
                    shape=(n_subjects, n_snps, 2),
                    dtype=output_dtype,
                    compression='gzip',
                    compression_opts=9,
                    chunks=True,
                    shuffle=True,
                    fletcher32=True,
                    fillvalue=fill_value
                )
                
                # Store metadata
                output_f.attrs['n_subjects'] = n_subjects
                output_f.attrs['n_snps'] = n_snps
                output_f.attrs['format'] = '3D genetic data (subjects, SNPs, alleles)'
                output_f.attrs['allele_order'] = 'First allele in [:,:,0], second in [:,:,1]'
                output_f.attrs['created_from'] = os.path.basename(input_file)
                output_f.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                if output_format == 'numeric':
                    allele_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
                    output_f.attrs['allele_encoding'] = json.dumps(allele_map)
                
                # Process in chunks
                total_chunks = (n_subjects + chunk_size - 1) // chunk_size
                print(f"Processing {total_chunks} chunks...")
                
                start_time = time.time()
                
                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, n_subjects)
                    
                    print(f"  Chunk {chunk_idx + 1}/{total_chunks}: subjects {start_idx:,} to {end_idx-1:,}")
                    
                    # Read chunk from input
                    chunk_2d = input_dataset[start_idx:end_idx, :]
                    
                    # Convert to 3D
                    chunk_3d = convert_genotype_chunk_to_3d(chunk_2d, output_format)
                    
                    # Write to output
                    if output_format == 'string':
                        output_dataset[start_idx:end_idx, :, :] = chunk_3d.astype('S1')
                    else:
                        output_dataset[start_idx:end_idx, :, :] = chunk_3d
                    
                    # Progress update
                    progress = (chunk_idx + 1) / total_chunks * 100
                    elapsed = time.time() - start_time
                    eta = elapsed * (total_chunks / (chunk_idx + 1)) - elapsed
                    
                    print(f"    ✓ Progress: {progress:.1f}% (ETA: {eta/60:.1f} min)")
        
        # Final statistics
        processing_time = time.time() - start_time
        output_size = os.path.getsize(output_file)
        input_size = os.path.getsize(input_file)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"  Processing time: {processing_time/60:.1f} minutes")
        print(f"  Input size: {input_size/(1024*1024):.1f} MB")
        print(f"  Output size: {output_size/(1024*1024):.1f} MB")
        print(f"  Compression ratio: {input_size/output_size:.1f}:1")
        print(f"  Final shape: ({n_subjects:,}, {n_snps:,}, 2)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_3d_output(filename, sample_size=1000):
    """Verify the 3D output file is correct"""
    print(f"\nVerifying output file: {filename}")
    
    try:
        with h5py.File(filename, 'r') as f:
            dataset = f['genotype_data']
            shape = dataset.shape
            
            print(f"  ✓ File readable")
            print(f"  ✓ Shape: {shape}")
            print(f"  ✓ Data type: {dataset.dtype}")
            
            # Sample some data
            n_subjects, n_snps = shape[:2]
            sample_subjects = min(sample_size, n_subjects)
            sample_snps = min(10, n_snps)
            
            sample_data = dataset[:sample_subjects, :sample_snps, :]
            
            if dataset.dtype.kind == 'S':  # String data
                # Convert to Unicode for display
                sample_display = sample_data.astype('U1')
            else:  # Numeric data
                sample_display = sample_data
            
            print(f"  ✓ Sample data (first {min(3, sample_subjects)} subjects, first {sample_snps} SNPs):")
            for i in range(min(3, sample_subjects)):
                print(f"    Subject {i}: {sample_display[i, :sample_snps, 0]} | {sample_display[i, :sample_snps, 1]}")
            
            # Check for homozygous vs heterozygous
            allele1 = sample_data[:, :, 0]
            allele2 = sample_data[:, :, 1]
            
            if dataset.dtype.kind == 'S':
                is_homozygous = allele1 == allele2
            else:
                is_homozygous = allele1 == allele2
            
            homozygous_rate = np.mean(is_homozygous)
            print(f"  ✓ Homozygous rate in sample: {homozygous_rate*100:.1f}%")
            
            # Check unique alleles
            if dataset.dtype.kind == 'S':
                unique_alleles = np.unique(sample_data.astype('U1'))
            else:
                unique_alleles = np.unique(sample_data)
            
            print(f"  ✓ Unique alleles in sample: {sorted(unique_alleles)}")
            
            print(f"  ✅ Verification passed!")
            return True
            
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False

def main():
    print("="*60)
    print("HDF5 TO 3D FORMAT CONVERTER")
    print("="*60)
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Data key: {args.data_key}")
    print(f"Output format: {args.output_format}")
    print(f"Chromosomes: {args.chromosomes}")
    
    # Parse chromosomes
    chromosomes = parse_chromosome_range(args.chromosomes)
    print(f"Will process chromosomes: {chromosomes}")
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # First, inspect available files
    print(f"\n{'='*60}")
    print("INSPECTING INPUT FILES")
    print(f"{'='*60}")
    
    available_files = {}
    for chrom in chromosomes:
        input_file = os.path.join(args.input_folder, f'chr{chrom}.h5')
        if os.path.exists(input_file):
            info = inspect_h5_file(input_file)
            if info:
                available_files[chrom] = {
                    'file': input_file,
                    'info': info
                }
            else:
                print(f"  ❌ chr{chrom}: Cannot read file")
        else:
            print(f"  ❌ chr{chrom}: File not found - {input_file}")
    
    if not available_files:
        print("❌ No valid input files found!")
        return False
    
    print(f"\nFound {len(available_files)} valid files to process")
    
    # Process each chromosome
    print(f"\n{'='*60}")
    print("CONVERTING TO 3D FORMAT")
    print(f"{'='*60}")
    
    results = {}
    total_start_time = time.time()
    
    for chrom in sorted(available_files.keys()):
        file_info = available_files[chrom]
        input_file = file_info['file']
        output_file = os.path.join(args.output_folder, f'chr{chrom}_3d.h5')
        
        # Use the correct data key for this file
        data_key = file_info['info']['data_key']
        
        start_time = time.time()
        success = convert_h5_to_3d(
            input_file, output_file, 
            args.chunk_size, data_key, 
            args.output_format
        )
        
        if success:
            # Verify the output
            verify_success = verify_3d_output(output_file)
            success = success and verify_success
        
        process_time = time.time() - start_time
        
        results[chrom] = {
            'success': success,
            'time': process_time,
            'input_file': input_file,
            'output_file': output_file if success else None
        }
    
    # Final summary
    total_time = time.time() - total_start_time
    successful = [c for c, r in results.items() if r['success']]
    failed = [c for c, r in results.items() if not r['success']]
    
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Successful conversions: {len(successful)}")
    print(f"Failed conversions: {len(failed)}")
    
    if successful:
        print(f"\n✅ Successfully converted:")
        total_output_size = 0
        for chrom in successful:
            result = results[chrom]
            output_size = os.path.getsize(result['output_file'])
            total_output_size += output_size
            print(f"  chr{chrom}: {result['time']/60:.1f} min, {output_size/(1024*1024):.1f} MB")
        
        print(f"\nTotal output size: {total_output_size/(1024*1024*1024):.1f} GB")
        print(f"Average per chromosome: {total_output_size/(len(successful)*1024*1024):.1f} MB")
    
    if failed:
        print(f"\n❌ Failed conversions:")
        for chrom in failed:
            print(f"  chr{chrom}")
    
    print(f"\n📁 3D files saved in: {args.output_folder}")
    print(f"🎉 Ready for analysis! Use h5py to load the 3D data.")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)