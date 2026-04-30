#!/usr/bin/env python3
"""
Parallel converter for HDF5 genetic data to 3D format
Processes multiple chromosomes simultaneously using multiprocessing
"""

import h5py
import numpy as np
import os
import time
import argparse
from pathlib import Path
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import logging
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    try:
        with h5py.File(filename, 'r') as f:
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
                return None
            
            dataset = f[data_key]
            return {
                'data_key': data_key,
                'shape': dataset.shape,
                'dtype': dataset.dtype,
                'n_subjects': dataset.shape[0],
                'n_snps': dataset.shape[1] if len(dataset.shape) > 1 else 0,
                'file_size_mb': os.path.getsize(filename) / (1024*1024)
            }
    
    except Exception as e:
        logger.error(f"Error inspecting {filename}: {e}")
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

def convert_single_chromosome(args_tuple):
    """Convert a single chromosome file (designed for multiprocessing)"""
    
    (chrom, input_file, output_file, chunk_size, data_key, output_format, process_id) = args_tuple
    
    # Set up process-specific logging
    process_logger = logging.getLogger(f'chr{chrom}_worker')
    
    start_time = time.time()
    
    try:
        process_logger.info(f"[Process {process_id}] Starting chr{chrom} conversion")
        process_logger.info(f"[Process {process_id}] Input: {os.path.basename(input_file)}")
        process_logger.info(f"[Process {process_id}] Output: {os.path.basename(output_file)}")
        
        if not os.path.exists(input_file):
            return {
                'chromosome': chrom,
                'success': False,
                'error': f'Input file not found: {input_file}',
                'process_id': process_id
            }
        
        # Get memory info for this process
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024*1024)  # MB
        
        with h5py.File(input_file, 'r') as input_f:
            
            # Find data key if not specified
            if data_key not in input_f:
                for alt_key in ['genotype_data', 'data', 'genotypes', 'combined_genotypes']:
                    if alt_key in input_f:
                        data_key = alt_key
                        break
                else:
                    for key in input_f.keys():
                        if not key.startswith('__'):
                            data_key = key
                            break
                    else:
                        return {
                            'chromosome': chrom,
                            'success': False,
                            'error': 'No suitable data key found',
                            'process_id': process_id
                        }
            
            input_dataset = input_f[data_key]
            n_subjects, n_snps = input_dataset.shape
            
            process_logger.info(f"[Process {process_id}] chr{chrom}: {n_subjects:,} subjects × {n_snps:,} SNPs")
            
            # Create output file
            with h5py.File(output_file, 'w') as output_f:
                
                # Set up output dataset
                if output_format == 'string':
                    output_dtype = 'S1'
                    fill_value = b'N'
                else:
                    output_dtype = np.uint8
                    fill_value = 4
                
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
                output_f.attrs['chromosome'] = chrom
                output_f.attrs['format'] = '3D genetic data (subjects, SNPs, alleles)'
                output_f.attrs['created_from'] = os.path.basename(input_file)
                output_f.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                output_f.attrs['process_id'] = process_id
                
                if output_format == 'numeric':
                    allele_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
                    output_f.attrs['allele_encoding'] = json.dumps(allele_map)
                
                # Process in chunks
                total_chunks = (n_subjects + chunk_size - 1) // chunk_size
                process_logger.info(f"[Process {process_id}] chr{chrom}: Processing {total_chunks} chunks")
                
                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, n_subjects)
                    
                    # Read chunk
                    chunk_2d = input_dataset[start_idx:end_idx, :]
                    
                    # Convert to 3D
                    chunk_3d = convert_genotype_chunk_to_3d(chunk_2d, output_format)
                    
                    # Write to output
                    if output_format == 'string':
                        output_dataset[start_idx:end_idx, :, :] = chunk_3d.astype('S1')
                    else:
                        output_dataset[start_idx:end_idx, :, :] = chunk_3d
                    
                    # Progress update every 10 chunks
                    if (chunk_idx + 1) % 10 == 0 or chunk_idx == total_chunks - 1:
                        progress = (chunk_idx + 1) / total_chunks * 100
                        current_memory = process.memory_info().rss / (1024*1024)
                        process_logger.info(f"[Process {process_id}] chr{chrom}: {progress:.1f}% complete "
                                          f"(memory: {current_memory:.0f}MB)")
        
        # Final statistics
        processing_time = time.time() - start_time
        output_size = os.path.getsize(output_file)
        input_size = os.path.getsize(input_file)
        final_memory = process.memory_info().rss / (1024*1024)
        
        process_logger.info(f"[Process {process_id}] chr{chrom}: ✅ Completed in {processing_time/60:.1f} minutes")
        process_logger.info(f"[Process {process_id}] chr{chrom}: {output_size/(1024*1024):.1f} MB output "
                          f"({input_size/output_size:.1f}:1 compression)")
        
        return {
            'chromosome': chrom,
            'success': True,
            'processing_time': processing_time,
            'input_size_mb': input_size / (1024*1024),
            'output_size_mb': output_size / (1024*1024),
            'compression_ratio': input_size / output_size,
            'n_subjects': n_subjects,
            'n_snps': n_snps,
            'input_file': input_file,
            'output_file': output_file,
            'process_id': process_id,
            'peak_memory_mb': final_memory
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        process_logger.error(f"[Process {process_id}] chr{chrom}: ❌ Failed after {processing_time/60:.1f} minutes: {e}")
        
        return {
            'chromosome': chrom,
            'success': False,
            'error': str(e),
            'processing_time': processing_time,
            'process_id': process_id
        }

def main():
    parser = argparse.ArgumentParser(description="Parallel HDF5 to 3D format converter")
    parser.add_argument('-input_folder', type=str, required=True, help='Folder with chr*.h5 files')
    parser.add_argument('-output_folder', type=str, required=True, help='Output folder for chr*_3d.h5 files')
    parser.add_argument('-chunk_size', type=int, default=100, help='Number of subjects to process at once')
    parser.add_argument('-chromosomes', type=str, default='1-22', help='Chromosomes to process (e.g., 1-22, 1,2,3)')
    parser.add_argument('-data_key', type=str, default='combined_genotypes', help='Key name in HDF5 file for genetic data')
    parser.add_argument('-output_format', type=str, choices=['string', 'numeric'], default='string', 
                       help='Output format: string (A,T,C,G) or numeric (0,1,2,3)')
    parser.add_argument('-processes', type=int, default=5, help='Number of parallel processes (0=auto)')
    parser.add_argument('-memory_limit_gb', type=float, default=0, help='Memory limit in GB (0=auto)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PARALLEL HDF5 TO 3D FORMAT CONVERTER")
    print("="*80)
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
    
    # Find and inspect input files
    print(f"\n{'='*80}")
    print("INSPECTING INPUT FILES")
    print(f"{'='*80}")
    
    available_files = {}
    total_size_mb = 0
    
    for chrom in chromosomes:
        input_file = os.path.join(args.input_folder, f'chr{chrom}.h5')
        if os.path.exists(input_file):
            info = inspect_h5_file(input_file)
            if info:
                available_files[chrom] = {
                    'file': input_file,
                    'info': info
                }
                total_size_mb += info['file_size_mb']
                print(f"  ✓ chr{chrom}: {info['n_subjects']:,} subjects × {info['n_snps']:,} SNPs "
                      f"({info['file_size_mb']:.1f} MB)")
            else:
                print(f"  ❌ chr{chrom}: Cannot read file")
        else:
            print(f"  ❌ chr{chrom}: File not found")
    
    if not available_files:
        print("❌ No valid input files found!")
        return False
    
    print(f"\nFound {len(available_files)} valid files ({total_size_mb:.1f} MB total)")
    
    n_processes = min(args.processes, len(available_files))  
    
    print(f"\nUsing {n_processes} parallel processes")
    
    # Prepare arguments for parallel processing
    conversion_args = []
    for process_id, chrom in enumerate(sorted(available_files.keys())):
        file_info = available_files[chrom]
        input_file = file_info['file']
        output_file = os.path.join(args.output_folder, f'chr{chrom}_3d.h5')
        data_key = file_info['info']['data_key']
        
        conversion_args.append((
            chrom, input_file, output_file, args.chunk_size, 
            data_key, args.output_format, process_id
        ))
    
    # Run parallel conversion
    print(f"\n{'='*80}")
    print("PARALLEL CONVERSION")
    print(f"{'='*80}")
    
    start_time = time.time()
    results = {}
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all jobs
        future_to_chrom = {
            executor.submit(convert_single_chromosome, args): args[0] 
            for args in conversion_args
        }
        
        # Monitor progress
        completed = 0
        total_jobs = len(future_to_chrom)
        
        for future in as_completed(future_to_chrom):
            chrom = future_to_chrom[future]
            try:
                result = future.result()
                results[chrom] = result
                completed += 1
                
                if result['success']:
                    print(f"✅ chr{result['chromosome']} completed "
                          f"({completed}/{total_jobs}) - {result['processing_time']/60:.1f} min, "
                          f"{result['output_size_mb']:.1f} MB")
                else:
                    print(f"❌ chr{result['chromosome']} failed "
                          f"({completed}/{total_jobs}) - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ chr{chrom} failed with exception: {e}")
                results[chrom] = {
                    'chromosome': chrom,
                    'success': False,
                    'error': str(e)
                }
                completed += 1
    
    # Final summary
    total_time = time.time() - start_time
    successful = [c for c, r in results.items() if r['success']]
    failed = [c for c, r in results.items() if not r['success']]
    
    print(f"\n{'='*80}")
    print("PARALLEL CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Wall clock time per file: {total_time/len(available_files)/60:.1f} minutes")
    print(f"Speedup factor: {sum(r['processing_time'] for r in results.values() if r['success'])/total_time:.1f}x")
    print(f"Successful conversions: {len(successful)}/{len(available_files)}")
    print(f"Failed conversions: {len(failed)}")
    
    if successful:
        print(f"\n✅ Successfully converted:")
        total_input_size = 0
        total_output_size = 0
        total_processing_time = 0
        
        print(f"{'Chrom':<6} {'Time':<8} {'Input':<10} {'Output':<10} {'Ratio':<8} {'Subjects':<10} {'SNPs':<8}")
        print("-" * 70)
        
        for chrom in sorted(successful):
            result = results[chrom]
            total_input_size += result['input_size_mb']
            total_output_size += result['output_size_mb']
            total_processing_time += result['processing_time']
            
            print(f"chr{chrom:<3} {result['processing_time']/60:<7.1f}m "
                  f"{result['input_size_mb']:<9.1f}M {result['output_size_mb']:<9.1f}M "
                  f"{result['compression_ratio']:<7.1f}x {result['n_subjects']:<9,} {result['n_snps']:<7,}")
        
        print("-" * 70)
        print(f"{'Total':<6} {total_processing_time/60:<7.1f}m "
              f"{total_input_size:<9.1f}M {total_output_size:<9.1f}M "
              f"{total_input_size/total_output_size:<7.1f}x")
        
        print(f"\nStorage savings: {total_input_size - total_output_size:.1f} MB "
              f"({(1-total_output_size/total_input_size)*100:.1f}% reduction)")
    
    if failed:
        print(f"\n❌ Failed conversions:")
        for chrom in failed:
            result = results[chrom]
            print(f"  chr{chrom}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 3D files saved in: {args.output_folder}")
    print(f"🚀 Parallel processing completed!")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)