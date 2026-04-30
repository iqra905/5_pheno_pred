#!/usr/bin/env python3
"""
Script to convert existing float16 genotype files to uint8 format.
Provides 50% storage savings while maintaining acceptable precision.
"""

import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import shutil


def validate_conversion(original_float16, converted_uint8, max_val=255):
    """
    Validate that conversion maintains acceptable precision.
    
    Parameters:
    -----------
    original_float16 : np.ndarray
        Original float16 data
    converted_uint8 : np.ndarray
        Converted uint8 data
    max_val : int
        Scaling factor (255 for uint8, 65535 for uint16)
    
    Returns:
    --------
    dict : Validation results
    """
    # Convert uint8 back to float for comparison
    recovered = converted_uint8.astype(np.float32) / max_val
    
    # Calculate errors
    errors = np.abs(original_float16 - recovered)
    
    return {
        'max_error': np.max(errors),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'passed': np.max(errors) < 0.01  # Less than 1% error
    }


def convert_file(input_file, output_file, target_format='uint8', validate=True, verbose=False):
    """
    Convert a single file from float16 to uint format.
    
    Parameters:
    -----------
    input_file : Path
        Input .npy file (float16)
    output_file : Path
        Output .npy file (uint8 or uint16)
    target_format : str
        Target format: 'uint8' or 'uint16'
    validate : bool
        Whether to validate conversion
    verbose : bool
        Print detailed information
    
    Returns:
    --------
    dict : Conversion results
    """
    try:
        start_time = time.time()
        
        # Load float16 data
        data_float16 = np.load(input_file)
        
        # Check if it's actually float16
        if data_float16.dtype != np.float16:
            return {
                'status': 'skipped',
                'message': f'Not float16 (found {data_float16.dtype})',
                'input_size_mb': 0,
                'output_size_mb': 0,
                'time_seconds': 0
            }
        
        original_size = data_float16.nbytes
        
        # Convert based on target format
        if target_format == 'uint8':
            max_val = 255
            dtype = np.uint8
        elif target_format == 'uint16':
            max_val = 65535
            dtype = np.uint16
        else:
            raise ValueError(f"Unsupported format: {target_format}")
        
        # Scale and convert
        data_uint = (data_float16 * max_val).astype(dtype)
        
        # Validate if requested
        validation_results = None
        if validate:
            validation_results = validate_conversion(data_float16, data_uint, max_val)
            if not validation_results['passed']:
                return {
                    'status': 'failed',
                    'message': f"Validation failed: max error {validation_results['max_error']:.6f}",
                    'input_size_mb': original_size / (1024**2),
                    'output_size_mb': 0,
                    'time_seconds': time.time() - start_time
                }
        
        # Save converted data
        np.save(output_file, data_uint)
        
        converted_size = data_uint.nbytes
        elapsed_time = time.time() - start_time
        
        result = {
            'status': 'success',
            'input_size_mb': original_size / (1024**2),
            'output_size_mb': converted_size / (1024**2),
            'savings_mb': (original_size - converted_size) / (1024**2),
            'savings_pct': ((original_size - converted_size) / original_size) * 100,
            'time_seconds': elapsed_time,
            'n_variants': data_float16.shape[0] if len(data_float16.shape) > 0 else 1
        }
        
        if validation_results:
            result['validation'] = validation_results
        
        if verbose:
            print(f"  ✓ Converted: {input_file.name}")
            print(f"    Size: {result['input_size_mb']:.2f} MB → {result['output_size_mb']:.2f} MB")
            print(f"    Saved: {result['savings_mb']:.2f} MB ({result['savings_pct']:.1f}%)")
            if validation_results:
                print(f"    Max error: {validation_results['max_error']:.6f}")
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'input_size_mb': 0,
            'output_size_mb': 0,
            'time_seconds': 0
        }


def find_npy_files(input_dir, pattern='*.npy'):
    """Find all .npy files in directory and subdirectories."""
    input_path = Path(input_dir)
    return sorted(input_path.rglob(pattern))


def convert_directory(input_dir, output_dir, target_format='uint8', validate=True, 
                      dry_run=False, pattern='*.npy', backup=False):
    """
    Convert all float16 .npy files in a directory to uint format.
    
    Parameters:
    -----------
    input_dir : str
        Input directory containing float16 .npy files
    output_dir : str
        Output directory for converted files
    target_format : str
        Target format: 'uint8' or 'uint16'
    validate : bool
        Whether to validate each conversion
    dry_run : bool
        If True, only show what would be converted
    pattern : str
        File pattern to match (default: '*.npy')
    backup : bool
        If True, keep a backup of original files
    
    Returns:
    --------
    dict : Summary statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all .npy files
    print(f"Scanning for .npy files in {input_dir}...")
    npy_files = find_npy_files(input_dir, pattern)
    
    if not npy_files:
        print("No .npy files found!")
        return None
    
    print(f"Found {len(npy_files)} .npy files")
    print()
    
    if dry_run:
        print("DRY RUN MODE - No files will be converted")
        print("-" * 80)
        for f in npy_files[:10]:  # Show first 10
            print(f"  Would convert: {f.relative_to(input_path)}")
        if len(npy_files) > 10:
            print(f"  ... and {len(npy_files) - 10} more files")
        print()
        return None
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert files
    results = []
    total_input_size = 0
    total_output_size = 0
    
    print(f"Converting files to {target_format}...")
    print("-" * 80)
    
    for input_file in tqdm(npy_files, desc="Converting"):
        # Determine output file path (preserve directory structure)
        rel_path = input_file.relative_to(input_path)
        output_file = output_path / rel_path
        
        # Create output subdirectory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup original if requested
        if backup and output_file.exists():
            backup_file = output_file.with_suffix('.npy.bak')
            shutil.copy2(output_file, backup_file)
        
        # Convert file
        result = convert_file(input_file, output_file, target_format, validate, verbose=False)
        result['file'] = str(rel_path)
        results.append(result)
        
        if result['status'] == 'success':
            total_input_size += result['input_size_mb']
            total_output_size += result['output_size_mb']
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    print()
    print("=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print()
    print(f"Total files processed: {len(npy_files)}")
    print(f"  ✓ Successful:        {len(successful)}")
    print(f"  ✗ Failed:            {len(failed)}")
    print(f"  ⊘ Skipped:           {len(skipped)}")
    print()
    
    if successful:
        total_savings = total_input_size - total_output_size
        savings_pct = (total_savings / total_input_size) * 100 if total_input_size > 0 else 0
        
        print("Storage Statistics:")
        print("-" * 80)
        print(f"Original size:  {total_input_size:,.2f} MB")
        print(f"Converted size: {total_output_size:,.2f} MB")
        print(f"Total savings:  {total_savings:,.2f} MB ({savings_pct:.1f}%)")
        print()
        
        # Calculate average conversion metrics
        avg_time = np.mean([r['time_seconds'] for r in successful])
        total_time = sum([r['time_seconds'] for r in successful])
        
        print("Performance:")
        print("-" * 80)
        print(f"Average conversion time: {avg_time*1000:.2f} ms per file")
        print(f"Total conversion time:   {total_time:.2f} seconds")
        print()
        
        # Validation statistics if available
        if validate and successful[0].get('validation'):
            max_errors = [r['validation']['max_error'] for r in successful if 'validation' in r]
            mean_errors = [r['validation']['mean_error'] for r in successful if 'validation' in r]
            
            print("Validation Statistics:")
            print("-" * 80)
            print(f"Maximum error across all files: {max(max_errors):.6f}")
            print(f"Average max error:              {np.mean(max_errors):.6f}")
            print(f"Average mean error:             {np.mean(mean_errors):.6f}")
            print()
    
    if failed:
        print("Failed Conversions:")
        print("-" * 80)
        for r in failed[:5]:  # Show first 5 failures
            print(f"  ✗ {r['file']}: {r['message']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more failures")
        print()
    
    if skipped:
        print(f"Skipped {len(skipped)} files (not float16 format)")
        print()
    
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    return {
        'total_files': len(npy_files),
        'successful': len(successful),
        'failed': len(failed),
        'skipped': len(skipped),
        'total_input_size_mb': total_input_size,
        'total_output_size_mb': total_output_size,
        'total_savings_mb': total_input_size - total_output_size,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert float16 genotype files to uint8/uint16 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all files in directory to uint8 (50% savings)
  %(prog)s -input /data/genotypes_float16 -output /data/genotypes_uint8
  
  # Convert to uint16 (same size, better precision)
  %(prog)s -input /data/genotypes_float16 -output /data/genotypes_uint16 -format uint16
  
  # Convert specific chromosome
  %(prog)s -input /data/float16 -output /data/uint8 -pattern "chr1/*.npy"
  
  # Dry run (see what would be converted)
  %(prog)s -input /data/float16 -output /data/uint8 -dryrun
  
  # Convert without validation (faster)
  %(prog)s -input /data/float16 -output /data/uint8 -novalidate
  
  # Keep backup of originals
  %(prog)s -input /data/float16 -output /data/uint8 -backup

Output:
  Preserves directory structure from input to output.
  Example: input/chr1/sample_123.npy → output/chr1/sample_123.npy
        """
    )
    
    parser.add_argument('-input', type=str, required=True,
                       help='Input directory containing float16 .npy files')
    parser.add_argument('-output', type=str, required=True,
                       help='Output directory for converted files')
    parser.add_argument('-format', type=str, default='uint8', choices=['uint8', 'uint16'],
                       help='Target format (default: uint8)')
    parser.add_argument('-pattern', type=str, default='*.npy',
                       help='File pattern to match (default: *.npy)')
    parser.add_argument('-novalidate', action='store_true',
                       help='Skip validation (faster but less safe)')
    parser.add_argument('-dryrun', action='store_true',
                       help='Show what would be converted without converting')
    parser.add_argument('-backup', action='store_true',
                       help='Keep backup of original files if overwriting')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENOTYPE FORMAT CONVERTER: float16 → uint8/uint16")
    print("=" * 80)
    print()
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Target format:    {args.format}")
    print(f"File pattern:     {args.pattern}")
    print(f"Validation:       {'Disabled' if args.novalidate else 'Enabled'}")
    print(f"Dry run:          {'Yes' if args.dryrun else 'No'}")
    print(f"Backup:           {'Yes' if args.backup else 'No'}")
    print()
    
    # Convert directory
    start_time = time.time()
    
    summary = convert_directory(
        args.input,
        args.output,
        target_format=args.format,
        validate=not args.novalidate,
        dry_run=args.dryrun,
        pattern=args.pattern,
        backup=args.backup
    )
    
    if summary and not args.dryrun:
        elapsed = time.time() - start_time
        print()
        print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print()


if __name__ == "__main__":
    main()