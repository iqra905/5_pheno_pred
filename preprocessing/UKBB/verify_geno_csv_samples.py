#!/usr/bin/env python3
"""
Genotype File Validator

This script validates the presence of genotype .npy files for samples listed in a CSV file.
It checks if each sample (identified by ID_1 and ID_2) has corresponding genotype files
for all specified chromosomes.

File naming convention: sample_{ID_1}_{ID_2}_chr{chr_num}.npy
Directory structure: chr{chr_num}/sample_{ID_1}_{ID_2}_chr{chr_num}.npy
"""

import pandas as pd
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime


class GenotypeFileValidator:
    """Validates the presence of genotype files for samples in a CSV."""
    
    def __init__(self, csv_path, genotype_base_dir, chromosomes=None):
        """
        Initialize the validator.
        
        Args:
            csv_path (str): Path to the CSV file containing sample information
            genotype_base_dir (str): Base directory containing chr1/, chr2/, etc. folders
            chromosomes (list): List of chromosomes to check (default: 1-22)
        """
        self.csv_path = Path(csv_path)
        self.genotype_base_dir = Path(genotype_base_dir)
        self.chromosomes = chromosomes if chromosomes else list(range(1, 23))
        
        # Storage for results
        self.samples_df = None
        self.missing_files = defaultdict(list)
        self.found_files = defaultdict(list)
        self.sample_status = {}
        
    def read_csv(self):
        """Read and validate the CSV file."""
        print(f"\n{'='*80}")
        print(f"READING CSV FILE")
        print(f"{'='*80}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Read CSV (tab-separated based on your column description)
        try:
            self.samples_df = pd.read_csv(self.csv_path, sep='\t')
            print(f"✓ Successfully read CSV file: {self.csv_path}")
        except Exception as e:
            try:
                self.samples_df = pd.read_csv(self.csv_path)
                print(f"✓ Successfully read CSV file (comma-separated): {self.csv_path}")
            except Exception as e2:
                raise Exception(f"Failed to read CSV file: {e2}")
        
        # Validate required columns
        if 'ID_1' not in self.samples_df.columns or 'ID_2' not in self.samples_df.columns:
            raise ValueError("CSV must contain 'ID_1' and 'ID_2' columns")
        
        print(f"  Total rows in CSV: {len(self.samples_df)}")
        print(f"  Columns found: {', '.join(self.samples_df.columns.tolist())}")
        
        # Check for duplicates
        duplicates = self.samples_df.duplicated(subset=['ID_1', 'ID_2'], keep=False)
        if duplicates.any():
            n_duplicates = duplicates.sum()
            print(f"  ⚠ Warning: Found {n_duplicates} duplicate ID_1/ID_2 combinations")
            
        # Remove rows with missing ID_1 or ID_2
        missing_ids = self.samples_df[['ID_1', 'ID_2']].isna().any(axis=1)
        if missing_ids.any():
            n_missing = missing_ids.sum()
            print(f"  ⚠ Warning: Removing {n_missing} rows with missing ID_1 or ID_2")
            self.samples_df = self.samples_df[~missing_ids]
        
        print(f"  Unique samples to validate: {len(self.samples_df)}")
        
    def validate_directory_structure(self):
        """Check if the base directory and chromosome folders exist."""
        print(f"\n{'='*80}")
        print(f"VALIDATING DIRECTORY STRUCTURE")
        print(f"{'='*80}")
        
        if not self.genotype_base_dir.exists():
            raise FileNotFoundError(f"Genotype base directory not found: {self.genotype_base_dir}")
        
        print(f"✓ Base directory exists: {self.genotype_base_dir}")
        
        # Check chromosome directories
        missing_chr_dirs = []
        existing_chr_dirs = []
        
        for chr_num in self.chromosomes:
            chr_dir = self.genotype_base_dir / f"chr{chr_num}"
            if chr_dir.exists():
                existing_chr_dirs.append(chr_num)
                print(f"  ✓ chr{chr_num}/ exists")
            else:
                missing_chr_dirs.append(chr_num)
                print(f"  ✗ chr{chr_num}/ NOT FOUND")
        
        if missing_chr_dirs:
            print(f"\n⚠ Warning: Missing chromosome directories: {missing_chr_dirs}")
            print(f"  Validation will fail for these chromosomes")
        
        return existing_chr_dirs
    
    def construct_filename(self, id_1, id_2, chr_num):
        """
        Construct the expected filename for a sample.
        
        Args:
            id_1: First ID
            id_2: Second ID
            chr_num: Chromosome number
            
        Returns:
            Path object for the expected file
        """
        filename = f"sample_{id_1}_{id_2}_chr{chr_num}.npy"
        chr_dir = self.genotype_base_dir / f"chr{chr_num}"
        return chr_dir / filename
    
    def check_files(self):
        """Check for the existence of genotype files for all samples."""
        print(f"\n{'='*80}")
        print(f"CHECKING GENOTYPE FILES")
        print(f"{'='*80}")
        print(f"Checking {len(self.samples_df)} samples across {len(self.chromosomes)} chromosomes...")
        print(f"Total files to check: {len(self.samples_df) * len(self.chromosomes)}")
        
        total_files_checked = 0
        total_files_found = 0
        total_files_missing = 0
        
        for idx, row in self.samples_df.iterrows():
            id_1 = row['ID_1']
            id_2 = row['ID_2']
            sample_key = f"{id_1}_{id_2}"
            
            files_found = 0
            files_missing = 0
            missing_chromosomes = []
            
            for chr_num in self.chromosomes:
                file_path = self.construct_filename(id_1, id_2, chr_num)
                total_files_checked += 1
                
                if file_path.exists():
                    self.found_files[sample_key].append(chr_num)
                    files_found += 1
                    total_files_found += 1
                else:
                    self.missing_files[sample_key].append(chr_num)
                    missing_chromosomes.append(chr_num)
                    files_missing += 1
                    total_files_missing += 1
            
            # Store sample status
            self.sample_status[sample_key] = {
                'ID_1': id_1,
                'ID_2': id_2,
                'total_expected': len(self.chromosomes),
                'files_found': files_found,
                'files_missing': files_missing,
                'missing_chromosomes': missing_chromosomes,
                'complete': files_missing == 0
            }
            
            # Print progress every 10 samples
            if (idx + 1) % 10 == 0 or (idx + 1) == len(self.samples_df):
                print(f"  Progress: {idx + 1}/{len(self.samples_df)} samples checked", end='\r')
        
        print()  # New line after progress
        print(f"\n✓ File checking complete!")
        print(f"  Total files checked: {total_files_checked}")
        print(f"  Files found: {total_files_found}")
        print(f"  Files missing: {total_files_missing}")
        
    def generate_report(self, output_file=None):
        """Generate a detailed validation report."""
        print(f"\n{'='*80}")
        print(f"VALIDATION REPORT")
        print(f"{'='*80}")
        
        # Summary statistics
        total_samples = len(self.sample_status)
        complete_samples = sum(1 for s in self.sample_status.values() if s['complete'])
        incomplete_samples = total_samples - complete_samples
        
        print(f"\nSUMMARY:")
        print(f"  Total samples in CSV: {total_samples}")
        print(f"  Samples with all files: {complete_samples} ({complete_samples/total_samples*100:.1f}%)")
        print(f"  Samples with missing files: {incomplete_samples} ({incomplete_samples/total_samples*100:.1f}%)")
        
        # Detailed breakdown
        if incomplete_samples > 0:
            print(f"\n{'='*80}")
            print(f"SAMPLES WITH MISSING FILES ({incomplete_samples} samples):")
            print(f"{'='*80}")
            
            for sample_key, status in sorted(self.sample_status.items()):
                if not status['complete']:
                    missing_chr = status['missing_chromosomes']
                    print(f"\n  Sample: {sample_key}")
                    print(f"    ID_1: {status['ID_1']}")
                    print(f"    ID_2: {status['ID_2']}")
                    print(f"    Files found: {status['files_found']}/{status['total_expected']}")
                    print(f"    Missing chromosomes: {missing_chr}")
        
        # Chromosome-wise missing file count
        chr_missing_count = defaultdict(int)
        for status in self.sample_status.values():
            for chr_num in status['missing_chromosomes']:
                chr_missing_count[chr_num] += 1
        
        if chr_missing_count:
            print(f"\n{'='*80}")
            print(f"MISSING FILES BY CHROMOSOME:")
            print(f"{'='*80}")
            for chr_num in sorted(chr_missing_count.keys()):
                count = chr_missing_count[chr_num]
                print(f"  chr{chr_num}: {count} samples missing ({count/total_samples*100:.1f}%)")
        
        # Save detailed report to file if requested
        if output_file:
            self.save_detailed_report(output_file)
    
    def save_detailed_report(self, output_file):
        """Save detailed report to a CSV file."""
        print(f"\n{'='*80}")
        print(f"SAVING DETAILED REPORT")
        print(f"{'='*80}")
        
        report_data = []
        for sample_key, status in self.sample_status.items():
            report_data.append({
                'ID_1': status['ID_1'],
                'ID_2': status['ID_2'],
                'sample_key': sample_key,
                'files_expected': status['total_expected'],
                'files_found': status['files_found'],
                'files_missing': status['files_missing'],
                'complete': status['complete'],
                'missing_chromosomes': ','.join(map(str, status['missing_chromosomes'])) if status['missing_chromosomes'] else ''
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_file, index=False)
        print(f"✓ Detailed report saved to: {output_file}")
        print(f"  Total records: {len(report_df)}")
    
    def run(self, save_report=True):
        """Run the complete validation process."""
        print(f"\n{'#'*80}")
        print(f"GENOTYPE FILE VALIDATION")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        try:
            # Step 1: Read CSV
            self.read_csv()
            
            # Step 2: Validate directory structure
            self.validate_directory_structure()
            
            # Step 3: Check files
            self.check_files()
            
            # Step 4: Generate report
            output_file = None
            if save_report:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"genotype_validation_report_{timestamp}.csv"
            
            self.generate_report(output_file)
            
            print(f"\n{'#'*80}")
            print(f"VALIDATION COMPLETE")
            print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'#'*80}\n")
            
            return self.sample_status
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            raise


def main():
    """Main function to run the validator from command line."""
    parser = argparse.ArgumentParser(
        description='Validate genotype .npy files for samples in a CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with chromosomes 1-22
  python check_genotype_files.py samples.csv /path/to/genotypes
  
  # Check specific chromosomes
  python check_genotype_files.py samples.csv /path/to/genotypes --chromosomes 1 2 3 X Y
  
  # Skip saving detailed report
  python check_genotype_files.py samples.csv /path/to/genotypes --no-report
        """
    )
    
    parser.add_argument('-csv_file', default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/ukb_cancers_t2d_ukb676869_13102025.tsv', help='Path to the CSV file containing sample information')
    parser.add_argument('-genotype_dir', default ='/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8', help='Base directory containing chr1/, chr2/, etc. folders')
    parser.add_argument('-chromosomes', nargs='+', 
                       help='Chromosomes to check (default: 1-22). Can include X, Y, MT')
    parser.add_argument('-no-report', action='store_true',
                       help='Do not save detailed report to CSV file')
    
    args = parser.parse_args()
    
    # Parse chromosomes
    chromosomes = None
    if args.chromosomes:
        chromosomes = []
        for chr_val in args.chromosomes:
            # Handle numeric and non-numeric chromosomes
            if chr_val.isdigit():
                chromosomes.append(int(chr_val))
            else:
                chromosomes.append(chr_val)  # For X, Y, MT, etc.
    
    # Create validator and run
    validator = GenotypeFileValidator(
        csv_path=args.csv_file,
        genotype_base_dir=args.genotype_dir,
        chromosomes=chromosomes
    )
    
    validator.run(save_report=not args.no_report)


if __name__ == "__main__":
    main()