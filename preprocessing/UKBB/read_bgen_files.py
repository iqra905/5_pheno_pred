#!/usr/bin/env python3
"""
Script to read and process UK Biobank BGEN files across chromosomes 1-22
File pattern: ukb_imp_chr*_maf0.05.bgen where * ranges from 1 to 22
"""

import os
from pathlib import Path
from bgen_reader import open_bgen
import pandas as pd
import numpy as np


class BgenReader:
    """Class to handle reading multiple BGEN files across chromosomes"""
    
    def __init__(self, base_path=".", file_pattern="ukb_imp_chr{}_maf0.05.bgen"):
        """
        Initialize the BGEN reader
        
        Parameters:
        ------
        base_path : str
            Directory containing the BGEN files
        file_pattern : str
            Pattern for BGEN files with {} placeholder for chromosome number
        """
        self.base_path = Path(base_path)
        self.file_pattern = file_pattern
        self.chromosomes = range(1, 23)  # Chromosomes 1-22
        
    def get_bgen_file(self, chromosome):
        """Get the full path for a specific chromosome's BGEN file"""
        filename = self.file_pattern.format(chromosome)
        return self.base_path / filename
    
    def check_files_exist(self):
        """Check which BGEN files exist"""
        existing = []
        missing = []
        
        for chrom in self.chromosomes:
            filepath = self.get_bgen_file(chrom)
            if filepath.exists():
                existing.append(chrom)
                print(f"✓ Found: {filepath.name}")
            else:
                missing.append(chrom)
                print(f"✗ Missing: {filepath.name}")
        
        return existing, missing
    
    def get_file_info(self, chromosome):
        """Get basic information about a BGEN file"""
        filepath = self.get_bgen_file(chromosome)
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        with open_bgen(str(filepath), verbose=False) as bgen:

            print(f"File Summary:")
            print(f"  Chromosome: {chromosome}")
            print(f"  Total Variants: {bgen.nvariants:,}")
            print(f"  Total Samples: {bgen.nsamples:,}")
            print()

            info = {
                'chromosome': chromosome,
                'n_variants': bgen.nvariants,
                'n_samples': bgen.nsamples,
                'file_path': str(filepath)
            }
            
            # # Get sample IDs if available
            # try:
            #     info['sample_ids'] = bgen.samples[:2] if hasattr(bgen, 'samples') else None # First 2 samples
            # except:
            #     info['sample_ids'] = None
                
        return info
    
    def get_all_files_info(self):
        """Get information about all chromosome files"""
        info_list = []
        
        for chrom in self.chromosomes:
            info = self.get_file_info(chrom)
            if info:
                info_list.append(info)
        
        return pd.DataFrame(info_list)
    
    def read_variants_by_rsid(self, rsid_list, chromosome=None):
        """
        Read specific variants by their rsIDs
        Uses direct attribute access for FAST searching!
        """
        results = {}
        chromosomes = [chromosome] if chromosome else self.chromosomes
        
        for chrom in chromosomes:
            filepath = self.get_bgen_file(chrom)
            if not filepath.exists():
                continue
            
            with open_bgen(str(filepath), verbose=False) as bgen:
                # Get all rsIDs at once (FAST!)
                all_rsids = bgen.rsids
                
                # Find matching indices
                for target_rsid in rsid_list:
                    indices = np.where(all_rsids == target_rsid)[0]
                    
                    if len(indices) > 0:
                        idx = indices[0]
                        
                        # Get variant info from direct attributes
                        rsid = all_rsids[idx]
                        pos = bgen.positions[idx]
                        chrom_val = bgen.chromosomes[idx] if hasattr(bgen, 'chromosomes') else chrom
                        alleles = bgen.allele_ids[idx] if hasattr(bgen, 'allele_ids') else ['N/A', 'N/A']
                        
                        # Read genotype probabilities
                        geno_probs = bgen.read(idx)

                        # Handle (n_samples, 1, 3) format - squeeze out middle dimension
                        if len(geno_probs.shape) == 3 and geno_probs.shape[1] == 1:
                            geno_probs = geno_probs.squeeze(axis=1)
                        
                        results[rsid] = {
                            'chromosome': chrom_val,
                            'position': int(pos),
                            'rsid': rsid,
                            'allele_ref': alleles[0] if len(alleles) > 0 else 'N/A',
                            'allele_alt': alleles[1] if len(alleles) > 1 else 'N/A',
                            'genotype_probabilities': geno_probs,
                            'sample_ids': bgen.samples if hasattr(bgen, 'samples') else None
                        }
                        
                        print(f"  Found {rsid} on chr{chrom_val}:{pos:,}")
        
        return results
    
    def read_region(self, chromosome, start_pos, end_pos, max_variants=None):
        """
        Read variants in a specific genomic region
        Uses direct attribute access for FAST filtering!
        """
        filepath = self.get_bgen_file(chromosome)
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        results = {
            'variants': [],
            'genotype_probabilities': []
        }
        
        with open_bgen(str(filepath), verbose=False) as bgen:
            # Get all positions at once
            positions = bgen.positions
            
            # Find variants in range (FAST!)
            in_range = (positions >= start_pos) & (positions <= end_pos)
            indices = np.where(in_range)[0]
            
            if max_variants:
                indices = indices[:max_variants]
            
            print(f"Found {len(indices)} variants in region")
            
            # Get data for matching variants
            rsids = bgen.rsids[indices]
            allele_ids = bgen.allele_ids[indices] if hasattr(bgen, 'allele_ids') else None
            
            for i, idx in enumerate(indices):
                alleles = allele_ids[i] if allele_ids is not None else ['N/A', 'N/A']
                
                results['variants'].append({
                    'rsid': rsids[i],
                    'position': int(positions[idx]),
                    'allele_ref': alleles[0] if len(alleles) > 0 else 'N/A',
                    'allele_alt': alleles[1] if len(alleles) > 1 else 'N/A'
                })
                
                # Read genotype probabilities
                probs = bgen.read(idx)
                
                # Handle (n_samples, 1, 3) format - squeeze out middle dimension
                if len(probs.shape) == 3 and probs.shape[1] == 1:
                    probs = probs.squeeze(axis=1)
                
                results['genotype_probabilities'].append(probs)
            
            results['sample_ids'] = bgen.samples if hasattr(bgen, 'samples') else None
        
        return results
    
    def extract_dosages(self, genotype_probs):
        """
        Convert genotype probabilities to dosages
        Dosage = P(AB) + 2*P(BB)
        
        Parameters:
        ------
        genotype_probs : array
            Genotype probabilities (n_samples x 3)
            Columns: P(AA), P(AB), P(BB)
        
        Returns:
        ----
        array : Dosages for each sample
        """
        return genotype_probs[:, 1] + 2 * genotype_probs[:, 2]
    
    def preview_file(self, chromosome, n_variants=5, n_samples=5):
        """
        Display a preview of the BGEN file data
        
        Parameters:
        ------
        chromosome : int
            Chromosome number to preview
        n_variants : int
            Number of variants to display (default: 5)
        n_samples : int
            Number of samples to show genotype data for (default: 5)
        """
        filepath = self.get_bgen_file(chromosome)
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        
        print(f"PREVIEW OF: {filepath.name}")
        
        with open_bgen(str(filepath), verbose=False) as bgen:
            # File summary
            print(f"File Summary:")
            print(f"  Total Variants: {bgen.nvariants:,}")
            print(f"  Total Samples: {bgen.nsamples:,}")
            print(f"  Chromosome: {chromosome}")
            print()
            
            # Sample IDs
            if hasattr(bgen, 'samples') and bgen.samples is not None:
                print(f"First {min(n_samples, len(bgen.samples))} Sample IDs:")
                for i, sample_id in enumerate(bgen.samples[:n_samples]):
                    print(f"  Sample {i+1}: {sample_id}")
                print()
            
            # Get variant info for first n variants
            n_show = min(n_variants, bgen.nvariants)
            rsids = bgen.rsids[:n_show]
            positions = bgen.positions[:n_show]
            chroms = bgen.chromosomes[:n_show] if hasattr(bgen, 'chromosomes') else [chromosome] * n_show
            allele_ids = bgen.allele_ids[:n_show] if hasattr(bgen, 'allele_ids') else None
            
            print(f"First {n_show} Variants:")
            print(f"{'-'*80}")
            
            for idx in range(n_show):
                alleles = allele_ids[idx] if allele_ids is not None else ['N/A', 'N/A']
                
                print(f"\nVariant {idx+1}:")
                print(f"  rsID:       {rsids[idx]}")
                print(f"  Chromosome: {chroms[idx]}")
                print(f"  Position:   {int(positions[idx]):,} bp")
                print(f"  Ref Allele: {alleles[0] if len(alleles) > 0 else 'N/A'}")
                print(f"  Alt Allele: {alleles[1] if len(alleles) > 1 else 'N/A'}")
                
                # Read genotype probabilities
                probs = bgen.read(idx)
                
                # Handle different probability formats
                # Common formats: (n_samples, 3) or (n_samples, 1, 3)
                if len(probs.shape) == 3 and probs.shape[1] == 1 and probs.shape[2] == 3:
                    # Shape is (n_samples, 1, 3) - squeeze out middle dimension
                    probs = probs.squeeze(axis=1)  # Now (n_samples, 3)
                
                print(f"  Genotype Probabilities (first {min(n_samples, len(probs))} samples):")
                print(f"    Format: [P(AA), P(AB), P(BB)]")
                
                for i in range(min(n_samples, len(probs))):
                    sample_label = f"Sample {i+1}"
                    if hasattr(bgen, 'samples') and bgen.samples is not None:
                        sample_label += f" ({bgen.samples[i]})"
                    
                    # Format probabilities
                    p_aa, p_ab, p_bb = probs[i]
                    print(f"    {sample_label}: [{p_aa:.3f}, {p_ab:.3f}, {p_bb:.3f}]")

def main():
    """Command-line interface for BgenReader class"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Read and analyze UK Biobank BGEN files', formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check which files exist
  %(prog)s -check
  # Get information about all files
  %(prog)s -info
  # Get information and save to CSV
  %(prog)s -info -o chromosome_info.csv
  # Preview chromosome 1
  %(prog)s -preview 1
  %(prog)s -preview 1 -variants 10 -samples 8
  # Search for specific SNPs
  %(prog)s -rsids rs7412 rs429358
  %(prog)s -rsids rs7412 rs429358 -chr 19
  # Extract genomic region
  %(prog)s -region 1 1000000 2000000
  %(prog)s -region 1 1000000 2000000 -max 100
  # Use custom directory
  %(prog)s -path /data/bgen -check
        """
    )
    
    # Global options
    parser.add_argument('-path', type=str, default='/mnt/fast/datasets/ucdatasets/gwas/ukbb/iqra/ukb_maf0.05_bgen_Iqra', help='Path to directory containing BGEN files (default: current directory)')
    parser.add_argument('-pattern', type=str, default='ukb_imp_chr{}_maf0.05.bgen', help='File naming pattern with {} for chromosome (default: ukb_imp_chr{}_maf0.05.bgen)')

    # Actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True) 
    action_group.add_argument('-check', action='store_true', help='Check which BGEN files exist')
    action_group.add_argument('-info', action='store_true', help='Get summary information about all files')
    action_group.add_argument('-preview', type=int, metavar='CHR', help='Preview a specific chromosome')
    action_group.add_argument('-rsids', nargs='+', metavar='RSID', help='Search for specific rsIDs')
    action_group.add_argument('-region', nargs=3, metavar=('CHR', 'START', 'END'), help='Extract variants from genomic region')
    
    # Modifiers
    parser.add_argument('-variants', '-v', type=int, default=2, help='Number of variants to display (default: 5)')
    parser.add_argument('-samples', '-s', type=int, default=5, help='Number of samples to display (default: 5)')
    parser.add_argument('-chr', '-c', type=int, metavar='CHR', help='Limit rsID search to specific chromosome')
    parser.add_argument('-max', '-m', type=int, metavar='N', help='Maximum variants to return from region')
    parser.add_argument('-output', '-o', type=str, default='/mnt/fast/nobackup/users/if00208/5_disease_experiments/preprocessing/ukbb_chr_info.csv', help='Save results to file (CSV format for -info, -region, and -rsids)')
    
    args = parser.parse_args()
    
    # Initialize reader
    reader = BgenReader(base_path=args.path, file_pattern=args.pattern)
    
    print("BGEN File Reader for UK Biobank Data")
    print()
    
    # Execute actions
    if args.check:
        print("Checking for BGEN files...")
        existing, missing = reader.check_files_exist()
        print()
        if not existing:
            print("No BGEN files found. Please check the file path and pattern.")
            return
    
    elif args.info:
        print("Getting file information...")
        info_df = reader.get_all_files_info()
        if len(info_df) > 0:
            print(info_df.to_string(index=False))
            print()
            print(f"Total variants across all chromosomes: {info_df['n_variants'].sum():,}")

            # Save to CSV if output file specified
            if args.output:
                info_df.to_csv(args.output, index=False)
                print(f"Saved file information to: {args.output}")
        else:
            print("No BGEN files found!")
        print()
    
    elif args.preview:
        print(f"Previewing chromosome {args.preview}...")
        reader.preview_file(
            chromosome=args.preview,
            n_variants=args.variants,
            n_samples=args.samples
        )
    
    elif args.rsids:
        chromosomes = [args.chr] if args.chr else None
        print(f"Searching for rsIDs: {', '.join(args.rsids)}")
        if args.chr:
            print(f"Limiting search to chromosome {args.chr}")
        
        results = reader.read_variants_by_rsid(args.rsids, chromosome=args.chr)
        
        if results:
            print(f"\nFound {len(results)} variant(s):\n")
            for rsid, data in results.items():
                dosages = reader.extract_dosages(data['genotype_probabilities'])
                allele_freq = dosages.mean() / 2
                
                print(f"{rsid}:")
                print(f"  Position:   chr{data['chromosome']}:{data['position']:,}")
                print(f"  Alleles:    {data['allele_ref']}/{data['allele_alt']}")
                print(f"  Samples:    {len(dosages):,}")
                print(f"  Mean Dosage: {dosages.mean():.4f}")
                print(f"  Alt Allele Freq: {allele_freq:.4f} ({allele_freq*100:.2f}%)")
                print()
            
            if args.output:
                # Save dosages to CSV
                for rsid, data in results.items():
                    dosages = reader.extract_dosages(data['genotype_probabilities'])
                    df = pd.DataFrame({
                        'sample_id': data['sample_ids'] if data['sample_ids'] is not None else range(len(dosages)),
                        'dosage': dosages
                    })
                    output_file = args.output.replace('.csv', f'_{rsid}.csv')
                    df.to_csv(output_file, index=False)
                    print(f"Saved dosages to: {output_file}")
        else:
            print("No matching variants found!")
        print()
    
    elif args.region:
        chromosome = int(args.region[0])
        start = int(args.region[1])
        end = int(args.region[2])
        
        print(f"Extracting chr{chromosome}:{start:,}-{end:,}")
        if args.max:
            print(f"Limiting to {args.max} variants")
        
        region_data = reader.read_region(chromosome, start, end, max_variants=args.max)
        
        if region_data and len(region_data['variants']) > 0:
            print(f"\nExtracted {len(region_data['variants'])} variant(s)")
            
            # Display first 10
            for i, var in enumerate(region_data['variants'][:10]):
                dosages = reader.extract_dosages(region_data['genotype_probabilities'][i])
                print(f"{var['rsid']:15s} chr{chromosome}:{var['position']:10,}  "
                      f"{var['allele_ref']}/{var['allele_alt']:5s}  MAF: {dosages.mean()/2:.4f}")
            
            if len(region_data['variants']) > 10:
                print(f"... and {len(region_data['variants']) - 10} more")
            
            if args.output:
                # Create DataFrame with variant info and dosages
                variants_list = []
                for i, var in enumerate(region_data['variants']):
                    dosages = reader.extract_dosages(region_data['genotype_probabilities'][i])
                    variants_list.append({
                        'rsid': var['rsid'],
                        'chromosome': chromosome,
                        'position': var['position'],
                        'ref_allele': var['allele_ref'],
                        'alt_allele': var['allele_alt'],
                        'mean_dosage': dosages.mean(),
                        'alt_allele_freq': dosages.mean() / 2,
                        'n_samples': len(dosages)
                    })
                
                df = pd.DataFrame(variants_list)
                df.to_csv(args.output, index=False)
                print(f"\nSaved variant information to: {args.output}")
        else:
            print("No variants found in this region!")
        print()
    
    print("Analysis completed!")


if __name__ == "__main__":
    main()