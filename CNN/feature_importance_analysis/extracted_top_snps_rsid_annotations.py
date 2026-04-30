import csv
import pandas as pd
import argparse
import os
import re

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Annotate significant SNPs with genomic information from .gen files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python snp_annotator.py -s results.csv -g metadata.gen -o annotated.csv
  
  # With verbose output
  python snp_annotator.py -s top_snps.csv -g data.gen -o output.csv --verbose
        """
    )
    
    # Required arguments
    parser.add_argument("-significant_snps_file", type=str, required=True, help="Path to CSV file containing significant SNPs with SNP_Index column")
    
    parser.add_argument("-gen_file", type=str, required=True, help="Path to .gen file containing genomic metadata")
    
    parser.add_argument("-output_file", type=str, required=True, help="Path for output CSV file with annotated SNPs")
    
    # Optional arguments
    parser.add_argument("-verbose", action="store_true", help="Enable verbose output")
    
    parser.add_argument("-validate_order", action="store_true", help="Validate chromosome ordering in .gen file")
    
    return parser.parse_args()

def natural_sort_key(text):
    """
    Convert chromosome strings to natural sorting keys
    Examples: '1' -> [1], '10' -> [10], 'X' -> ['X'], 'MT' -> ['MT']
    """
    def convert(text):
        if text.isdigit():
            return int(text)
        else:
            return text.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', str(text))]

def validate_chromosome_order(df_gen, verbose=False):
    """
    Validate that chromosomes in .gen file are in proper numerical order
    """
    print(" Validating chromosome order in .gen file...")
    
    # Get unique chromosomes in order they appear
    chromosomes_in_order = df_gen['chromosome'].unique()
    
    # Get naturally sorted chromosomes
    chromosomes_sorted = sorted(chromosomes_in_order, key=natural_sort_key)
    
    if verbose:
        print(f"  • Chromosomes in .gen file order: {list(chromosomes_in_order)}")
        print(f"  • Expected numerical order: {chromosomes_sorted}")
    
    # Check if they match
    if list(chromosomes_in_order) == chromosomes_sorted:
        print("   Chromosomes are in correct numerical order")
        return True
    else:
        print("    WARNING: Chromosomes are NOT in numerical order!")
        print(f"     File order: {list(chromosomes_in_order)}")
        print(f"     Expected:   {chromosomes_sorted}")
        return False

def check_snp_index_integrity(df_gen, significant_indices, verbose=False):
    """
    Check if SNP indices from results file are within valid range
    """
    print("🔍 Checking SNP index integrity...")
    
    max_available_index = len(df_gen) - 1
    max_requested_index = max(significant_indices)
    min_requested_index = min(significant_indices)
    
    print(f"  • SNP indices in .gen file: 0 to {max_available_index:,}")
    print(f"  • SNP indices in results: {min_requested_index:,} to {max_requested_index:,}")
    
    if max_requested_index > max_available_index:
        print(f"   ERROR: Some SNP indices exceed .gen file range!")
        print(f"     Highest requested: {max_requested_index:,}")
        print(f"     Highest available: {max_available_index:,}")
        print(f"     Missing indices: {max_requested_index - max_available_index:,}")
        return False
    
    if min_requested_index < 0:
        print(f"   ERROR: Negative SNP indices found: {min_requested_index}")
        return False
    
    print("   All SNP indices are within valid range")
    return True

def create_position_based_mapping(df_gen, verbose=False):
    """
    Create mapping that preserves exact .gen file order (critical for SNP_Index correspondence)
    """
    print("  Creating position-based SNP mapping...")
    
    snp_info = {}
    chromosome_stats = {}
    
    for index, row in df_gen.iterrows():
        # Store exact position-based mapping
        snp_info[index] = {
            'chromosome': str(row['chromosome']), 
            'snp_id': str(row['snp_id']),
            'bp': int(row['bp']),
            'ref': str(row['ref']),
            'alt': str(row['alt'])
        }
        
        # Track chromosome statistics
        chrom = str(row['chromosome'])
        if chrom not in chromosome_stats:
            chromosome_stats[chrom] = {'count': 0, 'first_index': index, 'last_index': index}
        chromosome_stats[chrom]['count'] += 1
        chromosome_stats[chrom]['last_index'] = index
    
    if verbose:
        print("   Chromosome statistics in .gen file:")
        for chrom in sorted(chromosome_stats.keys(), key=natural_sort_key):
            stats = chromosome_stats[chrom]
            print(f"     Chr {chrom}: {stats['count']:,} SNPs (indices {stats['first_index']:,}-{stats['last_index']:,})")
    
    print(f"   Created mapping for {len(snp_info):,} SNPs")
    return snp_info

def update_snp_data_from_gen(significant_snps_file, gen_file, output_file, verbose=False, validate_order=False):
    """
    Update SNP data with genomic information from .gen files
    Now with proper chromosome ordering validation
    """
    
    if verbose:
        print(f" Input files:")
        print(f"  • SNPs file: {significant_snps_file}")
        print(f"  • Gen file: {gen_file}")
        print(f"  • Output file: {output_file}")
        print()
    
    # Validate input files
    if not os.path.exists(significant_snps_file):
        raise FileNotFoundError(f"Significant SNPs file not found: {significant_snps_file}")
    
    if not os.path.exists(gen_file):
        raise FileNotFoundError(f".gen file not found: {gen_file}")
    
    # Read the significant SNPs file
    print(f" Reading significant SNPs from: {os.path.basename(significant_snps_file)}")
    try:
        df_snps = pd.read_csv(significant_snps_file)
        if 'SNP_Index' not in df_snps.columns:
            raise ValueError("SNP_Index column not found in significant SNPs file")
        print(f"   Found {len(df_snps)} significant SNPs")
    except Exception as e:
        raise ValueError(f"Error reading significant SNPs file: {e}")
    
    # Read the .gen file metadata - PRESERVE EXACT ORDER
    print(f" Reading .gen file metadata: {os.path.basename(gen_file)}")
    
    column_names = ['chromosome', 'snp_id', 'bp', 'ref', 'alt']
    
    try:
        # CRITICAL: Read in exact order as file appears
        df_gen = pd.read_csv(gen_file, 
                            sep=r'\s+',  # Space or tab separated
                            header=None, 
                            names=column_names,
                            usecols=[0, 1, 2, 3, 4],  # Only read first 5 columns
                            dtype={'chromosome': str, 'snp_id': str, 'bp': int, 'ref': str, 'alt': str})
        
        print(f"  Successfully read {len(df_gen):,} SNPs from .gen file")
        
        if verbose:
            print(f"  • Columns: {list(df_gen.columns)}")
            unique_chroms = df_gen['chromosome'].unique()
            print(f"  • Chromosomes in file order: {list(unique_chroms)}")
        
    except Exception as e:
        print(f"   Error reading .gen file: {e}")
        print(f"   Check that the .gen file uses Oxford format: SNP_ID chromosome position ref_allele alt_allele")
        raise
    
    # Validate chromosome order if requested
    if validate_order:
        is_order_correct = validate_chromosome_order(df_gen, verbose)
        if not is_order_correct:
            print("  ⚠️  Consider sorting your .gen file by chromosome and position if this causes issues")
    
    # Check SNP index integrity
    significant_indices = df_snps['SNP_Index'].tolist()
    index_integrity = check_snp_index_integrity(df_gen, significant_indices, verbose)
    
    if not index_integrity:
        raise ValueError("SNP index integrity check failed. Please verify your data preprocessing.")
    
    # Create position-based mapping (preserves .gen file order)
    snp_info = create_position_based_mapping(df_gen, verbose)
    
    # Add new columns to the significant SNPs dataframe
    print(" Mapping SNP indices to genomic coordinates...")
    df_snps['chromosome'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('chromosome', 'N/A'))
    df_snps['snp_id'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('snp_id', 'N/A'))
    df_snps['bp'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('bp', 'N/A'))
    df_snps['ref_allele'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('ref', 'N/A'))
    df_snps['alt_allele'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('alt', 'N/A'))
    
    # Check mapping success
    mapped_count = (df_snps['chromosome'] != 'N/A').sum()
    mapping_rate = mapped_count/len(df_snps)*100
    
    print(f"   Successfully mapped {mapped_count:,}/{len(df_snps):,} SNPs ({mapping_rate:.1f}%)")
    
    if mapped_count < len(df_snps):
        unmapped_count = len(df_snps) - mapped_count
        print(f"    Warning: {unmapped_count:,} SNPs could not be mapped")
        
        if verbose:
            # Show some unmapped indices for debugging
            unmapped_indices = df_snps[df_snps['chromosome'] == 'N/A']['SNP_Index'].head(5).tolist()
            print(f"  • Example unmapped SNP_Index values: {unmapped_indices}")
            print(f"  • Max SNP_Index in results: {df_snps['SNP_Index'].max():,}")
            print(f"  • Max index in .gen file: {len(df_gen) - 1:,}")
    
    # Analysis of mapped results
    if mapped_count > 0:
        print("\n Mapping Results Analysis:")
        mapped_df = df_snps[df_snps['chromosome'] != 'N/A']
        
        # Show chromosome distribution
        chrom_counts = mapped_df['chromosome'].value_counts()
        chrom_counts_sorted = chrom_counts.reindex(sorted(chrom_counts.index, key=natural_sort_key))
        
        print("  SNPs per chromosome:")
        for chrom, count in chrom_counts_sorted.items():
            percentage = (count / mapped_count) * 100
            print(f"     Chr {chrom}: {count:,} SNPs ({percentage:.1f}%)")
        
        if verbose:
            # Show position ranges
            print("  📍 Position ranges per chromosome:")
            for chrom in sorted(mapped_df['chromosome'].unique(), key=natural_sort_key):
                chrom_data = mapped_df[mapped_df['chromosome'] == chrom]
                if len(chrom_data) > 0 and chrom_data['bp'].iloc[0] != 'N/A':
                    min_pos = chrom_data['bp'].min()
                    max_pos = chrom_data['bp'].max()
                    print(f"     Chr {chrom}: {min_pos:,} - {max_pos:,} bp")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f" Created output directory: {output_dir}")
    
    # Save the updated dataframe to a new CSV file
    df_snps.to_csv(output_file, index=False)
    print(f"Updated data saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n Final Summary:")
    print(f"  • Total SNPs processed: {len(df_snps):,}")
    print(f"  • Successfully mapped: {mapped_count:,} ({mapping_rate:.1f}%)")
    print(f"  • Unmapped SNPs: {len(df_snps) - mapped_count:,}")
    
    if mapped_count > 0:
        chromosomes = sorted([c for c in df_snps[df_snps['chromosome'] != 'N/A']['chromosome'].unique() if c != 'N/A'], key=natural_sort_key)
        print(f"  • Chromosomes represented: {chromosomes}")
    
    return df_snps

def main():
    """Main function to run the SNP annotation script"""
    args = parse_args()
    
    print(" SNP Genomic Annotation")
    print("="*60)
    
    # Validate arguments
    if not os.path.exists(args.significant_snps_file):
        print(f" Error: Significant SNPs file not found: {args.significant_snps_file}")
        return 1
    
    if not os.path.exists(args.gen_file):
        print(f" Error: .gen file not found: {args.gen_file}")
        return 1
    
    try:
        # Run the annotation
        print(" Starting SNP annotation...")
        updated_df = update_snp_data_from_gen(
            args.significant_snps_file,
            args.gen_file, 
            args.output_file,
            args.verbose,
            args.validate_order
        )
        
        print(f"\n Success! Annotation completed")
        print(f" Output saved to: {args.output_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n Error during annotation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print(f"\n Troubleshooting tips:")
        print(f"  • Use --validate_order to check chromosome ordering")
        print(f"  • Ensure .gen file has proper numerical chromosome order")
        print(f"  • Check that SNP_Index values match .gen file preprocessing")
        print(f"  • Use --verbose for detailed error information")
        return 1

if __name__ == "__main__":
    exit(main())