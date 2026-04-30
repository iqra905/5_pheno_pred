import pandas as pd
import os
import argparse
from itertools import zip_longest

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process disease files to find overlapping SNP indices between two sets of CSV files.")
    
    parser.add_argument('-summ_stats_files_path', type=str, default="/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/extracted_5M", help="Base path for Set 1 files (files with 'Global_SNP_Index' column)")
    parser.add_argument('-feat_imp_files_path', type=str, default="/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/01_snps_5d_ks/feat_imp_1M", help="Base path for Set 2 files (files with 'SNP_Index' column)")
    parser.add_argument('-pros_snps', type=int, default=1000000, help="Number of top SNPs for prostate cancer (default: 81854)")
    parser.add_argument('-pan_snps', type=int, default=1000000, help="Number of top SNPs for pancreatic cancer (default: 52396)")
    parser.add_argument('-col_snps', type=int, default=1000000, help="Number of top SNPs for colorectal cancer (default: 54455)")
    parser.add_argument('-brea_snps', type=int, default=1000000, help="Number of top SNPs for breast cancer (default: 214466)")
    parser.add_argument('-t2d_snps', type=int, default=1000000, help="Number of top SNPs for type 2 diabetes (default: 146778)")
    
    return parser.parse_args()

def process_disease_files(set1_base_path, set2_base_path, top_snps_config):
    """
    Process disease files to find overlapping SNP indices between two sets of CSV files.
    
    Args:
        set1_base_path (str): Base path for Set 1 files
        set2_base_path (str): Base path for Set 2 files  
        top_snps_config (dict): Dictionary with top_snps values for each disease
    
    Creates a single CSV file with common indices for all diseases and optionally merged data.
    """
    
    # Define disease mappings with configurable top_snps
    diseases = {
        'pros': {
            'set1_file': 'pros_0.1.csv',
            'set2_prefix': 'pros01_top_',
            'top_snps': top_snps_config['pros']
        },
        'pan': {
            'set1_file': 'pan_0.1.csv',
            'set2_prefix': 'panca_top_',
            'top_snps': top_snps_config['pan']
        },
        'col': {
            'set1_file': 'col_0.1.csv',
            'set2_prefix': 'crc_top_',
            'top_snps': top_snps_config['col']
        },
        'brea': {
            'set1_file': 'brea_0.1.csv',
            'set2_prefix': 'breacancer_top_',
            'top_snps': top_snps_config['brea']
        },
        't2d': {
            'set1_file': 't2d_0.1.csv',
            'set2_prefix': 't2dm_top_',
            'top_snps': top_snps_config['t2d']
        }
    }
    
    print("CONFIGURATION")
    print("=" * 50)
    print(f"Set 1 base path: {set1_base_path}")
    print(f"Set 2 base path: {set2_base_path}")
    print("Top SNPs configuration:")
    for disease, config in diseases.items():
        print(f"  {disease.upper()}: {config['top_snps']:,}")
    print("\n")
    
    # Create summary report and collect common indices
    summary_report = []
    all_common_indices = {}  # Dictionary to store common indices for each disease
    
    for disease, info in diseases.items():
        print(f"Processing {disease.upper()}...")
        print("-" * 50)
        
        # Construct file paths
        set1_file_path = os.path.join(set1_base_path, info['set1_file'])
        set2_file_path = os.path.join(set2_base_path, 
                                    f"{info['set2_prefix']}{info['top_snps']}_snps_loss_based_test_set_disease_specific_weighted.csv")
        
        print(f"Set 1 file: {set1_file_path}")
        print(f"Set 2 file: {set2_file_path}")
        
        try:
            # Check if files exist
            if not os.path.exists(set1_file_path):
                print(f"Error: Set 1 file not found - {set1_file_path}")
                all_common_indices[disease.upper()] = []  # Empty list for missing data
                continue
                
            if not os.path.exists(set2_file_path):
                print(f"Error: Set 2 file not found - {set2_file_path}")
                all_common_indices[disease.upper()] = []  # Empty list for missing data
                continue
            
            # Read the CSV files
            print(f"Reading Set 1: {info['set1_file']}")
            df1 = pd.read_csv(set1_file_path)
            
            print(f"Reading Set 2: {info['set2_prefix']}{info['top_snps']}_snps_loss_based_test_set_disease_specific_weighted.csv")
            df2 = pd.read_csv(set2_file_path)
            
            # Check if required columns exist
            if 'Global_SNP_Index' not in df1.columns:
                print(f"Warning: 'Global_SNP_Index' column not found in Set 1 file")
                print(f"Available columns: {list(df1.columns)}")
                all_common_indices[disease.upper()] = []  # Empty list for missing data
                continue
            
            if 'SNP_Index' not in df2.columns:
                print(f"Warning: 'SNP_Index' column not found in Set 2 file")
                print(f"Available columns: {list(df2.columns)}")
                all_common_indices[disease.upper()] = []  # Empty list for missing data
                continue
            
            # Get unique indices from both sets
            set1_indices = set(df1['Global_SNP_Index'].dropna())
            set2_indices = set(df2['SNP_Index'].dropna())
            
            # Find common indices
            common_indices = set1_indices.intersection(set2_indices)
            
            # Store sorted common indices for this disease
            all_common_indices[disease.upper()] = sorted(list(common_indices))
            
            # Print statistics
            print(f"Set 1 unique indices: {len(set1_indices):,}")
            print(f"Set 2 unique indices: {len(set2_indices):,}")
            print(f"Common indices: {len(common_indices):,}")
            if len(set1_indices) > 0 and len(set2_indices) > 0:
                print(f"Overlap percentage: {(len(common_indices) / min(len(set1_indices), len(set2_indices))) * 100:.2f}%")
            
            # Store summary info
            summary_report.append({
                'Disease': disease.upper(),
                'Set1_Indices': len(set1_indices),
                'Set2_Indices': len(set2_indices),
                'Common_Indices': len(common_indices),
                'Overlap_Percentage': round((len(common_indices) / min(len(set1_indices), len(set2_indices))) * 100, 2) if len(set1_indices) > 0 and len(set2_indices) > 0 else 0
            })
            
            if len(common_indices) == 0:
                print("Warning: No common indices found!")
            
            # Create individual merged dataset for this disease (optional)
            if len(common_indices) > 0:
                # Save filtered data from Set 1 (with common indices only)
                df1_filtered = df1[df1['Global_SNP_Index'].isin(common_indices)].copy()
                
                # Save filtered data from Set 2 (with common indices only)
                df2_filtered = df2[df2['SNP_Index'].isin(common_indices)].copy()
                
                # Create merged dataset
                df1_filtered_renamed = df1_filtered.rename(columns={'Global_SNP_Index': 'SNP_Index'})
                
                # Add suffix to distinguish columns from each set
                df1_cols_to_rename = [col for col in df1_filtered_renamed.columns if col != 'SNP_Index']
                df2_cols_to_rename = [col for col in df2_filtered.columns if col != 'SNP_Index']
                
                for col in df1_cols_to_rename:
                    df1_filtered_renamed.rename(columns={col: f"{col}_set1"}, inplace=True)
                
                for col in df2_cols_to_rename:
                    df2_filtered.rename(columns={col: f"{col}_set2"}, inplace=True)
                
                # Merge the datasets
                merged_df = pd.merge(df1_filtered_renamed, df2_filtered, on='SNP_Index', how='inner')
                merged_file = os.path.join(set2_base_path, f"{disease}_merged_common_indices.csv")
                merged_df.to_csv(merged_file, index=False)
                print(f"✓ Saved merged data to: {merged_file}")
            
        except Exception as e:
            print(f"Error processing {disease}: {str(e)}")
            all_common_indices[disease.upper()] = []  # Empty list for errors
            summary_report.append({
                'Disease': disease.upper(),
                'Set1_Indices': 'Error',
                'Set2_Indices': 'Error',
                'Common_Indices': 'Error',
                'Overlap_Percentage': 'Error'
            })
        
        print("\n")
    
    # Create single combined common indices file
    print("Creating combined common indices file...")
    print("-" * 50)
    
    # Create column names
    column_names = [f"{disease}_Common_SNP_Index" for disease in all_common_indices.keys()]
    
    # Use zip_longest to handle different column lengths, filling with None
    combined_data = list(zip_longest(*all_common_indices.values(), fillvalue=None))
    
    # Create DataFrame
    combined_df = pd.DataFrame(combined_data, columns=column_names)
    
    # Save combined common indices file
    combined_indices_file = os.path.join(set2_base_path, "all_diseases_common_snp_indices.csv")
    combined_df.to_csv(combined_indices_file, index=False)
    print(f"✓ Saved combined common indices to: {combined_indices_file}")
    
    # Print summary of combined file
    print(f"Combined file contains {len(combined_df)} rows")
    for col in combined_df.columns:
        non_null_count = combined_df[col].notna().sum()
        print(f"  {col}: {non_null_count:,} indices")
    
    # Save summary report
    if summary_report:
        summary_df = pd.DataFrame(summary_report)
        summary_file = os.path.join(set2_base_path, "snp_overlap_summary_report.csv")
        summary_df.to_csv(summary_file, index=False)
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print(f"\n✓ Summary report saved to: {summary_file}")

def verify_file_structure(set1_base_path, set2_base_path, top_snps_config):
    """
    Utility function to check if files exist and examine their structure.
    """
    diseases = {
        'pros': {'set1_file': 'pros_0.1.csv', 'set2_prefix': 'pros01_top_', 'top_snps': top_snps_config['pros']},
        'pan': {'set1_file': 'pan_0.1.csv', 'set2_prefix': 'panca_top_', 'top_snps': top_snps_config['pan']},
        'col': {'set1_file': 'col_0.1.csv', 'set2_prefix': 'crc_top_', 'top_snps': top_snps_config['col']},
        'brea': {'set1_file': 'brea_0.1.csv', 'set2_prefix': 'breacancer_top_', 'top_snps': top_snps_config['brea']},
        't2d': {'set1_file': 't2d_0.1.csv', 'set2_prefix': 't2dm_top_', 'top_snps': top_snps_config['t2d']}
    }
    
    print("FILE VERIFICATION")
    print("=" * 50)
    print(f"Set 1 base path: {set1_base_path}")
    print(f"Set 2 base path: {set2_base_path}")
    print()
    
    for disease, info in diseases.items():
        print(f"{disease.upper()}:")
        
        set1_file = os.path.join(set1_base_path, info['set1_file'])
        set2_file = os.path.join(set2_base_path, f"{info['set2_prefix']}{info['top_snps']}_snps_loss_based_test_set_disease_specific_weighted.csv")
        
        print(f"  Set 1: {set1_file}")
        print(f"  Exists: {os.path.exists(set1_file)}")
        
        print(f"  Set 2: {set2_file}")
        print(f"  Exists: {os.path.exists(set2_file)}")
        print()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create top SNPs configuration dictionary
    top_snps_config = {
        'pros': args.pros_snps,
        'pan': args.pan_snps,
        'col': args.col_snps,
        'brea': args.brea_snps,
        't2d': args.t2d_snps
    }
    
    # Verify files first, then run processing
    verify_file_structure(args.summ_stats_files_path, args.feat_imp_files_path, top_snps_config)
    print("\n")
    
    # Run the main processing
    process_disease_files(args.summ_stats_files_path, args.feat_imp_files_path, top_snps_config)