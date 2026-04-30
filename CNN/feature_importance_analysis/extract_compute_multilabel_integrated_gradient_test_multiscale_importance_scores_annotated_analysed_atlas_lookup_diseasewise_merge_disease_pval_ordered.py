#!/usr/bin/env python3
"""
Process CSV file to group rows by rsid, sort by pval within groups,
order groups by decreasing group size, and optionally filter to keep
only one row per trait (for specific traits) within each rsid group.
"""

import pandas as pd
import sys

# Define the 5 traits of interest
TARGET_TRAITS = ['diabetes', 'prostate', 'pancreatic', 'colon', 'colorectal', 'breast']

def match_trait(trait_value):
    """
    Check if a trait value matches any of the target traits.
    Returns the matched trait keyword or None.
    
    For 'colon' and 'colorectal', we consider them as the same trait
    and return 'colon_colorectal' to group them together.
    """
    if pd.isna(trait_value):
        return None
    
    trait_lower = str(trait_value).lower()
    
    # Check for colon/colorectal first (treat as same trait)
    if 'colon' in trait_lower or 'colorectal' in trait_lower:
        return 'colon_colorectal'
    
    # Check other traits
    for keyword in ['diabetes', 'prostate', 'pancreatic', 'breast']:
        if keyword in trait_lower:
            return keyword
    
    return None

def process_csv(input_file, output_file, filtered_output_file):
    """
    Process CSV file:
    1. Group rows by rsid
    2. Sort within each group by increasing pval
    3. Order groups by decreasing number of rows
    4. Create filtered version keeping only lowest pval per matched trait within each rsid group
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (all data, ordered)
        filtered_output_file: Path to filtered output CSV file
    """
    print(f"Reading input file: {input_file}")
    
    # Read the CSV file - comma separated
    df = pd.read_csv(input_file, sep=',')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    print(f"Total rows: {len(df)}")
    print(f"Columns found: {list(df.columns)}")
    
    # Check if required columns exist
    if 'rsid' not in df.columns:
        print("\nError: 'rsid' column not found!")
        print("Available columns:", list(df.columns))
        raise KeyError("'rsid' column not found in CSV file")
    
    if 'pval' not in df.columns:
        print("\nError: 'pval' column not found!")
        print("Available columns:", list(df.columns))
        raise KeyError("'pval' column not found in CSV file")
    
    if 'trait' not in df.columns:
        print("\nError: 'trait' column not found!")
        print("Available columns:", list(df.columns))
        raise KeyError("'trait' column not found in CSV file")
    
    print(f"Unique rsids: {df['rsid'].nunique()}")
    
    # Count rows per rsid
    rsid_counts = df.groupby('rsid').size().reset_index(name='group_size')
    
    # Calculate minimum pval per rsid (for tie-breaking when group sizes are equal)
    rsid_min_pval = df.groupby('rsid')['pval'].min().reset_index(name='min_pval')
    
    # Merge group size and min pval back to original dataframe
    df = df.merge(rsid_counts, on='rsid', how='left')
    df = df.merge(rsid_min_pval, on='rsid', how='left')
    
    # Sort by:
    # 1. Group size (descending) - groups with most rows first
    # 2. Minimum pval in group (ascending) - tie-breaker for same-sized groups
    # 3. rsid (to keep groups together)
    # 4. pval (ascending) - within each group, lowest pval first
    df_sorted = df.sort_values(
        by=['group_size', 'min_pval', 'rsid', 'pval'],
        ascending=[False, True, True, True]
    )
    
    # Remove the temporary columns
    df_sorted = df_sorted.drop(columns=['group_size', 'min_pval'])
    
    # Write the original sorted file
    print(f"\nWriting output file: {output_file}")
    df_sorted.to_csv(output_file, sep=',', index=False)
    print(f"Output rows: {len(df_sorted)}")
    
    # Now create filtered version
    print("\n" + "="*60)
    print("Creating filtered version for specific traits...")
    print("="*60)
    
    # Add a column to identify which trait (if any) each row matches
    df_sorted['matched_trait'] = df_sorted['trait'].apply(match_trait)
    
    # Count matches before filtering
    matched_count = df_sorted['matched_trait'].notna().sum()
    print(f"Rows matching target traits: {matched_count}")
    
    # Filter within each rsid group
    filtered_rows = []
    removed_count = 0
    
    for rsid, group in df_sorted.groupby('rsid', sort=False):
        # Separate rows that match target traits from those that don't
        matched_rows = group[group['matched_trait'].notna()].copy()
        unmatched_rows = group[group['matched_trait'].isna()].copy()
        
        # For matched rows, keep only the one with lowest pval per matched trait
        if len(matched_rows) > 0:
            # Group by matched trait and keep the row with minimum pval
            kept_indices = matched_rows.groupby('matched_trait')['pval'].idxmin()
            kept_matched = matched_rows.loc[kept_indices]
            
            removed_count += len(matched_rows) - len(kept_matched)
            filtered_rows.append(kept_matched)
        
        # Keep all unmatched rows (rows that don't match any of the 5 target traits)
        if len(unmatched_rows) > 0:
            filtered_rows.append(unmatched_rows)
    
    # Combine all filtered rows
    df_filtered = pd.concat(filtered_rows, ignore_index=False)
    
    # Remove the temporary matched_trait column
    df_filtered = df_filtered.drop(columns=['matched_trait'])
    
    # Re-calculate group sizes after filtering
    rsid_counts_filtered = df_filtered.groupby('rsid').size().reset_index(name='group_size')
    
    # Calculate minimum pval per rsid in filtered data (for tie-breaking)
    rsid_min_pval_filtered = df_filtered.groupby('rsid')['pval'].min().reset_index(name='min_pval')
    
    # Merge group size and min pval back to filtered dataframe
    df_filtered = df_filtered.merge(rsid_counts_filtered, on='rsid', how='left')
    df_filtered = df_filtered.merge(rsid_min_pval_filtered, on='rsid', how='left')
    
    # Re-sort by:
    # 1. Group size (descending) - groups with most rows first
    # 2. Minimum pval in group (ascending) - tie-breaker for same-sized groups
    # 3. rsid (to keep groups together)
    # 4. pval (ascending) - within each group, lowest pval first
    df_filtered = df_filtered.sort_values(
        by=['group_size', 'min_pval', 'rsid', 'pval'],
        ascending=[False, True, True, True]
    )
    
    # Remove only the temporary min_pval column, keep group_size
    df_filtered = df_filtered.drop(columns=['min_pval'])
    
    # Write to filtered output file
    print(f"\nWriting filtered output file: {filtered_output_file}")
    df_filtered.to_csv(filtered_output_file, sep=',', index=False)
    
    # Print statistics
    print("\nFiltering complete!")
    print(f"Filtered output rows: {len(df_filtered)}")
    print(f"Rows removed by filtering: {removed_count}")
    print(f"Unique rsids in filtered output: {df_filtered['rsid'].nunique()}")
    
    # Show top rsids by group size in filtered data
    top_rsids_filtered = rsid_counts_filtered.sort_values('group_size', ascending=False).head(10)
    print("\nTop 10 rsids by number of entries (filtered data):")
    for idx, row in top_rsids_filtered.iterrows():
        print(f"  {row['rsid']}: {row['group_size']} entries")
    
    # Show top rsids by group size in original data
    top_rsids = rsid_counts.sort_values('group_size', ascending=False).head(10)
    print("\nTop 10 rsids by number of entries (original data):")
    for idx, row in top_rsids.iterrows():
        print(f"  {row['rsid']}: {row['group_size']} entries")

if __name__ == "__main__":
    # Define file names
    input_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp_disease_wise_no_cov/ig_50/analysis_20260102_122403/filtered_snps/All-Diseases_all_snps_importance_integrated_gradients_test_set_std_filtered_altas_lookup_cancer_diabetes_deduplicated_manual.csv"
    output_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp_disease_wise_no_cov/ig_50/analysis_20260102_122403/filtered_snps/All-Diseases_all_snps_importance_integrated_gradients_test_set_std_filtered_altas_lookup_cancer_diabetes_deduplicated_manual_trait_pval_ordered.csv"
    filtered_output_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp_disease_wise_no_cov/ig_50/analysis_20260102_122403/filtered_snps/All-Diseases_all_snps_importance_integrated_gradients_test_set_std_filtered_altas_lookup_cancer_diabetes_deduplicated_manual_trait_pval_ordered_filtered.csv"
    
    try:
        process_csv(input_file, output_file, filtered_output_file)
        print("\n" + "="*60)
        print("SUCCESS: Both files created successfully!")
        print("="*60)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please make sure the file is in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)