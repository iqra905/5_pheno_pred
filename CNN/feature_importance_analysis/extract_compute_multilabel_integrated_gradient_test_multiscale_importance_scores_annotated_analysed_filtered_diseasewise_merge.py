#!/usr/bin/env python3
"""
Merge multiple SNP importance CSV files and remove duplicate rows.
Duplicates are detected based on all columns EXCEPT EA, NEA, ref_allele, and alt_allele.
When duplicates exist, keep the row with the maximum Importance_Score.
Final output is sorted by pval in ascending order.
Input files:
- breacancer_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv
- crc_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv
- panca_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv
- pros01_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv
- t2dm_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv
"""
import pandas as pd
from pathlib import Path
import sys

def main():
    # =========================
    # Configuration
    # =========================
    input_dir = Path("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp_disease_wise_no_cov/ig_50/analysis_20260102_122403/filtered_snps")  
    output_file = input_dir / (
        "All-Diseases_all_snps_importance_integrated_gradients_test_set_std_filtered_std_deduplicated.csv"
    )
    
    files = [
        "breacancer_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv",
        "crc_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv",
        "panca_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv",
        "pros01_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv",
        "t2dm_all_snps_importance_integrated_gradients_disease_wise_test_set_std_filtered.csv",
    ]
    
    # =========================
    # Load CSV files
    # =========================
    dataframes = []
    total_rows_before = 0
    
    print("Loading input files:\n")
    for fname in files:
        fpath = input_dir / fname
        if not fpath.exists():
            print(f"ERROR: File not found: {fpath}", file=sys.stderr)
            sys.exit(1)
        
        df = pd.read_csv(fpath)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        row_count = len(df)
        total_rows_before += row_count
        
        print(f"  {fname}")
        print(f"    Rows: {row_count}")
        dataframes.append(df)
    
    print("\n----------------------------------------")
    print(f"Total rows before merge: {total_rows_before}")
    
    # =========================
    # Merge all data
    # =========================
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Rows after concatenation: {len(merged_df)}")
    
    # Print actual column names for debugging
    print(f"\nActual columns in merged data: {list(merged_df.columns)}")
    
    # =========================
    # Remove duplicates based on all columns EXCEPT EA, NEA, ref_allele, alt_allele
    # Keep row with maximum Importance_Score
    # =========================
    # Columns to exclude from duplicate detection
    exclude_cols = ['Importance_Score', 'Rank']
    
    # Get all columns except the excluded ones (case-insensitive check)
    cols_to_check = [col for col in merged_df.columns 
                     if col not in exclude_cols and col.lower() not in [c.lower() for c in exclude_cols]]
    
    print(f"\nColumns excluded from duplicate detection: {exclude_cols}")
    print(f"Columns used for duplicate detection: {cols_to_check}")
    
    # Check for duplicates before removing
    duplicates_mask = merged_df.duplicated(subset=cols_to_check, keep=False)
    num_duplicate_rows = duplicates_mask.sum()
    print(f"\nTotal rows involved in duplicates: {num_duplicate_rows}")
    
    # Show some example duplicates
    if num_duplicate_rows > 0:
        print("\nExample duplicate groups (showing Importance_Score):")
        duplicate_examples = merged_df[duplicates_mask].head(10)
        print(duplicate_examples[['SNP_Index']].to_string())
    
    # Sort by Importance_Score in descending order (highest first)
    # Then drop duplicates, keeping the first occurrence (which will be the max Importance_Score)
    merged_df_sorted = merged_df.sort_values('Importance_Score', ascending=False)
    dedup_df = merged_df_sorted.drop_duplicates(subset=cols_to_check, keep='first')
    
    duplicates_removed = len(merged_df) - len(dedup_df)
    
    print("\n----------------------------------------")
    print(f"Rows after deduplication: {len(dedup_df)}")
    print(f"Duplicate rows removed (keeping max Importance_Score): {duplicates_removed}")
    
    # Verify no duplicates remain
    remaining_duplicates = dedup_df.duplicated(subset=cols_to_check).sum()
    print(f"Remaining duplicates after removal: {remaining_duplicates}")
    
    if remaining_duplicates > 0:
        print("\nWARNING: Duplicates still remain! This shouldn't happen.")
    
    # =========================
    # Sort final output by Importance_Score in Descending order
    # =========================
    dedup_df = dedup_df.sort_values('Importance_Score', ascending=False)
    print("\nFinal data sorted by Importance_Score (descending order)")
    
    # =========================
    # Save output
    # =========================
    dedup_df.to_csv(output_file, index=False)
    
    print("\n----------------------------------------")
    print("Output written to:")
    print(output_file)

if __name__ == "__main__":
    main()