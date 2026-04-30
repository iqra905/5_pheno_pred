import pandas as pd
import numpy as np
import re

def natural_sort_key(snp_id):
    """Function to create a sort key that handles chromosome numbers correctly"""
    chr_num = int(snp_id.split(':')[0])  # Extract chromosome number
    snp = snp_id.split(':')[1]  # Extract SNP ID
    # Extract numeric part from SNP ID for proper sorting
    num = int(re.search(r'\d+', snp).group())
    return (chr_num, num)

def analyze_snp_data(input_file, pval_file, bim_file, highlighted_output, multiple_methods_output):
    print("\nPreliminary Analysis:")
    print("--------------------")
    print("Expected Numbers:")
    print("1. Maximum possible SNPs per method: 300")
    print("2. Number of methods: 10")
    print("3. Maximum theoretical total (if all SNPs were unique): 3000")
    
    # Read the main CSV file
    df = pd.read_csv(input_file)
    
    # Read the additional CSV file
    additional_df = pd.read_csv(pval_file)
    
    # Read the BIM file with appropriate column names
    bim_df = pd.read_csv(bim_file, sep=r'\s+', header=None,
                         names=['chr', 'snp_id', 'distance', 'position', 'ref_allele', 'alt_allele'])
    
    # Create dictionaries for quick lookup
    additional_info = {}
    for _, row in additional_df.iterrows():
        key = (int(row['chromosome_up']), str(row['snp_id_up']))
        additional_info[key] = {
            'P_Value': row['P_Value'],
            'snp_index': row['SNP_index_all']
        }
    
    # Create dictionary for BIM file information
    bim_info = {}
    for _, row in bim_df.iterrows():
        key = (int(row['chr']), str(row['snp_id']))
        bim_info[key] = row['position']
    
    # Define the method pairs and their short names
    method_pairs = [
        ('chr_sa', 'snp_id_sa', 'phenotype_sa', 'SA'),
        ('chr_dc', 'snp_id_dc', 'phenotype_dc', 'DC'),
        ('chr_gn', 'snp_id_gn', 'phenotype_gn', 'GN'),
        ('chr_mlp_0.1', 'snp_id_mlp_0.1', 'phenotype_mlp_0.1', 'MLP_0.1'),
        ('chr_mlp_chr_0.1', 'snp_id_mlp_chr_0.1', 'phenotype_mlp_chr_0.1', 'MLP_CHR_0.1'),
        ('chr_cnn_0.1', 'snp_id_cnn_0.1', 'phenotype_cnn_0.1', 'CNN_0.1'),
        ('chr_cnn_skip_0.1', 'snp_id_cnn_skip_0.1', 'phenotype_cnn_skip_0.1', 'CNN_SKIP_0.1'),
        ('chr_cnn_chr_0.1', 'snp_id_cnn_chr_0.1', 'phenotype_cnn_chr_0.1', 'CNN_CHR_0.1'),
       # ('chr_mlp_full', 'snp_id_mlp_full', 'phenotype_mlp_full', 'MLP_full'),
       # ('chr_cnn_full', 'snp_id_cnn_full', 'phenotype_cnn_full', 'CNN_full')
    ]
    
    # Modified comparison dictionary creation
    comparison_dict = {}
    sa_entries = []
    all_sa_entries = []  # To store all SA entries for later comparison
    
    # First process SA method to get all entries
    sa_method = next(m for m in method_pairs if m[3] == 'SA')
    sa_triplets = list(zip(df[sa_method[0]], df[sa_method[1]], df[sa_method[2]]))
    sa_triplets = [(chr_num, snp_id, pheno) for chr_num, snp_id, pheno in sa_triplets 
                   if pd.notna(chr_num) and pd.notna(snp_id)]
    
    # Get all SA entries with P-values
    for chr_num, snp_id, phenotype in sa_triplets:
        key = (int(chr_num), str(snp_id))
        p_value = additional_info.get(key, {}).get('P_Value', float('inf'))
        all_sa_entries.append((key, phenotype, p_value))
    
    # Sort all SA entries by P-value
    all_sa_entries.sort(key=lambda x: float(x[2]) if x[2] != 'NA' else float('inf'))
    
    # Take first 300 entries from each method
    for chr_col, snp_col, pheno_col, method_name in method_pairs:
        triplets = list(zip(df[chr_col], df[snp_col], df[pheno_col]))
        valid_triplets = [(chr_num, snp_id, pheno) for chr_num, snp_id, pheno in triplets 
                         if pd.notna(chr_num) and pd.notna(snp_id)][:300]
        
        for chr_num, snp_id, phenotype in valid_triplets:
            key = (int(chr_num), str(snp_id))
            if key not in comparison_dict:
                comparison_dict[key] = {
                    'methods': {method: 'No' for method in [pair[3] for pair in method_pairs]},
                    'phenotypes': set(),
                    'P_Value': additional_info.get(key, {}).get('P_Value', 'NA'),
                    'snp_index': additional_info.get(key, {}).get('snp_index', 'NA'),
                    'position': bim_info.get(key, 'NA')
                }
            comparison_dict[key]['methods'][method_name] = 'Yes'
            if pd.notna(phenotype):
                comparison_dict[key]['phenotypes'].add(str(phenotype))
    
    # Check for additional SA SNPs beyond 300 that match with other methods
    non_sa_snps = {key for key in comparison_dict.keys() 
                   if comparison_dict[key]['methods']['SA'] == 'No'}
    
    for key, phenotype, p_value in all_sa_entries[300:]:  # Look through remaining SA entries
        if key in non_sa_snps:  # If this SA SNP matches one found by other methods
            comparison_dict[key]['methods']['SA'] = 'Yes'
            if pd.notna(phenotype):
                comparison_dict[key]['phenotypes'].add(str(phenotype))

    # Create a new DataFrame from the comparison dictionary
    result_data = []
    for (chr_num, snp_id), data in comparison_dict.items():
        row = {
            'Chr': chr_num,
            'SNP_ID': snp_id,
            'Position': data['position'],
            'P-value': data['P_Value'],
            'SNP_index': data['snp_index']
        }
        row.update(data['methods'])
        row['Phenotype'] = '; '.join(sorted(data['phenotypes'])) if data['phenotypes'] else ''
        result_data.append(row)
    
    # Create the final DataFrame
    result_df = pd.DataFrame(result_data)

    # Filter out SNPs with ID '.'
    result_df = result_df[result_df['SNP_ID'] != '.']
    
    # Convert P-value to numeric for sorting
    result_df['P-value_num'] = pd.to_numeric(result_df['P-value'], errors='coerce')
    
    # Sort by P-value
    result_df = result_df.sort_values('P-value_num', ascending=True)
    result_df = result_df.drop('P-value_num', axis=1)
    
    # Calculate the number of methods that found each SNP
    method_columns = [pair[3] for pair in method_pairs]
    result_df['Methods_Count'] = result_df[method_columns].apply(
        lambda x: sum(x == 'Yes'), axis=1)
    
    # Reorder columns
    column_order = ['SNP_index', 'Chr', 'SNP_ID', 'Position', 'P-value'] + method_columns + ['Phenotype', 'Methods_Count']
    result_df = result_df[column_order]
    
    # Define the style function for highlighting
    def highlight_multiple_methods(row):
        if row['Methods_Count'] > 1:
            return ['background-color: #FFFF00'] * len(row)
        return [''] * len(row)
    
    # Apply styling and save highlighted Excel file
    styled_df = result_df.style.apply(highlight_multiple_methods, axis=1)
    styled_df.to_excel(highlighted_output, index=False, engine='openpyxl')
    print(f"Highlighted comparison matrix has been saved to {highlighted_output}")
    
    # Create multiple methods Excel file
    multiple_methods_df = result_df[result_df['Methods_Count'] > 1].copy()
    
    # Add count row
    count_row = {'SNP_index': 'Total', 'Chr': '', 'SNP_ID': '', 'Position': '', 'P-value': ''}
    for method in method_columns:
        count_row[method] = str(sum(multiple_methods_df[method] == 'Yes'))
    count_row['Phenotype'] = ''
    count_row['Methods_Count'] = ''
    
    multiple_methods_df.loc[len(multiple_methods_df)] = count_row
    
    # Save to Excel instead of CSV
    multiple_methods_df.to_excel(multiple_methods_output, index=False, engine='openpyxl')
    print(f"SNPs found in multiple methods have been saved to {multiple_methods_output}")
    
    # Print summary statistics with detailed analysis
    print("\nDetailed Summary Statistics:")
    print("-------------------------")
    
    # Basic counts
    total_snps = len(result_df)
    multiple_methods_snps = len(multiple_methods_df) - 1  # Subtract 1 for count row
    single_method_snps = total_snps - multiple_methods_snps
    
    print("1. Overall SNP Counts:")
    print(f"   - Total unique SNPs: {total_snps}")
    print(f"   - SNPs found in multiple methods: {multiple_methods_snps}")
    print(f"   - SNPs found in single method only: {single_method_snps}")
    
    # Calculate overlap statistics
    max_possible = 7 * 300  # 7 methods × 300 SNPs each
    overlap_rate = (max_possible - total_snps) / max_possible * 100
    
    print("\n2. Detailed Overlap Analysis:")
    print(f"   - Maximum possible SNPs (7 methods × 300 each): {max_possible}")
    print(f"   - Actual unique SNPs: {total_snps}")
    print(f"   - Overall overlap rate: {overlap_rate:.2f}%")
    
    # Analyze overlap patterns
    method_counts = result_df['Methods_Count'].value_counts().sort_index()
    print("\n   Overlap Distribution:")
    for count, num_snps in method_counts.items():
        print(f"   - Found by {count} methods: {num_snps} SNPs")
    
    # Calculate percentage of SNPs found by multiple methods
    multi_method_percentage = (multiple_methods_snps / total_snps * 100)
    print(f"\n   - Percentage of SNPs found by multiple methods: {multi_method_percentage:.2f}%")
    
    # SA method specific analysis
    sa_snps = sum(result_df['SA'] == 'Yes')
    sa_overlap = sum((result_df['SA'] == 'Yes') & (result_df['Methods_Count'] > 1))
    
    print("\n3. SA Method Analysis:")
    print(f"   - Total SNPs in SA method: {sa_snps}")
    print(f"   - SA SNPs overlapping with other methods: {sa_overlap}")
    print(f"   - SA overlap rate: {(sa_overlap/sa_snps*100 if sa_snps > 0 else 0):.2f}%")
    
    print("\nNumber of SNPs by method count:")
    print(result_df['Methods_Count'].value_counts().sort_index())
    
    print("\nChromosome-wise distribution:")
    chr_dist = result_df['Chr'].value_counts().sort_index()
    print(chr_dist)
    
    print("\nMethod-wise SNP counts:")
    for method in method_columns:
        count = sum(result_df[method] == 'Yes')
        print(f"{method}: {count}")
    
    # print("\nPhenotype distribution:")
    # all_phenotypes = set()
    # for phenotypes in result_df['Phenotype'].str.split('; '):
    #     if isinstance(phenotypes, list):
    #         all_phenotypes.update(phenotypes)
    
    # for phenotype in sorted(all_phenotypes):
    #     count = sum(result_df['Phenotype'].str.contains(phenotype, na=False))
    #     print(f"{phenotype}: {count}")
    
    # Print statistics about matched SNPs
    matched_snps_additional = sum(result_df['P-value'] != 'NA')
    matched_snps_position = sum(result_df['Position'] != 'NA')
    print(f"\nSNPs matched with additional info: {matched_snps_additional} out of {len(result_df)}")
    print(f"SNPs matched with position info: {matched_snps_position} out of {len(result_df)}")


# Usage
if __name__ == "__main__":
    input_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/Significant_SNPS/sig_snps_t2d_all.csv"
    pval_file = "/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new/t2d/all_snps_chr_merged_updated.csv"
    bim_file = "/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d.bim"
    highlighted_output = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/Significant_SNPS/sig_snps_t2d_methods.xlsx"
    multiple_methods_output = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/Significant_SNPS/sig_snps_t2d_methods_common.xlsx"
    analyze_snp_data(input_file, pval_file, bim_file, highlighted_output, multiple_methods_output)