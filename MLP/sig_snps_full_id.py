import csv
import pandas as pd

def update_snp_data(significant_snps_file, filtered_bim_file, output_file):
    # Read the significant SNPs file
    df_snps = pd.read_csv(significant_snps_file)
    
    # Read the filtered BIM file
    df_bim = pd.read_csv(filtered_bim_file, sep=r'\s+', header=None, 
                         names=['chromosome', 'snp_id', 'bp', 'ref', 'alt'])
    
     # Create a dictionary to store chromosome, SNP ID, and index information
    snp_info = {}
    for index, row in df_bim.iterrows():
        snp_info[index] = {
            'chromosome': row['chromosome'],
            'snp_id': row['snp_id'],
            'bim_index': index
        }
    
    # # Add new columns to the significant SNPs dataframe for MLP
    # df_snps['chromosome'] = df_snps['SNP_ID'].map(lambda x: snp_info.get(x, {}).get('chromosome', 'N/A'))
    # df_snps['bim_snp_id'] = df_snps['SNP_ID'].map(lambda x: snp_info.get(x, {}).get('snp_id', 'N/A'))
    # df_snps['bim_index'] = df_snps['SNP_ID'].map(lambda x: snp_info.get(x, {}).get('bim_index', 'N/A'))

    # Add new columns to the significant SNPs dataframe for CNN skip
    df_snps['chromosome'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('chromosome', 'N/A'))
    df_snps['bim_snp_id'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('snp_id', 'N/A'))
    df_snps['bim_index']  = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('bim_index', 'N/A'))
    
    # Save the updated dataframe to a new CSV file
    df_snps.to_csv(output_file, index=False)
    print(f"Updated data saved to {output_file}")


if __name__ == "__main__":

    significant_snps_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.1/residual/0_best_imp_01_41_32_0.5_100_0.001_gelu_warmup_exponential_0.1_adamw_0.5/feature_importance_class_1.csv"
    filtered_bim_file = "/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d_0.1.gen"
    #filtered_bim_file = "/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d.bim"
    output_file =  "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.1/residual/0_best_imp_01_41_32_0.5_100_0.001_gelu_warmup_exponential_0.1_adamw_0.5/updated_significant_snps_t2d_cnn_skip_0.1.csv"
    
  
    update_snp_data(significant_snps_file, filtered_bim_file, output_file)




