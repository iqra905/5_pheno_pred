import csv
import pandas as pd

def update_snp_data(significant_snps_file, filtered_bim_file, output_file):
    # Read the significant SNPs file
    df_snps = pd.read_csv(significant_snps_file)
    
    # Read the filtered BIM file
    df_bim = pd.read_csv(filtered_bim_file, sep=r'\s+', header=None, 
                         names=['chromosome', 'snp_id', 'distance', 'bp', 'ref', 'alt'])
    
    # Create a dictionary to store chromosome and SNP ID information
    snp_info = {}
    for index, row in df_bim.iterrows():
        snp_info[index] = {'chromosome': row['chromosome'], 'snp_id': row['snp_id']}
    
    # Add new columns to the significant SNPs dataframe
    df_snps['chromosome'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('chromosome', 'N/A'))
    df_snps['snp_id'] = df_snps['SNP_Index'].map(lambda x: snp_info.get(x, {}).get('snp_id', 'N/A'))
    
    # Save the updated dataframe to a new CSV file
    df_snps.to_csv(output_file, index=False)
    print(f"Updated data saved to {output_file}")

if __name__ == "__main__":
    # significant_snps_file = "/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new/bmi/sig_snps_bmi.csv"
    # filtered_bim_file = "/vol/research/fmodal_mmmed/Codes/Datasets/BMI/filtered_plinkfiles/filtered_plink.bim"
    # output_file = "/vol/research/fmodal_mmmed/Codes/stat_analysis_lr/results_new/bmi/updated_significant_snps.csv"

    # significant_snps_file = "/vol/research/fmodal_mmmed/Codes/GenNet/results/GenNet_experiment_6_6_batch_size_32_epochs_70_patience_30_lr_0.0001_l1_0.003_/top_46_indices_gennet.csv"
    # filtered_bim_file = "/vol/research/fmodal_mmmed/Codes/Datasets/BMI/filtered_plinkfiles/filtered_plink.bim"
    # output_file = "/vol/research/fmodal_mmmed/Codes/GenNet/results/GenNet_experiment_6_6_batch_size_32_epochs_70_patience_30_lr_0.0001_l1_0.003_/updated_significant_snps.csv"
    

    # significant_snps_file = "/vol/research/fmodal_mmmed/Codes/GenNet_MLP/results/results_bmi_final_chi_processed/exp_final_preprocessed_61_2/top50_indices.csv"
    # filtered_bim_file = "/vol/research/fmodal_mmmed/Codes/Datasets/BMI/preprocessed_chi/filtered_plink_processed.bim"
    # output_file = "/vol/research/fmodal_mmmed/Codes//GenNet_MLP/results/results_bmi_final_chi_processed/exp_final_preprocessed_61_2/updated_significant_snps.csv"
    

    # significant_snps_file = "/vol/research/fmodal_mmmed/Codes/GenNet_conv/results/results_cnn_exp_final_new/15/exp_final/top_45_indices.csv"
    # filtered_bim_file = "/vol/research/fmodal_mmmed/Codes/Datasets/BMI/filtered_plinkfiles/filtered_plink.bim"
    # output_file = "/vol/research/fmodal_mmmed/Codes/GenNet_conv/results/results_cnn_exp_final_new/15/exp_final/updated_significant_snps.csv"

    # significant_snps_file = "/vol/research/fmodal_mmmed/Codes/GenNet_conv/results/results_cnn_exp_final_new/chi_processed/best_exp_01_df_0.3_dp_0.7_lr_0.0001_0.00001_hs_64_wd_0.08/top_45_indices.csv"
    # filtered_bim_file = "/vol/research/fmodal_mmmed/Codes/Datasets/BMI/preprocessed_chi/filtered_plink_processed.bim"
    # output_file = "/vol/research/fmodal_mmmed/Codes/GenNet_conv/results/results_cnn_exp_final_new/chi_processed/best_exp_01_df_0.3_dp_0.7_lr_0.0001_0.00001_hs_64_wd_0.08/updated_significant_snps.csv"
    
    # significant_snps_file = "/vol/research/fmodal_mmmed/Codes/DeepCombi/tests/results/BMI/chromosome_22/top_indices.csv"
    # filtered_bim_file = "/vol/research/fmodal_mmmed/Codes/DeepCombi/data/BMI/filtered_plink_chr22_updated_alleles.bim"
    # output_file = "/vol/research/fmodal_mmmed/Codes/DeepCombi/tests/results/BMI/chromosome_22/updated_significant_snps.csv"

    significant_snps_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.1/full/0_best_imp_01_36_32_0.5_100_0.005_relu_exponential_decay_0.1_adamw_0.5/top_66713_indices.csv"
    filtered_bim_file = "/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d_0.1.gen"
    output_file = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.1/full/0_best_imp_01_36_32_0.5_100_0.005_relu_exponential_decay_0.1_adamw_0.5/updated_significant_snps.csv"

    # significant_snps_file = "/vol/research/fmodal_mmmed/Codes/GenNet_MLP/results/results_bmi_final_chr/best_153_128_120_0.0005_sigmoid_0.001_AdamW_ExponentialDecay/significant_snps.csv"
    # filtered_bim_file = "/vol/research/fmodal_mmmed/Codes/Datasets/BMI/filtered_plinkfiles/filtered_plink.bim"
    # output_file = "/vol/research/fmodal_mmmed/Codes/GenNet_MLP/results/results_bmi_final_lr_0.05/best_236_128_200_3_AdamW_ExponentialDecay_sigmoid_0.001/updated_significant_snps.csv"




    
    update_snp_data(significant_snps_file, filtered_bim_file, output_file)