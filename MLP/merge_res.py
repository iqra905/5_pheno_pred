import os
import pandas as pd
from collections import OrderedDict

def merge_summary_csv(directory):
    # Initialize an ordered dictionary to store DataFrames
    dfs = OrderedDict()
    experiment_folders = sorted(os.listdir(directory))

    # Iterate over each folder in the directory
    for experiment_folder in experiment_folders:
        # Check if the item in the directory is a folder
        if os.path.isdir(os.path.join(directory, experiment_folder)):
            # Path to the summary.csv file in the experiment folder
            summary_file = os.path.join(directory, experiment_folder, 'experiment_results.csv')

            # Check if the summary.csv file exists
            if os.path.exists(summary_file):
                # Read the summary.csv file into a DataFrame and add it to the dictionary
                df = pd.read_csv(summary_file)
                dfs[experiment_folder] = df

    # Concatenate all DataFrames in the dictionary in the order they were inserted
    merged_data = pd.concat(dfs.values(), ignore_index=True)
    # Sort the merged DataFrame based on the "AUC Test" column in decreasing order
    #merged_data.sort_values(by="test_auc", ascending=False, inplace=True)

    # Write the merged data to a single CSV file
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/col/skip_conn/results_cnn_exp_skip.csv', index=False)
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/col/pretrained/01_BS_32/with_avg_pooling/results_cnn_exp_trans_col_pool.csv', index=False)
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/col/transformer/scratch/01/results_cnn_exp_trans_scratch.csv', index=False)
    
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_chr_wise/chr_wise/results_mlp_chr_exp_chr_wise.csv', index=False)

    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_full/full_covariates/Last_layer/exp_cov/HS_2/results_mlp_exp_cov_last_layer_cov_col.csv', index=False)
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pca/results_mlp_exp_cov_last_layer_pca_col.csv', index=False)
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/results_mlp_exp_cov_last_layer_pca_col_pruned.csv', index=False)
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_wo_cov/results_mlp_exp_cov_last_layer_wo_cov.csv', index=False)
    #merged_data.to_csv('/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned_overlap_analysis/exp_mlp/results_mlp_chr_wise_pruned_brea_0.1.csv', index=False)
    #merged_data.to_csv('/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned_overlap_analysis_brea_full_80_20/exp_mlp/results_mlp_pruned_brea_0.1_overlap_full_80_20.csv', index=False)
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_5M_stratified_kfold/brea/results_mlp_pruned_brea_0.1_5_fold.csv', index=False)
    merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/m4/with_cov_2/results_cnn_multilabel_m4_cov_2.csv', index=False)
    #merged_data.to_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_AIiH_2025/disease_wise_lr_split_100/cnn/0.1/results_cnn_lr_split_100_0.1.csv', index=False)






    print("Merged summary CSV file sorted and created successfully.")

#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/col/skip_conn'
#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/col/pretrained/01_BS_32/with_avg_pooling'
#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/col/transformer/scratch/01'

#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_full/full_covariates/Last_layer/exp_cov/HS_2'
#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pca'
#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca'
#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_wo_cov'
#directory_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned_overlap_analysis/exp_mlp'
#directory_path = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned_overlap_analysis_brea_full_80_20/exp_mlp'
#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_5M_stratified_kfold/brea' 
directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/m4/with_cov_2'
#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_AIiH_2025/disease_wise_lr_split_100/cnn/0.1' 





#directory_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_0.5/chr_wise'



# Call the function to merge and sort summary.csv files
merge_summary_csv(directory_path)