#**************************** Plots with 2 varying parameters*************
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')  # Use the 'Agg' backend
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math

# # Read the CSV file
# df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/t2d/t2d_pca/results_mlp_exp_cov_last_layer_pca_t2d.csv')
# #df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/t2d/t2d_pruned_pca/results_mlp_exp_cov_last_layer_pca_t2d_pruned.csv')


# # Create a composite Model Architecture string
# df['Model_Architecture'] = df.apply(lambda row: f"{row['Start_LR']} - H:{row['hidden_sizes']}", axis=1)

# # Extract part of the Job ID for y-axis label
# df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[:2]))

# # Create color condition

# # pros
# #df['high_performance'] = (df['test_auc'] >= 0.58) & (df['test_acc'] >= 0.49)

# # col
# #df['high_performance'] = (df['test_auc'] >= 0.59) & (df['test_acc'] > 0.52)

# # COL
# #df['high_performance'] = (df['test_auc'] > 0.54) & (df['test_acc'] >0.51)

# # brea
# #df['high_performance'] = (df['test_auc'] > 0.52) & (df['test_acc'] > 0.51)

# # T2D
# df['high_performance'] = (df['test_auc'] > 0.56) & (df['test_acc'] > 0.55)

# # Columns for main grouping
# main_columns = ['Act']

# # Get unique combinations of main columns
# unique_configs = df[main_columns].drop_duplicates()

# # Maximum number of subplots per figure
# max_subplots_per_figure = 20

# # Iterate over unique configurations
# for _, config in unique_configs.iterrows():
#     # Filter data for current main configuration
#     config_df = df
#     for col in main_columns:
#         config_df = config_df[config_df[col] == config[col]]
    
#     # Get unique values for Sch and Act
#     unique_sch = sorted(config_df['Sch'].unique())
#     unique_act = sorted(config_df['WD'].unique())
    
#     # Calculate total number of subplots
#     n_subplots = len(unique_sch) * len(unique_act)
#     n_figures = math.ceil(n_subplots / max_subplots_per_figure)

#     for fig_num in range(n_figures):
#         # Calculate number of rows and columns for subplots
#         start_idx = fig_num * max_subplots_per_figure
#         end_idx = min((fig_num + 1) * max_subplots_per_figure, n_subplots)
#         n_subplots_current = end_idx - start_idx
#         n_cols = min(5, n_subplots_current)
#         n_rows = (n_subplots_current - 1) // n_cols + 1

#         # Create figure and subplots
#         fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows), squeeze=False)
#         fig.suptitle(f"Performance Metrics for different Experiments with PCA features \n BS: 64, Dropout: 0.5, Opt: AdamW, Epochs: 100, Act: {config['Act']}\n T2D", fontsize=14)
#         #fig.suptitle(f"Performance Metrics for different Experiments (Concatenating Covariates at Last Hidden Layer) with \n BS: {config['BS']}, Dropout: 0.5, WD: 0.5, Opt: AdamW", fontsize=14)


#         subplot_idx = 0
#         for act in unique_act:
#             for sch in unique_sch:
#                 if subplot_idx >= start_idx and subplot_idx < end_idx:
#                     current_idx = subplot_idx - start_idx
#                     ax = axs[current_idx // n_cols, current_idx % n_cols]
                    
#                     # Filter data for current subplot configuration
#                     subplot_df = config_df[(config_df['WD'] == act) & (config_df['Sch'] == sch)]
                    
#                     # Sort the dataframe
#                     subplot_df = subplot_df.sort_values(
#                         by=['Start_LR', 'hidden_sizes', 'WD'],
#                         ascending=[True, True, True]
#                     )

#                     # Calculate y-positions for the bars
#                     n_experiments = len(subplot_df)
#                     y_positions = range(n_experiments)
#                     bar_height = 0.35

#                     # Plot AUC bars
#                     for i, (idx, row) in enumerate(subplot_df.iterrows()):
#                         color = 'blue' if row['high_performance'] else 'green'
#                         ax.barh(i - bar_height/2, row['test_auc'], height=bar_height, 
#                                color=color, label='AUC' if i == 0 else "")
                        
#                         # Add AUC value label
#                         ax.text(row['test_auc'], i - bar_height/2, 
#                                f'{row["test_auc"]:.3f}', 
#                                ha='left', va='center', fontsize=8)

#                     # Plot ACC bars
#                     for i, (idx, row) in enumerate(subplot_df.iterrows()):
#                         ax.barh(i + bar_height/2, row['test_acc'], height=bar_height, 
#                                color='red', alpha=0.6, label='ACC' if i == 0 else "")
                        
#                         # Add ACC value label
#                         ax.text(row['test_acc'], i + bar_height/2, 
#                                f'{row["test_acc"]:.3f}', 
#                                ha='left', va='center', fontsize=8)

#                         # Add model architecture
#                         ax.text(0.01, i, row['Model_Architecture'], 
#                                ha='left', va='center', fontsize=8, color='white', weight='bold')
                    
#                     # Customize subplot
#                     ax.set_xlim(0, 1)
#                     ax.set_title(f"WD: {act}, Start_LR: {sch}", fontsize=12, weight='bold')
#                     ax.set_xlabel('Metric Value')
#                     ax.set_ylabel('Job Label')
#                     ax.set_yticks(range(len(subplot_df)))
#                     ax.set_yticklabels(subplot_df['Exp_Label'])
                    
#                     # Add legend if it's the first subplot
#                     if subplot_idx == start_idx:
#                         ax.legend()
                
#                 subplot_idx += 1

#         # Remove empty subplots
#         for j in range(current_idx + 1, n_rows * n_cols):
#             fig.delaxes(axs[j // n_cols, j % n_cols])

#         # Adjust layout and save figure
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/t2d/t2d_pca/plots/experiment_results_{config["Act"]}_{fig_num+1}_t2d.png', dpi=300, bbox_inches='tight')
#         #plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/t2d/t2d_pruned_pca/plots/experiment_results_{config["Act"]}_{fig_num+1}_t2d_pruned.png',dpi=300, bbox_inches='tight')
#         plt.close(fig)

# print("All plots have been generated and saved.")


#**************************** Plots with Single varying parameters*************
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Read the CSV file
#df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pca/results_mlp_exp_col_cov_last_layer_pca.csv')
df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_stats_info/results_mlp_exp_cov_last_layer_stats_info_5d.csv')


# Create a composite Model Architecture string
df['Model_Architecture'] = df.apply(lambda row: f"{row['Label_Column']} - {row['Start_LR']} - H:{row['hidden_sizes']} - WD: {row['WD']} ", axis=1)

# Extract part of the Job ID for y-axis label
#df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[:2] + x.split('_')[4:]))
df['Exp_Label'] = df['Exp_ID']


# Create color condition
# col
#df['high_performance'] = (df['test_auc'] >= 0.60) & (df['test_acc'] > 0.58)

# col
#df['high_performance'] = (df['test_auc'] >= 0.80) & (df['test_acc'] >= 0.80)
df['high_performance'] = (df['test_auc'] >= 0.90) & (df['test_acc'] >= 0.90)

# Columns for main grouping
main_columns = ['Act']

# Get unique combinations of main columns
unique_configs = df[main_columns].drop_duplicates()

# Maximum number of subplots per figure
max_subplots_per_figure = 6

# Iterate over unique configurations
for _, config in unique_configs.iterrows():
    # Filter data for current main configuration
    config_df = df
    for col in main_columns:
        config_df = config_df[config_df[col] == config[col]]
    
    # Get unique values for Sch and Act
    unique_sch = sorted(config_df['model_type'].unique())

    # Calculate total number of subplots
    n_subplots = len(unique_sch) 
    n_figures = math.ceil(n_subplots / max_subplots_per_figure)

    for fig_num in range(n_figures):
        # Calculate number of rows and columns for subplots
        start_idx = fig_num * max_subplots_per_figure
        end_idx = min((fig_num + 1) * max_subplots_per_figure, n_subplots)
        n_subplots_current = end_idx - start_idx
        n_cols = min(3, n_subplots_current)
        n_rows = (n_subplots_current - 1) // n_cols + 1

        # Create figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows), squeeze=False)
        fig.suptitle(f"Performance Metrics for different Experiments \n Using Country-Code with max samples for Training and 2nd max for Testing \n BS: 32, Dropout: 0.5, Opt: AdamW, Sch: Exponential Decay, Act:GELU", fontsize=14)
        #fig.suptitle(f"Performance Metrics for different Experiments (Concatenating Covariates at Last Hidden Layer) with \n BS: {config['BS']}, Dropout: 0.5, WD: 0.5, Opt: AdamW", fontsize=14)


        subplot_idx = 0
        for sch in unique_sch:
            if subplot_idx >= start_idx and subplot_idx < end_idx:
                current_idx = subplot_idx - start_idx
                ax = axs[current_idx // n_cols, current_idx % n_cols]
                
                # Filter data for current subplot configuration
                subplot_df = config_df[(config_df['model_type'] == sch)]
                
                # Sort the dataframe
                subplot_df = subplot_df.sort_values(
                    by=['Label_Column','Start_LR','WD','hidden_sizes'],
                    ascending=[True, True, True, True]
                )

                # Calculate y-positions for the bars
                n_experiments = len(subplot_df)
                y_positions = range(n_experiments)
                bar_height = 0.35

                # Plot AUC bars
                for i, (idx, row) in enumerate(subplot_df.iterrows()):
                    color = 'blue' if row['high_performance'] else 'green'
                    ax.barh(i - bar_height/2, row['test_auc'], height=bar_height, 
                            color=color, label='AUC' if i == 0 else "")
                    
                    # Add AUC value label
                    ax.text(row['test_auc'], i - bar_height/2, 
                            f'{row["test_auc"]:.3f}', 
                            ha='left', va='center', fontsize=8)

                # Plot ACC bars
                for i, (idx, row) in enumerate(subplot_df.iterrows()):
                    ax.barh(i + bar_height/2, row['test_acc'], height=bar_height, 
                            color='red', alpha=0.6, label='ACC' if i == 0 else "")
                    
                    # Add ACC value label
                    ax.text(row['test_acc'], i + bar_height/2, 
                            f'{row["test_acc"]:.3f}', 
                            ha='left', va='center', fontsize=8)

                    # Add model architecture
                    ax.text(0.01, i, row['Model_Architecture'], 
                            ha='left', va='center', fontsize=8, color='white', weight='bold')
                
                # Customize subplot
                ax.set_xlim(0, 1)
                ax.set_title(f"Model Type: {sch}", fontsize=12, weight='bold')
                ax.set_xlabel('Metric Value')
                ax.set_ylabel('Job Label')
                ax.set_yticks(range(len(subplot_df)))
                ax.set_yticklabels(subplot_df['Exp_Label'])
                
                # Add legend if it's the first subplot
                if subplot_idx == start_idx:
                    ax.legend()
            
            subplot_idx += 1

        # Remove empty subplots
        for j in range(current_idx + 1, n_rows * n_cols):
            fig.delaxes(axs[j // n_cols, j % n_cols])

        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pca/plots/experiment_results_no_cov_{config["Act"]}_{fig_num+1}.png',dpi=300, bbox_inches='tight')
        plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_exp_stats_info/experiment_results_stats_info_5d_{fig_num+1}.png', dpi=300, bbox_inches='tight')
                  
        plt.close(fig)

print("All plots have been generated and saved.")



# #********************* Experiment Plots Models and with different Hyperparameters *****************************#
# import coldas as pd
# import matplotlib
# matplotlib.use('Agg')  # Use the 'Agg' backend
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math

# # Read the CSV file
# df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_full/full_covariates/Last_layer/exp_no_cov_wd_lr/results_mlp_exp_col_cov_Last_layer_no_cov.csv')

# # Create a composite Model Architecture string
# df['Model_Architecture'] = df.apply(lambda row: f"H:{row['hidden_sizes']} -- {row['Epochs']} -- {row['Start_LR']}", axis=1)

# # Extract part of the Job ID for y-axis label
# df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[:2]))

# # Columns for main grouping
# main_columns = ['Sch']

# # Columns for subplots
# subplot_columns = ['Start_LR']
# #subplot_columns = ['Start_LR','WD']


# # Function to create subplot title
# def create_subplot_title(row):
#     return f"Start_LR: {row['Start_LR']}"
#     #return f"Start_LR: {row['Start_LR']}, WD: {row['WD']}"


# # Get unique combinations of main columns
# unique_configs = df[main_columns].drop_duplicates()

# # Maximum number of subplots per figure
# max_subplots_per_figure = 12

# # Iterate over unique configurations
# for _, config in unique_configs.iterrows():
#     # Filter data for current main configuration
#     config_df = df
#     for col in main_columns:
#         config_df = config_df[config_df[col] == config[col]]
    
#     # Get unique combinations of subplot columns for this configuration
#     subplot_configs = config_df[subplot_columns].drop_duplicates()
    
#     # Calculate number of figures needed
#     n_subplots = len(subplot_configs)
#     n_figures = math.ceil(n_subplots / max_subplots_per_figure)

#     for fig_num in range(n_figures):
#         # Calculate number of rows and columns for subplots
#         start_idx = fig_num * max_subplots_per_figure
#         end_idx = min((fig_num + 1) * max_subplots_per_figure, n_subplots)
#         n_subplots_current = end_idx - start_idx
#         n_cols = min(3, n_subplots_current)
#         n_rows = (n_subplots_current - 1) // n_cols + 1

        

#         # Create figure and subplots
#         fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows), squeeze=False)
#         fig.suptitle(f"AUC Test for different Experiments with \n BS: {config['Sch']}, Dropout: 0.5, WD: 0.5, Opt: AdamW", fontsize=14)

#         # fig.suptitle(f"LR Sch: {config['LR Sch']}, Dropout: {config['Dropout']}\n"
#         #              f"WD: {config['WD']}, Cov: {config['Cov']}\n"
#         #              f"(Figure {fig_num+1}/{n_figures})", fontsize=12)

#         # Iterate over subplot configurations
#         for i, (_, subplot_config) in enumerate(subplot_configs.iloc[start_idx:end_idx].iterrows()):
#             ax = axs[i // n_cols, i % n_cols]
            
#             # Filter data for current subplot configuration
#             # subplot_df = config_df[(config_df['Start_LR'] == subplot_config['Start_LR']) &
#             #                     (config_df['Opt'] == subplot_config['Opt'])]

#             subplot_df = config_df[(config_df['Start_LR'] == subplot_config['Start_LR'])]
#             #subplot_df = config_df[(config_df['Start_LR'] == subplot_config['Start_LR']) & (config_df['WD'] == subplot_config['WD'])]
            
#             # Sort the dataframe by test_auc in descending order
#             subplot_df = subplot_df.sort_values('test_auc', ascending=False)
            
#             # Create bar plot for filtered data
#             sns.barplot(x='test_auc', y='Exp_Label', data=subplot_df, ax=ax, orient='h')
            
#             # Customize subplot
#             ax.set_xlim(0, 1)  # Assuming AUC is between 0 and 1
#             ax.set_title(create_subplot_title(subplot_config), fontsize=10)
#             ax.set_xlabel('test_auc')
#             ax.set_ylabel('Job Label')
            
#             # Add value labels and model architecture inside bars
#             for j, (v, arch) in enumerate(zip(subplot_df['test_auc'], subplot_df['Model_Architecture'])):
#                 ax.text(v, j, f'{v:.3f}', ha='left', va='center', fontsize=10)
#                 ax.text(0.01, j, arch, ha='left', va='center', fontsize=10, color='white')

#         # Remove empty subplots
#         for j in range(i+1, n_rows*n_cols):
#             fig.delaxes(axs[j // n_cols, j % n_cols])

#         # Adjust layout and save figure
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_full/full_covariates/Last_layer/exp_no_cov_wd_lr/plots/experiment_results_BS_{config["Sch"]}_{fig_num+1}.png', dpi=300, bbox_inches='tight')
#         plt.close(fig)

# print("All plots have been generated and saved.")

#********************* Experiment Plots with covariates combinations *****************************#
# import coldas as pd
# import matplotlib
# matplotlib.use('Agg')  # Use the 'Agg' backend
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math

# # Read the CSV file
# df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_full/full_covariates/Last_layer/exp_cov/HS_2/results_mlp_exp_cov_last_layer_cov_col.csv')

# # Create a composite Model Architecture string
# df['Model_Architecture'] = df.apply(lambda row: f"LR: {row['Start_LR']} - H:{row['hidden_sizes']} - {row['Epochs']}", axis=1)

# # Extract part of the Job ID for y-axis label
# #df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[:]))
# df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[:2] + x.split('_')[12:]))

# # Create color condition
# # col
# #df['high_performance'] = (df['test_auc'] >= 0.60) & (df['test_acc'] > 0.58)

# # col
# df['high_performance'] = (df['test_auc'] >= 0.80) | (df['test_acc'] >= 0.80)

# # Columns for main grouping
# main_columns = ['BS']

# # Get unique combinations of main columns
# unique_configs = df[main_columns].drop_duplicates()

# # Maximum number of subplots per figure
# max_subplots_per_figure = 18

# # Iterate over unique configurations
# for _, config in unique_configs.iterrows():
#     # Filter data for current main configuration
#     config_df = df
#     for col in main_columns:
#         config_df = config_df[config_df[col] == config[col]]
    
#     # Get unique values for Sch and Act
#     unique_sch = sorted(config_df['Sch'].unique())
#     unique_act = sorted(config_df['Act'].unique())
    
#     # Calculate total number of subplots
#     n_subplots = len(unique_sch) * len(unique_act)
#     n_figures = math.ceil(n_subplots / max_subplots_per_figure)

#     for fig_num in range(n_figures):
#         # Calculate number of rows and columns for subplots
#         start_idx = fig_num * max_subplots_per_figure
#         end_idx = min((fig_num + 1) * max_subplots_per_figure, n_subplots)
#         n_subplots_current = end_idx - start_idx
#         n_cols = min(5, n_subplots_current)
#         n_rows = (n_subplots_current - 1) // n_cols + 1

#         # Create figure and subplots
#         fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows), squeeze=False)
#         fig.suptitle(f"Performance Metrics for different Experiments with covariate combinations (Concatenating Covariates at Last Hidden Layer) with \n BS: 32, Dropout: 0.5, Opt: AdamW, WD: 0.5 \n COLON CANCER", fontsize=14)
#         #fig.suptitle(f"Performance Metrics for different Experiments (Concatenating Covariates at Last Hidden Layer) with \n BS: {config['BS']}, Dropout: 0.5, WD: 0.5, Opt: AdamW", fontsize=14)


#         subplot_idx = 0
#         for act in unique_act:
#             for sch in unique_sch:
#                 if subplot_idx >= start_idx and subplot_idx < end_idx:
#                     current_idx = subplot_idx - start_idx
#                     ax = axs[current_idx // n_cols, current_idx % n_cols]
                    
#                     # Filter data for current subplot configuration
#                     subplot_df = config_df[(config_df['Act'] == act) & (config_df['Sch'] == sch)]
                    
#                     # Sort the dataframe
#                     subplot_df = subplot_df.sort_values(
#                         by=['Use_PCs', 'Use_Age', 'Use_Gender'],
#                         #by=['Start_LR', 'hidden_sizes', 'Epochs'],
#                         ascending=[True, True, True]
#                     )

#                     # Calculate y-positions for the bars
#                     n_experiments = len(subplot_df)
#                     y_positions = range(n_experiments)
#                     bar_height = 0.35

#                     # Plot AUC bars
#                     for i, (idx, row) in enumerate(subplot_df.iterrows()):
#                         color = 'blue' if row['high_performance'] else 'green'
#                         ax.barh(i - bar_height/2, row['test_auc'], height=bar_height, 
#                                color=color, label='AUC' if i == 0 else "")
                        
#                         # Add AUC value label
#                         ax.text(row['test_auc'], i - bar_height/2, 
#                                f'{row["test_auc"]:.3f}', 
#                                ha='left', va='center', fontsize=8)

#                     # Plot ACC bars
#                     for i, (idx, row) in enumerate(subplot_df.iterrows()):
#                         ax.barh(i + bar_height/2, row['test_acc'], height=bar_height, 
#                                color='red', alpha=0.6, label='ACC' if i == 0 else "")
                        
#                         # Add ACC value label
#                         ax.text(row['test_acc'], i + bar_height/2, 
#                                f'{row["test_acc"]:.3f}', 
#                                ha='left', va='center', fontsize=8)

#                         # Add model architecture
#                         ax.text(0.01, i, row['Model_Architecture'], 
#                                ha='left', va='center', fontsize=8, color='white', weight='bold')
                    
#                     # Customize subplot
#                     ax.set_xlim(0, 1)
#                     ax.set_title(f"Act: {act}, Sch: {sch}", fontsize=12, weight='bold')
#                     ax.set_xlabel('Metric Value')
#                     ax.set_ylabel('Job Label')
#                     ax.set_yticks(range(len(subplot_df)))
#                     ax.set_yticklabels(subplot_df['Exp_Label'])
                    
#                     # Add legend if it's the first subplot
#                     if subplot_idx == start_idx:
#                         ax.legend()
                
#                 subplot_idx += 1

#         # Remove empty subplots
#         for j in range(current_idx + 1, n_rows * n_cols):
#             fig.delaxes(axs[j // n_cols, j % n_cols])

#         # Adjust layout and save figure
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_full/full_covariates/Last_layer/exp_cov/HS_2/plots/experiment_results_COL_BS_{config["BS"]}_{fig_num+1}.png', 
#                    dpi=300, bbox_inches='tight')
#         plt.close(fig)

# print("All plots have been generated and saved.")


