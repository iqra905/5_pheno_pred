import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Read the CSV file
df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/pros_full/full_covariates/Separate_layers/exp_cov_norm/results_mlp_exp_pros_cov_sep_layer_cov_norm.csv')

# Create a composite Model Architecture string
df['Model_Architecture'] = df.apply(lambda row: f"LR: {row['Start_LR']} - H:{row['hidden_sizes']} - {row['Epochs']}", axis=1)

# Extract part of the Job ID for y-axis label
#df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[10:]))
df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[:2] + x.split('_')[10:]))


# Create color condition
# T2D
#df['high_performance'] = (df['test_auc'] >= 0.60) & (df['test_acc'] > 0.58)

# PROS
df['high_performance'] = (df['test_auc'] >= 0.80) & (df['test_acc'] >= 0.80)

# Columns for main grouping
main_columns = ['Use_PCs', 'Use_Age', 'Use_Gender']

# Get unique combinations of main columns
unique_configs = df[main_columns].drop_duplicates()

# Maximum number of subplots per figure
max_subplots_per_figure = 12

# Iterate over unique configurations
for _, config in unique_configs.iterrows():
    # Filter data for current main configuration
    config_df = df
    for col in main_columns:
        config_df = config_df[config_df[col] == config[col]]
    
    # Get unique values for Sch and Act
    unique_sch = sorted(config_df['Sch'].unique())
    unique_act = sorted(config_df['Act'].unique())
    
    # Calculate total number of subplots
    n_subplots = len(unique_sch) * len(unique_act)
    n_figures = math.ceil(n_subplots / max_subplots_per_figure)

    for fig_num in range(n_figures):
        # Calculate number of rows and columns for subplots
        start_idx = fig_num * max_subplots_per_figure
        end_idx = min((fig_num + 1) * max_subplots_per_figure, n_subplots)
        n_subplots_current = end_idx - start_idx
        n_cols = min(4, n_subplots_current)
        n_rows = (n_subplots_current - 1) // n_cols + 1

        # Create figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10*n_rows), squeeze=False)
        fig.suptitle(f"Performance Metrics for different Experiments (Modelling covariates with separate layer) with \n Use_PCs: {config['Use_PCs']}, Use_Age: {config['Use_Age']}, 'Use_Gender: {config['Use_Gender']}' , Dropout: 0.5, WD: 0.5, Opt: AdamW", fontsize=14)

        subplot_idx = 0
        for act in unique_act:
            for sch in unique_sch:
                if subplot_idx >= start_idx and subplot_idx < end_idx:
                    current_idx = subplot_idx - start_idx
                    ax = axs[current_idx // n_cols, current_idx % n_cols]
                    
                    # Filter data for current subplot configuration
                    subplot_df = config_df[(config_df['Act'] == act) & (config_df['Sch'] == sch)]
                    
                    # Sort the dataframe
                    subplot_df = subplot_df.sort_values(
                        by=['Start_LR', 'hidden_sizes', 'Epochs'],
                        ascending=[True, True, True]
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
                    ax.set_title(f"Act: {act}, Sch: {sch}", fontsize=12, weight='bold')
                    ax.set_xlabel('Metric Value')
                    ax.set_ylabel('Job Label')
                    ax.set_yticks(range(len(subplot_df)))
                    ax.set_yticklabels(subplot_df['Exp_Label'],fontsize=8)
                    
                    # Add legend if it's the first subplot
                    if subplot_idx == start_idx:
                        ax.legend()
                
                subplot_idx += 1

        # Remove empty subplots
        for j in range(current_idx + 1, n_rows * n_cols):
            fig.delaxes(axs[j // n_cols, j % n_cols])

        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0.03, 0.90, 0.95])
        plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/pros_full/full_covariates/Separate_layers/exp_cov_norm/plots/experiment_results_Use_PCs_{config["Use_PCs"]}_Use_Age_{config["Use_Age"]}_Use_Gender_{config["Use_Gender"]}_{fig_num+1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

print("All plots have been generated and saved.")

