#********************* Experiment Plots Models and with different Hyperparameters *****************************#
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Read the CSV file
#df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/01/results_mlp_exp.csv')
df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/transformer/scratch/01/results_cnn_exp_trans_scratch.csv')

#df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/skip_conn/results_cnn_exp_skip.csv')

#df = pd.read_csv('/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/02/results_cnn_exp_t2d.csv')

# Create a composite Model Architecture string
#df['Model_Architecture'] = df.apply(lambda row: f"HS:{row['hidden_sizes']}", axis=1)
#df['Model_Architecture'] = df.apply(lambda row: f"K:{row['Kernel_sizes']}, S:{row['Stride']}, OC:{row['conv_channels']}, \nAct:{row['Act']}, Sch:{row['Sch']}", axis=1)

df['Model_Architecture'] = df.apply(lambda row: f"K:{row['Kernel_sizes']}, S:{row['Stride']}, \n OC:{row['conv_channels']}, FC:{row['fc_layers']}", axis=1)


# Extract part of the Job ID for y-axis label
df['Exp_Label'] = df['Exp_ID'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Columns for main grouping
#main_columns = ['Act','Sch']
main_columns = ['Dropout', 'WD','Start_LR']


# Columns for subplots
#subplot_columns = ['Start_LR','Opt']
#subplot_columns = ['layer_indices']

subplot_columns = ['Act','Sch']


# Function to create subplot title
def create_subplot_title(row):
    #return f"Start_LR: {row['Start_LR']},Opt: {row['Opt']}"
    #return f"Act: {row['Act']}, Sch: {row['Sch']}, layer_indices: {row['layer_indices']}"
    return f"Act: {row['Act']}, Sch: {row['Sch']}"


# Get unique combinations of main columns
unique_configs = df[main_columns].drop_duplicates()

# Maximum number of subplots per figure
max_subplots_per_figure = 9

# Iterate over unique configurations
for _, config in unique_configs.iterrows():
    # Filter data for current main configuration
    config_df = df
    for col in main_columns:
        config_df = config_df[config_df[col] == config[col]]
    
    # Get unique combinations of subplot columns for this configuration
    subplot_configs = config_df[subplot_columns].drop_duplicates()
    
    # Calculate number of figures needed
    n_subplots = len(subplot_configs)
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
        fig.suptitle(f"AUC Test for different Model Architectures with transformer layers from scratch \n BS: 128, Epochs: 100, Dropout: 0.5, WD: 0.5 \n maxpool + cls token + Avg pool", fontsize=14)
        #fig.suptitle(f"AUC Test for different Model Architectures \n BS: 32, Epochs: 100, Optimizer: AdamW\nStart_LR: {config['Start_LR']}, Dropout: {config['Dropout']}, WD: {config['WD']}", fontsize=14)

        # fig.suptitle(f"LR Sch: {config['LR Sch']}, Dropout: {config['Dropout']}\n"
        #              f"WD: {config['WD']}, Cov: {config['Cov']}\n"
        #              f"(Figure {fig_num+1}/{n_figures})", fontsize=12)

        # Iterate over subplot configurations
        for i, (_, subplot_config) in enumerate(subplot_configs.iloc[start_idx:end_idx].iterrows()):
            ax = axs[i // n_cols, i % n_cols]
            
            # Filter data for current subplot configuration
            # subplot_df = config_df[(config_df['Start_LR'] == subplot_config['Start_LR']) &
            #                     (config_df['Opt'] == subplot_config['Opt'])]

            subplot_df = config_df[(config_df['Act'] == subplot_config['Act']) & 
                                   (config_df['Sch'] == subplot_config['Sch'])]
            
            # Sort the dataframe by test_auc in descending order
            subplot_df = subplot_df.sort_values('test_auc', ascending=False)
            
            # Create bar plot for filtered data
            sns.barplot(x='test_auc', y='Exp_Label', data=subplot_df, ax=ax, orient='h')
            
            # Customize subplot
            ax.set_xlim(0, 1)  # Assuming AUC is between 0 and 1
            ax.set_title(create_subplot_title(subplot_config), fontsize=10)
            ax.set_xlabel('test_auc')
            ax.set_ylabel('Job Label')
            
            # Add value labels and model architecture inside bars
            for j, (v, arch) in enumerate(zip(subplot_df['test_auc'], subplot_df['Model_Architecture'])):
                ax.text(v, j, f'{v:.3f}', ha='left', va='center', fontsize=10)
                ax.text(0.01, j, arch, ha='left', va='center', fontsize=10, color='white')

        # Remove empty subplots
        for j in range(i+1, n_rows*n_cols):
            fig.delaxes(axs[j // n_cols, j % n_cols])

        # Adjust layout and save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/transformer/scratch/01/plots/experiment_results_Dropout_{config["Dropout"]}_WD_{config["WD"]}_LR_{config["Start_LR"]}_{fig_num+1}.png', dpi=300, bbox_inches='tight')
        #plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/02/experiment_results_Dropout_{config["Dropout"]}_WD_{config["WD"]}_{fig_num+1}.png', dpi=300, bbox_inches='tight')
        #plt.savefig(f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/01/plots/experiment_results_{config["Act"]}_{config["Sch"]}_{fig_num+1}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

print("All plots have been generated and saved.")



