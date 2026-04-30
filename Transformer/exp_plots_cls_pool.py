#****************************** Subplots with layer wise subplots for all csv combined **********************#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re

def find_csv_files(directory):
    files = glob.glob(os.path.join(directory, '*', '*.csv'), recursive=True)
    return sorted(files, key=lambda x: x.split('results_cnn_exp')[-1])[:2]  # Limit to first two files

def get_short_job_id(job_id):
    return '_'.join(str(job_id).split('_')[1:2])

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

sns.set_style("whitegrid")

base_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/transformer/pretrained/with_transformer/02_BS_32_2/'
csv_files = find_csv_files(base_dir)
output_dir = os.path.join(base_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

if len(csv_files) != 2:
    raise ValueError("Exactly two CSV files are required for comparison.")

dfs = []
for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if 'Exp_ID' in df.columns:
        df['Short Jobid'] = df['Exp_ID'].apply(get_short_job_id)
    else:
        print(f"Warning: 'Exp_ID' column not found in {csv_file}. Using index as Short Jobid.")
        df['Short Jobid'] = df.index.astype(str)

    required_columns = ['test_auc', 'layer_indices', 'Sch', 'Final_LR', 'Act', 'Kernel_sizes', 'Stride', 'conv_channels']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in {csv_file}: {', '.join(missing_columns)}")
        continue

    df['sort_key'] = df['Short Jobid'].apply(natural_sort_key)
    df = df.sort_values('sort_key')
    df = df.drop('sort_key', axis=1)
    
    df['File'] = os.path.basename(csv_file)
    dfs.append(df)

# Merge the two dataframes
merged_df = pd.merge(dfs[0][required_columns + ['Short Jobid']], 
                     dfs[1][required_columns + ['Short Jobid']], 
                     on=['Short Jobid', 'layer_indices'], 
                     suffixes=('_1', '_2'))

# Group by layer_indices
grouped = merged_df.groupby('layer_indices')
n_groups = len(grouped)

# Set up the subplot grid
n_cols = min(3, n_groups)
n_rows = (n_groups + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15*n_cols, 8*n_rows))  # Increased figure size
if n_groups > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for (layer_index, group), ax in zip(grouped, axes):
    y = range(len(group))
    height = 0.35
    
    ax.barh([i - height/2 for i in y], group['test_auc_1'], height, label=os.path.basename(csv_files[0]), color='blue', alpha=0.7)
    ax.barh([i + height/2 for i in y], group['test_auc_2'], height, label=os.path.basename(csv_files[1]), color='red', alpha=0.7)
    
    ax.set_title(f"Layer Index: {layer_index}", fontsize=10)
    ax.set_xlabel('AUC Test', fontsize=8)
    ax.set_ylabel('Experiment ID', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(group['Short Jobid'], fontsize=6)
    
    for i, (v1, v2) in enumerate(zip(group['test_auc_1'], group['test_auc_2'])):
        ax.text(v1, i - height/2, f'{v1:.3f}', va='center', ha='left', fontsize=8, weight='bold')
        ax.text(v2, i + height/2, f'{v2:.3f}', va='center', ha='left', fontsize=8, weight='bold')
        
        # Add detailed information for file 1
        details_1 = f"Sch:{group['Sch_1'].iloc[i]}, LR:{group['Final_LR_1'].iloc[i]:.2e}, Act:{group['Act_1'].iloc[i]}, "
        details_1 += f"K:{group['Kernel_sizes_1'].iloc[i]}, S:{group['Stride_1'].iloc[i]}, C:{group['conv_channels_1'].iloc[i]}"
        ax.text(0.01, i - height/2, details_1, va='center', ha='left', fontsize=8, color='black')
        
        # Add detailed information for file 2
        details_2 = f"Sch:{group['Sch_2'].iloc[i]}, LR:{group['Final_LR_2'].iloc[i]:.2e}, Act:{group['Act_2'].iloc[i]}, "
        details_2 += f"K:{group['Kernel_sizes_2'].iloc[i]}, S:{group['Stride_2'].iloc[i]}, C:{group['conv_channels_2'].iloc[i]}"
        ax.text(0.01, i + height/2, details_2, va='center', ha='left', fontsize=8, color='black')
    
    ax.legend(fontsize=6, loc='lower right')
    ax.set_xlim(0, 1)  # Set x-axis limits from 0 to 1 for AUC scores

# Remove any unused subplots
for j in range(n_groups, len(axes)):
    fig.delaxes(axes[j])

# Add an overall title
fig.suptitle(f"Comparison of AUC Test Results with Experiment Details\n BS: 32, Dropout: 0.5, Epochs: 100\n{os.path.basename(csv_files[0])} vs {os.path.basename(csv_files[1])}", fontsize=14)

# Adjust the layout
plt.tight_layout()

# Save the plot as a figure
filename = "comparison_results_pool_cls.png"
plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

print(f"Detailed comparison plot has been saved as: {os.path.join(output_dir, filename)}")

# Close the figure to free up memory
plt.close(fig)

print("Detailed comparison plot with horizontal bars has been generated and saved.")


#****************************** Subplots per csv with layer wise subplots**********************#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import glob
# import re

# def find_csv_files(directory):
#     files = glob.glob(os.path.join(directory, '*', '*.csv'), recursive=True)
#     return sorted(files, key=lambda x: x.split('results_cnn_exp')[-1])

# def get_short_job_id(job_id):
#     return '_'.join(str(job_id).split('_')[1:2])

# def natural_sort_key(s):
#     return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# sns.set_style("whitegrid")

# base_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/transformer/pretrained/with_transformer/02_BS_32_2/'
# csv_files = find_csv_files(base_dir)
# output_dir = os.path.join(base_dir, 'plots')
# os.makedirs(output_dir, exist_ok=True)

# for csv_file in csv_files:
#     print(f"Processing file: {csv_file}")
#     df = pd.read_csv(csv_file)
    
#     if 'Exp_ID' in df.columns:
#         df['Short Jobid'] = df['Exp_ID'].apply(get_short_job_id)
#     else:
#         print(f"Warning: 'Exp_ID' column not found in {csv_file}. Using index as Short Jobid.")
#         df['Short Jobid'] = df.index.astype(str)

#     required_columns = ['test_auc', 'Sch', 'Start_LR', 'Final_LR', 'Act', 'Kernel_sizes', 'Stride', 'conv_channels', 'layer_indices']
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         print(f"Error: Missing columns in {csv_file}: {', '.join(missing_columns)}")
#         continue

#     df['sort_key'] = df['Short Jobid'].apply(natural_sort_key)
#     df = df.sort_values('sort_key')
#     df = df.drop('sort_key', axis=1)

#     # Group by layer_indices
#     grouped = df.groupby('layer_indices')
#     n_groups = len(grouped)

#     # Set up the subplot grid
#     n_cols = min(3, n_groups)
#     n_rows = (n_groups + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
#     if n_groups > 1:
#         axes = axes.flatten()
#     else:
#         axes = [axes]

#     for (layer_index, group), ax in zip(grouped, axes):
#         max_auc_index = group['test_auc'].idxmax()
#         colors = ['blue' if idx != max_auc_index else 'red' for idx in group.index]
        
#         bars = sns.barplot(y='Short Jobid', x='test_auc', data=group, ax=ax, orient='h', palette=colors)
        
#         ax.set_title(f"Layer Index: {layer_index}", fontsize=10)
#         ax.set_ylabel('Experiment ID', fontsize=8)
#         ax.set_xlabel('AUC Test', fontsize=8)
        
#         for j, bar in enumerate(bars.patches):
#             width = bar.get_width()
#             ax.text(width/2, bar.get_y() + bar.get_height()/2,
#                     f"Sch:{group.iloc[j]['Sch']}, Final:{group.iloc[j]['Final_LR']:.2e}, Act:{group.iloc[j]['Act']}, K:{group.iloc[j]['Kernel_sizes']}, S:{group.iloc[j]['Stride']}, OC:{group.iloc[j]['conv_channels']}",
#                     ha='center', va='center', fontsize=6, color='white', rotation=0)
        
#         for j, v in enumerate(group['test_auc']):
#             ax.text(v, j, f'{v:.3f}', ha='left', va='center', fontsize=6, color='black')
        
#         ax.set_yticks(range(len(group['Short Jobid'])))
#         ax.set_yticklabels(group['Short Jobid'], fontsize=6)

#     # Remove any unused subplots
#     for j in range(n_groups, len(axes)):
#         fig.delaxes(axes[j])

#     # Add an overall title
#     fig.suptitle(f"AUC Test Results for {os.path.basename(csv_file)}\nBS: 32, Dropout: 0.5, Epochs: 100", fontsize=14)

#     # Adjust the layout
#     plt.tight_layout()

#     # Save the plot as a figure
#     filename = f"{os.path.basename(csv_file).replace('.csv', '_results.png')}"
#     plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

#     print(f"Plot for {csv_file} has been saved as: {os.path.join(output_dir, filename)}")

#     # Close the figure to free up memory
#     plt.close(fig)

# print("All plots have been generated and saved.")

#****************************** Subplots per csv without layer wise subplots**********************#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import glob
# import re

# def find_csv_files(directory):
#     #files = glob.glob(os.path.join(directory, '*.csv'), recursive=True)
#     files = glob.glob(os.path.join(directory, '*', '*.csv'), recursive=True)

#     # Sort files based on the numeric part of the folder name
#     return sorted(files, key=lambda x: x.split('results_cnn_exp')[-1])

# def get_short_job_id(job_id):
#     return '_'.join(str(job_id).split('_')[:2])

# def natural_sort_key(s):
#     return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# sns.set_style("whitegrid")

# base_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/transformer/pretrained/with_transformer/02_BS_32_2/'
# csv_files = find_csv_files(base_dir)
# output_dir = os.path.join(base_dir, 'plots')
# os.makedirs(output_dir, exist_ok=True)

# # Determine the number of CSV files and set up the subplot grid
# n_files = len(csv_files)
# print(n_files)
# n_cols = min(3, n_files)  # Maximum 3 columns
# n_rows = (n_files + n_cols - 1) // n_cols

# # Create a single figure with subplots for all CSV files
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
# if n_files > 1:
#     axes = axes.flatten()
# else:
#     axes = [axes]

# for i, csv_file in enumerate(csv_files):
#     print(f"Processing file: {csv_file}")
#     df = pd.read_csv(csv_file)
    
#     # Check if 'Jobid' column exists
#     if 'Exp_ID' in df.columns:
#         df['Short Jobid'] = df['Exp_ID'].apply(get_short_job_id)
#     else:
#         print(f"Warning: 'Exp_ID' column not found in {csv_file}. Using index as Short Jobid.")
#         df['Short Jobid'] = df.index.astype(str)

#     # Check if required columns exist
#     required_columns = ['test_auc', 'Sch', 'Start_LR','Final_LR','Act','Kernel_sizes','Stride', 'conv_channels']
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         print(f"Error: Missing columns in {csv_file}: {', '.join(missing_columns)}")
#         continue

#     # Sort by test_auc and reset index
#     #df = df.sort_values('test_auc', ascending=False).reset_index(drop=True)

#     # # Sort the dataframe by Short Jobid
#     # df = df.sort_values('Short Jobid')

#      # Sort the dataframe by Short Jobid using natural sort
#     df['sort_key'] = df['Short Jobid'].apply(natural_sort_key)
#     df = df.sort_values('sort_key')
#     df = df.drop('sort_key', axis=1)
    
#     # Create the subplot for this CSV file
#     ax = axes[i]

#     # Find the index of the maximum test_auc
#     max_auc_index = df['test_auc'].idxmax()

#      # Create a color list, with red for the max AUC bar and blue for others
#     colors = ['blue' if idx != max_auc_index else 'red' for idx in df.index]
    
#     bars = sns.barplot(y='Short Jobid', x='test_auc', data=df, ax=ax, orient='h', palette=colors)
    
#     ax.set_title(f"File: {os.path.basename(csv_file)}", fontsize=10)
#     ax.set_ylabel('Experiment ID', fontsize=8)
#     ax.set_xlabel('AUC Test', fontsize=8)
    
#     # Add labels for experiment information
#     for j, bar in enumerate(bars.patches):
#         width = bar.get_width()
#         ax.text(width/2, bar.get_y() + bar.get_height()/2,
#                 f"Sch:{df.iloc[j]['Sch']}, Final:{df.iloc[j]['Final_LR']:.2e}, Act:{df.iloc[j]['Act']}, K:{df.iloc[j]['Kernel_sizes']}, S:{df.iloc[j]['Stride']}, OC:{df.iloc[j]['conv_channels']}",
#                 ha='center', va='center', fontsize=6, color='white', rotation=0)
    
#     # Add value labels at the end of the bars
#     for j, v in enumerate(df['test_auc']):
#         ax.text(v, j, f'{v:.3f}', ha='left', va='center', fontsize=6, color='black')
    
#     # Adjust y-axis tick labels
#     ax.set_yticks(range(len(df['Short Jobid'])))
#     ax.set_yticklabels(df['Short Jobid'], fontsize=6)
# # Remove any unused subplots
# for j in range(i+1, len(axes)):
#     fig.delaxes(axes[j])

# # Add an overall title
# fig.suptitle(f"AUC Test Results for All Experiments \n BS: 128, Dropout: 0.5, Epochs: 40", fontsize=14)

# # Adjust the layout
# plt.tight_layout()

# # Save the plot as a figure
# filename = "all_experiments_results.png"
# plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

# print(f"Combined plot has been saved as: {os.path.join(output_dir, filename)}")

# # Close the figure to free up memory
# plt.close(fig)




