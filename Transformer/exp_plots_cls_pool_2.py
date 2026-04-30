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

    required_columns = ['test_auc', 'layer_indices']
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
merged_df = pd.merge(dfs[0][['Short Jobid', 'test_auc', 'layer_indices']], 
                     dfs[1][['Short Jobid', 'test_auc', 'layer_indices']], 
                     on=['Short Jobid', 'layer_indices'], 
                     suffixes=('_1', '_2'))

# Group by layer_indices
grouped = merged_df.groupby('layer_indices')
n_groups = len(grouped)

# Set up the subplot grid
n_cols = min(3, n_groups)
n_rows = (n_groups + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
if n_groups > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for (layer_index, group), ax in zip(grouped, axes):
    x = range(len(group))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], group['test_auc_1'], width, label=os.path.basename(csv_files[0]), color='blue', alpha=0.7)
    ax.bar([i + width/2 for i in x], group['test_auc_2'], width, label=os.path.basename(csv_files[1]), color='red', alpha=0.7)
    
    ax.set_title(f"Layer Index: {layer_index}", fontsize=10)
    ax.set_ylabel('AUC Test', fontsize=8)
    ax.set_xlabel('Experiment ID', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(group['Short Jobid'], rotation=45, ha='right', fontsize=6)
    
    for i, (v1, v2) in enumerate(zip(group['test_auc_1'], group['test_auc_2'])):
        ax.text(i - width/2, v1, f'{v1:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
        ax.text(i + width/2, v2, f'{v2:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
    
    ax.legend(fontsize=6)

# Remove any unused subplots
for j in range(n_groups, len(axes)):
    fig.delaxes(axes[j])

# Add an overall title
fig.suptitle(f"Comparison of AUC Test Results\n{os.path.basename(csv_files[0])} vs {os.path.basename(csv_files[1])}", fontsize=14)

# Adjust the layout
plt.tight_layout()

# Save the plot as a figure
filename = "comparison_results.png"
plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

print(f"Comparison plot has been saved as: {os.path.join(output_dir, filename)}")

# Close the figure to free up memory
plt.close(fig)

print("Comparison plot has been generated and saved.")

