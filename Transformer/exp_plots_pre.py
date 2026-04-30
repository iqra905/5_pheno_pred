import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re

def find_csv_files(directory):
    files = glob.glob(os.path.join(directory, '*', '*.csv'), recursive=True)
    # Sort files based on the numeric part of the folder name
    return sorted(files, key=lambda x: x.split('results_cnn_exp')[-1])

def get_short_job_id(job_id):
    return '_'.join(str(job_id).split('_')[1:2])

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


sns.set_style("whitegrid")

base_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pan/pre/'
csv_files = find_csv_files(base_dir)
output_dir = os.path.join(base_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

# Determine the number of CSV files and set up the subplot grid
n_files = len(csv_files)
n_cols = min(4, n_files)  # Maximum 3 columns
n_rows = (n_files + n_cols - 1) // n_cols

# Create a single figure with subplots for all CSV files
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
if n_files > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for i, csv_file in enumerate(csv_files):
    print(f"Processing file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Check if 'Jobid' column exists
    if 'Exp_ID' in df.columns:
        df['Short Jobid'] = df['Exp_ID'].apply(get_short_job_id)
    else:
        print(f"Warning: 'Exp_ID' column not found in {csv_file}. Using index as Short Jobid.")
        df['Short Jobid'] = df.index.astype(str)

    # Check if required columns exist
    required_columns = ['test_auc', 'Sch', 'Start_LR','Final_LR','Act','Kernel_sizes','Stride', 'conv_channels']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in {csv_file}: {', '.join(missing_columns)}")
        continue

    # Sort by test_auc and reset index
    #df = df.sort_values('test_auc', ascending=False).reset_index(drop=True)

    # # Sort the dataframe by Short Jobid
    # df = df.sort_values('Short Jobid')

     # Sort the dataframe by Short Jobid using natural sort
    df['sort_key'] = df['Short Jobid'].apply(natural_sort_key)
    df = df.sort_values('sort_key')
    df = df.drop('sort_key', axis=1)
    
    # Create the subplot for this CSV file
    ax = axes[i]

    # Find the index of the maximum test_auc
    max_auc_index = df['test_auc'].idxmax()

     # Create a color list, with red for the max AUC bar and blue for others
    colors = ['blue' if idx != max_auc_index else 'red' for idx in df.index]
    
    bars = sns.barplot(y='Short Jobid', x='test_auc', data=df, ax=ax, orient='h', palette=colors)
    
    ax.set_title(f"File: {os.path.basename(csv_file)}", fontsize=10)
    ax.set_ylabel('Experiment ID', fontsize=8)
    ax.set_xlabel('AUC Test', fontsize=8)
    
    # Add labels for experiment information
    for j, bar in enumerate(bars.patches):
        width = bar.get_width()
        ax.text(width/2, bar.get_y() + bar.get_height()/2,
                f"Sch:{df.iloc[j]['Sch']}, Final:{df.iloc[j]['Final_LR']:.2e}, Act:{df.iloc[j]['Act']}, \n K:{df.iloc[j]['Kernel_sizes']}, S:{df.iloc[j]['Stride']}, OC:{df.iloc[j]['conv_channels']}",
                ha='center', va='center', fontsize=10, color='white', rotation=0)
    
    # Add value labels at the end of the bars
    for j, v in enumerate(df['test_auc']):
        ax.text(v, j, f'{v:.3f}', ha='left', va='center', fontsize=9, color='black')
    
    # Adjust y-axis tick labels
    ax.set_yticks(range(len(df['Short Jobid'])))
    ax.set_yticklabels(df['Short Jobid'], fontsize=6)
# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add an overall title
fig.suptitle(f"AUC Test Results for All Experiments \n BS: 128, Dropout: 0.5,Epochs: 40, Start_LR: 1e-4, Optimizer: AdamW, WD: 0.5", fontsize=14)

# Adjust the layout
plt.tight_layout()

# Save the plot as a figure
filename = "all_experiments_results.png"
plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

print(f"Combined plot has been saved as: {os.path.join(output_dir, filename)}")

# Close the figure to free up memory
plt.close(fig)




