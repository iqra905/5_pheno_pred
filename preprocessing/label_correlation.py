import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_correlation(disease, samples, excel_file, col1, col2):
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Analyzing correlation between '{col1}' and '{col2}'")
    print("="*60)
    
    # Basic statistics
    print(f"\n{col1} Statistics:")
    print(df[col1].describe())
    
    print(f"\n{col2} Value Counts:")
    print(df[col2].value_counts().sort_index())
    
    # Remove missing values for correlation analysis
    clean_data = df[[col1, col2]].dropna()
    print(f"\nSamples with both values available: {len(clean_data)}")
    print(f"Missing values - {col1}: {df[col1].isnull().sum()}, {col2}: {df[col2].isnull().sum()}")
    
    if len(clean_data) < 2:
        print("Not enough data for correlation analysis!")
        return
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(clean_data[col1], clean_data[col2])
    #spearman_corr, spearman_p = spearmanr(clean_data[col1], clean_data[col2])
    
    print(f"\nCorrelation Results:")
    print(f"Pearson correlation:  {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    #print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
    
    # Interpret the correlation
    if abs(pearson_corr) < 0.1:
        strength = "negligible"
    elif abs(pearson_corr) < 0.3:
        strength = "weak"
    elif abs(pearson_corr) < 0.5:
        strength = "moderate"
    elif abs(pearson_corr) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if pearson_corr > 0 else "negative"
    print(f"\nInterpretation: {strength} {direction} correlation")
    
    # # Statistical significance
    # alpha = 0.05
    # if pearson_p < alpha:
    #     print(f"The correlation is statistically significant (p < {alpha})")
    # else:
    #     print(f"The correlation is NOT statistically significant (p >= {alpha})")
    
    # Group analysis
    print(f"\n{col1} analysis by {col2} groups:")
    col_by_group = clean_data.groupby(col2)[col1].agg(['count', 'mean', 'std', 'min', 'max'])
    print(col_by_group)
    
    # Create a simple visualization
    # try:
    #     plt.figure(figsize=(12, 8))
        
    #     # Subplot 1: Scatter plot
    #     plt.subplot(1, 2, 1)
    #     plt.scatter(clean_data[col2], clean_data[col1], alpha=0.6)
    #     plt.xlabel(f'{col2} (0=Control, 1=Case)')
    #     plt.ylabel(f'{col1}')
    #     plt.title(f'Scatter Plot\nr = {pearson_corr:.3f}')
    #     plt.grid(True, alpha=0.3)

    #     # Add count annotations
    #     controls_count = len(clean_data[clean_data[col2] == 0])
    #     cases_count = len(clean_data[clean_data[col2] == 1])

    #     # Get y-axis limits for positioning text
    #     y_min, y_max = plt.ylim()
    #     text_height = y_max - (y_max - y_min) * 0.1  # 10% from top

    #     # Add count text
    #     plt.text(0, text_height, f'n = {controls_count}', 
    #             ha='center', va='top', fontsize=10, fontweight='bold', alpha=0.7)
    #     plt.text(1, text_height, f'n = {cases_count}', 
    #             ha='center', va='top', fontsize=10, fontweight='bold', alpha=0.7)


    #     # Subplot 2: Histogram
    #     plt.subplot(1, 2, 2)
    #     for group in clean_data[col2].unique():
    #         group_data = clean_data[clean_data[col2] == group][col1]
    #         plt.hist(group_data, alpha=0.7, label=f'{col2}={group}', bins=20)
    #     plt.xlabel(f'{col1}')
    #     plt.ylabel('Frequency')
    #     plt.title(f'{col1} Distribution')
    #     plt.legend()
        
    #     # # Subplot 3: Box plot
    #     # plt.subplot(1, 3, 3)
    #     # clean_data.boxplot(column=col1, by=col2, ax=plt.gca())
    #     # plt.title(f'{col1} Distribution by {col2}')
    #     plt.suptitle(f'Correlation between {col1} and {col2} for {disease} \n Samples ({samples})')
        
        
        
    #     plt.tight_layout()
    #     output_folder = f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/preprocessing/correlation/{disease}'
    #     os.makedirs(output_folder, exist_ok=True)
    #     plt.savefig(os.path.join(output_folder,f'correlation_analysis_{samples}_{col2}_{col1}_distribution.png'), dpi=150, bbox_inches='tight')
    #     print(f"\nVisualization saved as correlation_analysis_{samples}_{col2}_{col1}_distribution.png")
        
    # except ImportError:
    #     print("\nNote: matplotlib not available for visualization")

    try:
        plt.figure(figsize=(8, 8))
        # Subplot 2: Histogram
        for group in clean_data[col2].unique():
            group_data = clean_data[clean_data[col2] == group][col1]
            plt.hist(group_data, alpha=0.7, label=f'{col2}={group}', bins=20)
        plt.xlabel(f'{col1}')
        plt.ylabel('Frequency')
        plt.title(f'Age Distribution for Prostate Cancer Samples ({samples})')
        plt.legend()
        plt.tight_layout()
        output_folder = f'/vol/research/fmodal_mmmed/Codes/5_disease_experiments/preprocessing/correlation/{disease}'
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder,f'histogram_{samples}_{col2}_{col1}.png'), dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved as correlation_analysis_{samples}_{col2}_{col1}_distribution.png")
        
    except ImportError:
        print("\nNote: matplotlib not available for visualization")
    
    return pearson_corr, pearson_p

if __name__ == "__main__":
    diseases = ['Prostrate_Cancer', 'Pancreatic_Cancer', 'Colon_Cancer', 'Breast_Cancer', 'T2D']
    samples = 'Disease-wise only'
    excels_all = ['pros_can', 'pan_can', 'col_can', 'brea_can', 't2d']
    col2_all = ['pros01', 'panca', 'crc', 'breacancer', 't2dm']
    excel_file_base = '/vol/research/ucdatasets/gwas/data_files/disease_pheno/'
    excel_file_all_samples = '/vol/research/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx'
    col1 = 'Agexit'

    for disease, excel, col2 in zip(diseases, excels_all, col2_all):
        excel_file = os.path.join(excel_file_base,f'{excel}.xlsx')
        analyze_correlation(disease, samples, excel_file, col1, col2)

    # for disease, col2 in zip(diseases, col2_all):
    #     analyze_correlation(disease, samples, excel_file_all_samples, col1, col2)