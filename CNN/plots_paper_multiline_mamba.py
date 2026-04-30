"""
Multi-line Performance Chart Generator
Reads data from Excel workbook and creates performance scaling visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION - Modify these settings for your data
# ============================================================================

# Excel file path
EXCEL_FILE = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Mamba/results_mamba.xlsx'  # Change this to your Excel file name

# Sheet name containing the data
SHEET_NAME = 'Sheet3'  # Change to your sheet name

# Data structure configuration
# Option 1: If data is structured like your screenshot (diseases as rows)
DATA_STRUCTURE = 'rows'  # 'rows' or 'columns'

# Column names or indices for sequence lengths in Excel
# If your columns are named, use names; if not, use indices (0-based)
SEQ_LENGTH_COLS = {
    1224: 'Seq Len: 1224',   # or use index like: 1224: 1
    2448: 'Seq Len: 2448',   # or use index like: 2448: 2
    4895: 'Seq Len: 4895',   # or use index like: 4895: 3
    9789: 'Seq Len: 9789',   # or use index like: 9789: 4
}

# Transformer baseline column (if exists)
TRANSFORMER_COL = 'Transformer Seq Len: 1224'  # or index, or None if not present

# Disease names (should match what's in your Excel)
# If using row structure, these should match your 'Disease' column values
DISEASES = [
    'Prostate Cancer',
    'Pancreatic Cancer',
    'Colorectal Cancer',
    'Breast Cancer',
    'T2D'
]

# Color and marker configuration for each disease
DISEASE_STYLES = {
    'Prostate Cancer': {
        'color': '#1f77b4',      # Blue
        'marker': 'o',
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 8
    },
    'Pancreatic Cancer': {
        'color': '#2ca02c',      # Green
        'marker': 's',           # Square
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 8
    },
    'Colorectal Cancer': {
        'color': '#8c564b',      # Brown
        'marker': '^',           # Triangle
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 8
    },
    'Breast Cancer': {
        'color': '#7f7f7f',      # Gray
        'marker': 'D',           # Diamond
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 7
    },
    'T2D': {
        'color': '#17becf',      # Cyan
        'marker': 'v',           # Triangle down
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 8
    }
}

# Plot configuration
PLOT_CONFIG = {
    'figsize': (12, 7),
    'title': ' ',
    'xlabel': 'Sequence Length (Tokens)',
    'ylabel': 'Performance Score (AUC)',
    'ylim': (0.65, 1.00),
    'use_log_scale': True,
    'grid': True,
    'legend_location': 'upper center',
    'legend_bbox': (0.5, -0.15),
    'legend_ncol': 2
}

# Output file
OUTPUT_FILE = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Mamba/performance_scaling_chart_no_cov.pdf'

# ============================================================================
# MAIN SCRIPT - You typically won't need to modify below this line
# ============================================================================

def read_excel_data(file_path, sheet_name, structure='rows'):
    """
    Read performance data from Excel file
    
    Parameters:
    -----------
    file_path : str
        Path to Excel file
    sheet_name : str
        Name of the sheet to read
    structure : str
        'rows' if diseases are in rows, 'columns' if in columns
    
    Returns:
    --------
    dict : Dictionary with disease names as keys and performance data as values
    """
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    print(f"Excel file loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    data = {}
    
    if structure == 'rows':
        # Diseases are in rows
        # Assuming first column is 'Disease' or similar
        disease_col = df.columns[0]
        
        for disease in DISEASES:
            # Find the row for this disease
            disease_row = df[df[disease_col] == disease]
            
            if disease_row.empty:
                print(f"Warning: Disease '{disease}' not found in data")
                continue
            
            # Extract Mamba performance values for different sequence lengths
            mamba_values = []
            for seq_len in sorted(SEQ_LENGTH_COLS.keys()):
                col = SEQ_LENGTH_COLS[seq_len]
                if isinstance(col, str):
                    value = disease_row[col].values[0]
                else:
                    value = disease_row.iloc[0, col]
                mamba_values.append(value)
            
            data[disease] = {
                'mamba': mamba_values,
                'transformer': None
            }
            
            # Get transformer baseline if exists
            if TRANSFORMER_COL:
                try:
                    if isinstance(TRANSFORMER_COL, str):
                        transformer_value = disease_row[TRANSFORMER_COL].values[0]
                    else:
                        transformer_value = disease_row.iloc[0, TRANSFORMER_COL]
                    data[disease]['transformer'] = transformer_value
                except:
                    print(f"Warning: Could not find transformer data for {disease}")
    
    return data


def create_performance_chart(data, seq_lengths, output_file='output.png'):
    """
    Create multi-line performance scaling chart
    
    Parameters:
    -----------
    data : dict
        Dictionary with disease performance data
    seq_lengths : list
        List of sequence lengths
    output_file : str
        Output file path
    """
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    
    # Plot Mamba lines for each disease
    for disease, values in data.items():
        if disease not in DISEASE_STYLES:
            print(f"Warning: No style defined for {disease}, using default")
            style = {'color': 'black', 'marker': 'o', 'linestyle': '-', 
                    'linewidth': 2, 'markersize': 8}
        else:
            style = DISEASE_STYLES[disease]
        
        mamba_values = values['mamba']
        
        ax.plot(seq_lengths, mamba_values, 
                marker=style['marker'],
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                label=f"Mamba - {disease}")
    
    # Plot Transformer baselines
    transformer_plotted = False
    for disease, values in data.items():
        if values['transformer'] is not None:
            style = DISEASE_STYLES.get(disease, {})
            color = style.get('color', 'black')
            
            label = 'Transformer (1224)' if not transformer_plotted else None
            transformer_plotted = True
            
            ax.plot(seq_lengths[0], values['transformer'], 
                   'x', 
                   markersize=12, 
                   markeredgewidth=3,
                   color=color,
                   label=label)
    
    # Customize the plot
    ax.set_xlabel(PLOT_CONFIG['xlabel'], fontsize=12, fontweight='bold')
    ax.set_ylabel(PLOT_CONFIG['ylabel'], fontsize=12, fontweight='bold')
    ax.set_title(PLOT_CONFIG['title'], fontsize=14, fontweight='bold', pad=15)
    
    # Set x-axis scale
    if PLOT_CONFIG['use_log_scale']:
        ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([str(sl) for sl in seq_lengths])
    
    # Set y-axis limits
    if PLOT_CONFIG['ylim']:
        ax.set_ylim(PLOT_CONFIG['ylim'])
    
    # Add grid
    if PLOT_CONFIG['grid']:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc=PLOT_CONFIG['legend_location'], 
             bbox_to_anchor=PLOT_CONFIG['legend_bbox'],
             fontsize=10, 
             framealpha=0.9,
             ncol=PLOT_CONFIG['legend_ncol'])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved successfully to: {output_file}")
    
    return fig, ax


def main():
    """Main execution function"""
    print("=" * 70)
    print("Multi-line Performance Chart Generator")
    print("=" * 70)
    
    # Read data from Excel
    try:
        data = read_excel_data(EXCEL_FILE, SHEET_NAME, DATA_STRUCTURE)
    except FileNotFoundError:
        print(f"\nError: Excel file '{EXCEL_FILE}' not found!")
        print("Please update EXCEL_FILE path in the configuration section.")
        return
    except Exception as e:
        print(f"\nError reading Excel file: {e}")
        return
    
    # Get sequence lengths
    seq_lengths = sorted(SEQ_LENGTH_COLS.keys())
    
    # Create chart
    try:
        fig, ax = create_performance_chart(data, seq_lengths, OUTPUT_FILE)
        plt.show()
    except Exception as e:
        print(f"\nError creating chart: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()