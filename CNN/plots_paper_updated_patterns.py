import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for Nature Communications compliance
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # Use available font instead of Arial
    'font.size': 8,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.005,
    'grid.linewidth': 0.005,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'hatch.linewidth': 0.0001,  # Control hatch pattern line width
    'figure.dpi': 500,
    'savefig.dpi': 500,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Option 5: Nature Accessible (Maximum Color-blind Support)
# Optimized for deuteranopia and protanopia
disease_colors_option5 = {
    'Prostate Cancer': '#0173B2',    # Blue
    'Pancreatic Cancer': '#DE8F05',  # Orange
    'Colon Cancer': '#029E73',       # Bluish Green
    'Breast Cancer': '#CC78BC',      # Rose
    'T2D': '#949494'                 # Gray
}


class NatureGWASVisualizer:
    def __init__(self, excel_path, output_dir):
        """Initialize with Excel file path and output directory"""
        self.excel_path = excel_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use colorblind-friendly disease colors
        self.disease_colors = disease_colors_option5
        
        # Method colors
        self.method_colors = {
            'MLP': '#1f77b4',           # Blue
            'MLP - Chr': '#aec7e8',     # Light Blue
            'CNN': '#ff7f0e',           # Orange
            'CNN - Chr': '#ffbb78',     # Light Orange
            'DeepCombi': '#d62728',     # Red
            'GenNet': '#2ca02c',        # Green
            'Population_No_COV': '#d62728',    # Warning Red
            'Population_COV': '#8B0000',       # Dark Red
            'Train_No_COV': '#2ca02c',         # Safe Green
            'Train_COV': '#006400',            # Dark Green
            'Without_Covariates': '#1f77b4',   # Blue
            'With_Covariates': '#ff7f0e',      # Orange
            'Disease-wise Singlescale': '#1f77b4',            # Blue
            'Disease-wise Multiscale': '#d62728', #Red
            'Multilabel Singlescale': '#ff7f0e',           # Orange  
            'Multilabel Multiscale': '#2ca02c'            # Green
        }
        
        self.load_data()
    
    def load_data(self):
        """Load all data sheets from Excel file"""
        try:
            self.data = {}
            sheet_names = ['Setting01', 'Setting02', 'Multilabel', 'Multiscale', 
                          'Comparison_no_cov', 'Comparison_cov']
            
            for sheet in sheet_names:
                if sheet == 'Setting02':
                    # Handle multi-level headers for Setting02
                    self.data[sheet] = pd.read_excel(self.excel_path, sheet_name=sheet, header=[0, 1])
                    print(f"Loaded {sheet}: {self.data[sheet].shape}")
                    print(f"Columns: {self.data[sheet].columns.tolist()}")
                else:
                    self.data[sheet] = pd.read_excel(self.excel_path, sheet_name=sheet)
                    print(f"Loaded {sheet}: {self.data[sheet].shape}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def apply_disease_colors_to_labels(self, ax, diseases, rotation=0):
        """Apply black color to x-axis labels (as requested)"""
        for i, disease in enumerate(diseases):
            ax.get_xticklabels()[i].set_color('black')
            ax.get_xticklabels()[i].set_fontweight('bold')
    
    def create_figure01_baseline(self):
        """Figure 1: Baseline comparison (Setting01) - grouped by methods"""
        fig, ax = plt.subplots(figsize=(12,2.5))
        
        df = self.data['Setting01']
        diseases = df['Disease'].tolist()
        methods = ['MLP', 'MLP - Chr', 'CNN', 'CNN - Chr', 'DeepCombi', 'GenNet']
        
        # Prepare data - group by methods instead of diseases
        x = np.arange(len(methods))
        width = 0.14  # Reduced width to add more space between bars
        spacing = 1.1  # Increased spacing multiplier to add space between bars
        
        # Create legend handles for diseases
        legend_handles = []
        
        # Plot bars for each disease within each method group
        for i, disease in enumerate(diseases):
            disease_values = []
            disease_bars = []
            
            for method in methods:
                value = df[df['Disease'] == disease][method].iloc[0]
                disease_values.append(value)
            
            # Plot bars for this disease across all methods with increased spacing
            bars = ax.bar(x + (i - 2) * width * spacing, disease_values, width, 
                        label=disease, color=self.disease_colors[disease], edgecolor='black', linewidth=0.05)
            disease_bars.extend(bars)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, disease_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=5.5)
            
            # Create legend handle for this disease
            legend_handles.append(plt.Rectangle((0,0),1,1, facecolor=self.disease_colors[disease], edgecolor='black', linewidth=0.05, label=disease))
        
        ax.set_ylabel('AUC', fontweight='bold', fontsize=10)
        ax.set_xlabel('Method', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8, color='black', fontweight='bold')
        
        # Remove top and right spines (borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Reduce space between y-axis and bars
        ax.margins(x=0.02)  # Reduce horizontal margins
        
        # Create custom legend for diseases
        ax.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_handles), 
            fontsize=8,
            frameon=False
        )
        ax.grid(True, alpha=0.1, axis='y')
        ax.set_ylim(0.4, 0.7)
        ax.set_yticks(np.arange(0.4, 0.71, 0.1))  # Include 0.7
        ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.4, 0.71, 0.1)], fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.25)

        plt.savefig(os.path.join(self.output_dir, 'fig_01_baseline.pdf'), format='pdf', bbox_inches='tight', dpi=300)
        print("Figure 1 saved: fig_01_baseline.pdf")
    
    def create_figure02_data_leakage(self):
        """Figure 2: Data leakage analysis (Setting02) - comparing with/without covariates for each training condition"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2.5))
        
        df = self.data['Setting02']
        
        # Handle different possible column structures
        try:
            # Try multi-level column access first
            diseases = df[df.columns[0]].tolist()  # First column should be Disease
            
            # Check if we have multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                pop_no_cov = df[('Population Dataset', 'No COV')].tolist()
                train_no_cov = df[('Train Set', 'No COV')].tolist()
                pop_cov = df[('Population Dataset', 'COV')].tolist()
                train_cov = df[('Train Set', 'COV')].tolist()
            else:
                # If not multi-level, access by position or try to find column names
                col_names = df.columns.tolist()
                print(f"Column names in Setting02: {col_names}")
                
                # Assume order: Disease, Pop_No_COV, Pop_COV, Train_No_COV, Train_COV
                diseases = df.iloc[:, 0].tolist()
                pop_no_cov = df.iloc[:, 1].tolist()
                pop_cov = df.iloc[:, 2].tolist()
                train_no_cov = df.iloc[:, 3].tolist()
                train_cov = df.iloc[:, 4].tolist()
                
        except Exception as e:
            print(f"Error accessing Setting02 data: {e}")
            print(f"Dataframe structure: {df.head()}")
            print(f"Columns: {df.columns}")
        
        # Common settings
        x = np.arange(len(diseases))
        width = 0.32  # Reduced width for better spacing
        spacing = 1.1  # Add spacing between bars
        
        # Subplot 1: Population Dataset (With vs Without Covariates)
        for i, disease in enumerate(diseases):
            base_color = self.disease_colors[disease]
            
            # Without Covariates - solid bars
            ax1.bar(x[i] - width/2 * spacing, pop_no_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.05, label='Without Covariates' if i == 0 else "")
                
            # Label for without covariates bar
            ax1.text(x[i] - width/2 * spacing, pop_no_cov[i] + 0.001, f"{pop_no_cov[i]:.3f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=5.5)
            
            # With Covariates - hatched bars
            bars = ax1.bar(x[i] + width/2 * spacing, pop_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.05, hatch='///', label='With Covariates' if i == 0 else "")
            
            # Label for with covariates bar
            ax1.text(x[i] + width/2 * spacing, pop_cov[i] + 0.001, f"{pop_cov[i]:.3f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=5.5)

        # Create custom legend for subplot 1
        legend_elements_1 = [
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, label='Without Covariates'),
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, hatch='///', label='With Covariates')
        ]
        
        # Remove top and right spines for subplot 1
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.margins(x=0.05)
        
        ax1.set_ylabel('AUC', fontweight='bold', fontsize=10)
        ax1.set_xlabel('Disease', fontweight='bold', fontsize=10)
        ax1.set_title('A. Population Dataset (Data Leakage)', fontweight='bold', fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(diseases, fontsize=7, color='black', fontweight='bold')
        ax1.legend(
            handles=legend_elements_1,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_elements_1), 
            fontsize=8,
            frameon=False
        )
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0.2, 1.0)
        ax1.set_yticks(np.arange(0.2,1.2, 0.2)) 
        ax1.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.2,1.2, 0.2)], fontweight='bold', fontsize=8)
        
        
        # Subplot 2: Train Set Only (With vs Without Covariates)
        for i, disease in enumerate(diseases):
            base_color = self.disease_colors[disease]
            
            # Without Covariates - solid bars
            ax2.bar(x[i] - width/2 * spacing, train_no_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.05, label='Without Covariates' if i == 0 else "")
                
            # Label for without covariates bar
            ax2.text(x[i] - width/2 * spacing, train_no_cov[i] + 0.01, f"{train_no_cov[i]:.3f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=5.5)
            
            # With Covariates
            bars = ax2.bar(x[i] + width/2 * spacing, train_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.05, hatch='///', label='With Covariates' if i == 0 else "")
            
            # Label for with covariates bar
            ax2.text(x[i] + width/2 * spacing, train_cov[i] + 0.001, f"{train_cov[i]:.3f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=5.5)
        
        # Create custom legend for subplot 2
        legend_elements_2 = [
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, label='Without Covariates'),
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, hatch='///', label='With Covariates')
        ]
        
        # Remove top and right spines for subplot 2
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.margins(x=0.02)
        
        ax2.set_ylabel('AUC', fontweight='bold', fontsize=10)
        ax2.set_xlabel('Disease', fontweight='bold', fontsize=10)
        ax2.set_title('B. Train Set Only (Proper Training)', fontweight='bold', fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(diseases, fontsize=7, color='black', fontweight='bold')
        ax2.legend(
            handles=legend_elements_2,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_elements_2), 
            fontsize=8,
            frameon=False
        )
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0.2, 1.0)
        ax2.set_yticks(np.arange(0.2,1.2, 0.2)) 
        ax2.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.2,1.2, 0.2)], fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.25)
        plt.savefig(os.path.join(self.output_dir, 'fig_02_leakage.pdf'), format='pdf', bbox_inches='tight', dpi=300)
        print("Figure 2 saved: fig_02_leakage.pdf")

    def create_figure03_multilabel(self):
        """Figure 3: Multilabel framework performance with patterns"""
        fig, ax = plt.subplots(figsize=(8, 2.5))
        
        df = self.data['Multilabel']
        diseases = df['Disease'].tolist()
        
        without_cov = df['Without Covariates'].tolist()
        with_cov = df['With Covariates'].tolist()
        
        x = np.arange(len(diseases))
        width = 0.3  # Reduced width for better spacing
        spacing = 1.1 # Add spacing between bars
        
        # Use disease-based colors with patterns instead of light/dark shades
        for i, disease in enumerate(diseases):
            base_color = self.disease_colors[disease]
            
            # Without covariates - solid bars
            bar1 = ax.bar(x[i] - width/2 * spacing, without_cov[i], width, 
                        color=base_color, edgecolor='black', linewidth=0.05)
            
            # With covariates - striped pattern
            bar2 = ax.bar(x[i] + width/2 * spacing, with_cov[i], width, 
                        color=base_color, edgecolor='black', linewidth=0.05, hatch='///')
            
            # Add value labels
            ax.text(x[i] - width/2 * spacing, without_cov[i] + 0.001,
                f'{without_cov[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(x[i] + width/2 * spacing, with_cov[i] + 0.001,
                f'{with_cov[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Calculate and show improvements/decreases
            improvement = with_cov[i] - without_cov[i]
            if improvement >= 0:
                ax.annotate(f'+{improvement:.3f}', 
                        xy=(x[i] + width/2 * spacing, with_cov[i] - 0.03), 
                        ha='center', va='top', fontsize=7, 
                        fontweight='bold', color='white')
            else:
                ax.annotate(f'{improvement:.3f}', 
                        xy=(x[i] + width/2 * spacing, with_cov[i] - 0.03), 
                        ha='center', va='top', fontsize=7, 
                        fontweight='bold', color='white')
        
        # Create custom legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, label='Without Covariates'),
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, hatch='///', label='With Covariates')
        ]
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.02)
        
        ax.set_ylabel('AUC', fontweight='bold', fontsize=10)
        ax.set_xlabel('Disease', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(diseases, fontsize=8, color='black', fontweight='bold')
        # Apply black color to disease labels
        self.apply_disease_colors_to_labels(ax, diseases)
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_elements), 
            fontsize=8,
            frameon=False
        )
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks(np.arange(0.4,1.1, 0.1)) 
        ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.4,1.1, 0.1)], fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.25)

        plt.savefig(os.path.join(self.output_dir, 'fig_03_multilabel.pdf'), 
                format='pdf', bbox_inches='tight', dpi=300)
        print("Figure 3 saved: fig_03_multilabel.pdf")

    def create_figure04_multiscale(self):
        """Figure 4: Multiscale architecture performance with patterns"""
        fig, ax = plt.subplots(figsize=(8, 2.5))
        
        df = self.data['Multiscale']
        diseases = df['Disease'].tolist()
        
        without_cov = df['Without Covariates'].tolist()
        with_cov = df['With Covariates'].tolist()
        
        x = np.arange(len(diseases))
        width = 0.3  # Reduced width for better spacing
        spacing = 1.1  # Add spacing between bars
        
        # Use disease-based colors with patterns instead of light/dark shades
        for i, disease in enumerate(diseases):
            base_color = self.disease_colors[disease]
            
            # Without covariates - solid bars
            bar1 = ax.bar(x[i] - width/2 * spacing, without_cov[i], width, 
                        color=base_color, edgecolor='black', linewidth=0.05)
            
            # With covariates - grid pattern
            bar2 = ax.bar(x[i] + width/2 * spacing, with_cov[i], width, 
                        color=base_color, edgecolor='black', linewidth=0.05, hatch='///')
            
            # Add value labels
            ax.text(x[i] - width/2 * spacing, without_cov[i] + 0.001,
                f'{without_cov[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(x[i] + width/2 * spacing, with_cov[i] + 0.001,
                f'{with_cov[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Calculate and show improvements/decreases
            improvement = with_cov[i] - without_cov[i]
            if improvement >= 0:
                ax.annotate(f'+{improvement:.3f}', 
                        xy=(x[i] + width/2 * spacing, with_cov[i] - 0.03), 
                        ha='center', va='top', fontsize=7, 
                        fontweight='bold', color='white')
            else:
                ax.annotate(f'{improvement:.3f}', 
                        xy=(x[i] + width/2 * spacing, with_cov[i] - 0.03), 
                        ha='center', va='top', fontsize=7, 
                        fontweight='bold', color='white')
        
        # Create custom legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, label='Without Covariates'),
            plt.Rectangle((0,0),1,1, facecolor=base_color, edgecolor='black', 
                        linewidth=0.05, hatch='///', label='With Covariates')
        ]
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.02)
        
        ax.set_ylabel('AUC', fontweight='bold', fontsize=10)
        ax.set_xlabel('Disease', fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(diseases, fontsize=8, color='black', fontweight='bold')
        # Apply black color to disease labels
        self.apply_disease_colors_to_labels(ax, diseases)
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_elements), 
            fontsize=8,
            frameon=False
        )
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks(np.arange(0.4,1.1, 0.1)) 
        ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.4,1.1, 0.1)], fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.25)

        plt.savefig(os.path.join(self.output_dir, 'fig_04_multiscale.pdf'), 
                format='pdf', bbox_inches='tight', dpi=300)
        print("Figure 4 saved: fig_04_multiscale.pdf")
        
    def create_radar_plot(self, data_dict, title, filename, show_covariates_label=False):
        """Create radar plot for method comparison"""
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
        
        diseases = list(data_dict[list(data_dict.keys())[0]].keys())
        methods = list(data_dict.keys())
        
        # Calculate angles for each disease
        angles = [(n + 0.9) / float(len(diseases)) * 2 * pi for n in range(len(diseases))]
        angles += angles[:1]  # Complete the circle
        
        # Plot each method
        for method in methods:
            values = [data_dict[method][disease] for disease in diseases]
            values += values[:1]  # Complete the circle
            
            # Plot with method-specific styling - NO FILL, only outline
            ax.plot(angles, values, color=self.method_colors[method], linewidth=1, 
                marker='o', markersize=2, label=method)
            
            # Add value labels ONLY for 'Multilabel + Multiscale' method
            if method == 'Multilabel Multiscale':
                for i, (angle, value, disease) in enumerate(zip(angles[:-1], values[:-1], diseases)):
                    
                    # Set label radius for Multilabel + Multiscale
                    label_radius = value 

                    # Adjust label position based on angle to avoid overlap
                    if angle < pi/2 or angle > 3*pi/2:  # Right side
                        ha_align = 'left'
                    else:  # Left side
                        ha_align = 'right'
                    
                    ax.text(angle + 0.12, label_radius, f'{value:.3f}', 
                        ha=ha_align, va='center', fontsize=6, fontweight='bold',
                        color=self.method_colors[method],
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor=self.method_colors[method], linewidth=0.0001))
        
        # Customize the plot - hide default tick labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # Hide default labels
        ax.set_ylim(0.3, 1)  # Extended range to accommodate external labels
        ax.set_rmax(1.0)       # Set maximum radius limit
        ax.set_clip_on(True)   # Enable clipping to keep everything inside
        ax.set_yticks(np.arange(0.4,1.1, 0.1)) 
        ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.4,1.1, 0.1)], fontsize=6)
        ax.grid(True, alpha=0.4, linewidth=.05)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels=[])  # Hide radial grid labels that extend outside
        
        # Add disease labels outside the main circle
        label_radius = 1.02  # Position outside the main plot area
        for angle, disease in zip(angles[:-1], diseases):
            # Calculate text alignment based on angle
            if angle == 0:  # Right
                ha, va = 'left', 'center'
            elif angle == pi/2:  # Top
                ha, va = 'center', 'bottom'
            elif angle == pi:  # Left
                ha, va = 'right', 'center'
            elif angle == 3*pi/2:  # Bottom
                ha, va = 'center', 'top'
            else:
                # For other angles, determine alignment based on quadrant
                if 0 < angle < pi/2:  # Top-right
                    ha, va = 'left', 'bottom'
                elif pi/2 < angle < pi:  # Top-left
                    ha, va = 'right', 'bottom'
                elif pi < angle < 3*pi/2:  # Bottom-left
                    ha, va = 'right', 'top'
                else:  # Bottom-right
                    ha, va = 'left', 'top'
            
            ax.text(angle, label_radius, disease, fontsize=8, fontweight='bold', 
                    color='black', ha=ha, va=va,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                            edgecolor='lightgray', linewidth=0.005))
        
        # Remove outer spine for cleaner look
        ax.spines['polar'].set_visible(False)
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=6, framealpha=False)
        
        # Add title
        title_text = title
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.25)
        plt.savefig(os.path.join(self.output_dir, filename), format='pdf', bbox_inches='tight', dpi=300)
    
    def create_figure05_radar_no_cov(self):
        """Figure 5: Radar plot comparison without covariates"""
        df = self.data['Comparison_no_cov']
        diseases = df['Disease'].tolist()
        
        # Prepare data dictionary
        data_dict = {
            'Disease-wise Singlescale': {},
            'Disease-wise Multiscale': {},
            'Multilabel Singlescale': {},
            'Multilabel Multiscale': {}
        }
        
        for i, disease in enumerate(diseases):
            data_dict['Disease-wise Singlescale'][disease] = df.iloc[i]['Disease-wise Singlescale']
            data_dict['Disease-wise Multiscale'][disease] = df.iloc[i]['Disease-wise Multiscale']
            data_dict['Multilabel Singlescale'][disease] = df.iloc[i]['Multilabel Singlescale'] 
            data_dict['Multilabel Multiscale'][disease] = df.iloc[i]['Multilabel Multiscale']
        
        self.create_radar_plot(data_dict, 
                             'Method Comparison Across Diseases \n (Without Covariates)', 
                             'fig_05_comparison_no_cov.pdf',
                             show_covariates_label=True)
        print("Figure 5 saved: fig_05_comparison_no_cov.pdf")
    
    def create_figure06_radar_with_cov(self):
        """Figure 6: Radar plot comparison with covariates"""
        df = self.data['Comparison_cov']
        diseases = df['Disease'].tolist()
        
        # Prepare data dictionary
        data_dict = {
            'Disease-wise Singlescale': {},
            'Disease-wise Multiscale': {},
            'Multilabel Singlescale': {},
            'Multilabel Multiscale': {}
        }
        
        for i, disease in enumerate(diseases):
            data_dict['Disease-wise Singlescale'][disease] = df.iloc[i]['Disease-wise Singlescale']
            data_dict['Disease-wise Multiscale'][disease] = df.iloc[i]['Disease-wise Multiscale']
            data_dict['Multilabel Singlescale'][disease] = df.iloc[i]['Multilabel Singlescale'] 
            data_dict['Multilabel Multiscale'][disease] = df.iloc[i]['Multilabel Multiscale']

        self.create_radar_plot(data_dict, 
                             'Method Comparison Across Diseases \n (With Covariates)', 
                             'fig_06_comparison_cov.pdf',
                             show_covariates_label=True)
        print("Figure 6 saved: fig_05_comparison_cov.pdf")
    
    def generate_all_figures(self):
        """Generate all 6 figures"""
        print("Starting figure generation...")
        print("="*50)
        
        print("Generating Figure 1: Baseline Comparison...")
        self.create_figure01_baseline()
        
        print("\nGenerating Figure 2: Data Leakage Analysis...")
        self.create_figure02_data_leakage()
        
        print("\nGenerating Figure 3: Multilabel Framework...")
        self.create_figure03_multilabel()
        
        print("\nGenerating Figure 4: Multiscale Architecture...")
        self.create_figure04_multiscale()
        
        print("\nGenerating Figure 5: Radar Plot (No Covariates)...")
        self.create_figure05_radar_no_cov()
        
        print("\nGenerating Figure 6: Radar Plot (With Covariates)...")
        self.create_figure06_radar_with_cov()
        
        print("\n" + "="*50)
        print("All figures generated successfully!")
        print(f"Files saved in: {self.output_dir}")
        print("\nGenerated files:")
        print("- Figure01_Baseline_Comparison.pdf")
        print("- Figure02_Data_Leakage_Analysis.pdf")
        print("- Figure03_Multilabel_Framework.pdf")
        print("- Figure04_Multiscale_Architecture.pdf")
        print("- Figure05_Radar_Comparison_No_Covariates.pdf")
        print("- Figure06_Radar_Comparison_With_Covariates.pdf")

# Main execution
if __name__ == "__main__":
    # File paths
    excel_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Nature/results_nature.xlsx'
    output_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Nature'
    
    # Initialize visualizer
    print("Initializing Nature GWAS Visualizer...")
    visualizer = NatureGWASVisualizer(excel_path, output_dir)
    
    # Generate all figures
    visualizer.generate_all_figures()
    
    print("\n All visualizations completed successfully!")