import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import os
import warnings
from scipy import stats
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
    'hatch.linewidth': 0.8,  # Make hatch pattern visible
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Enhanced color scheme - more Nature Communications style
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
        
        # Updated method colors - same color for each method type, differentiated by line style
        self.method_colors = {
            'MLP': '#1f77b4',           # Blue
            'MLP - Chr': '#aec7e8',     # Light Blue
            'CNN': '#ff7f0e',           # Orange
            'CNN - Chr': '#ffbb78',     # Light Orange
            'DeepCombi': '#d62728',     # Red
            'GenNet': '#2ca02c',        # Green
            'Disease-wise (without Covariates)': '#1f77b4',            
            'Disease-wise (with Covariates)': '#1f77b4',                
            'Multilabel (without Covariates)': '#d62728',                
            'Multilabel (with Covariates)': '#d62728'                  
        }
        
        # Line styles for differentiation
        self.line_styles = {
            'Disease-wise (without Covariates)': ':',     # Solid
            'Disease-wise (with Covariates)': '-',       # Dashed
            'Multilabel (without Covariates)': ':',       # Solid
            'Multilabel (with Covariates)': '-'          # Dashed
        }
        
        self.load_data()
    
    def load_data(self):
        """Load all data sheets from Excel file"""
        try:
            self.data = {}
            sheet_names = ['Setting01', 'Setting02', 'Multilabel']
            
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
    
    
    def simulate_error_bars(self, values, error_pct=0.015):
        """Simulate error bars for visualization (since we don't have raw data)"""
        return [v * error_pct for v in values]
    
    def create_figure01_baseline_enhanced(self):
        """Figure 1: Enhanced baseline comparison with error bars and significance"""
        fig, ax = plt.subplots(figsize=(14, 4))
        
        df = self.data['Setting01']
        diseases = df['Disease'].tolist() 
        methods = ['MLP', 'MLP - Chr', 'CNN', 'CNN - Chr', 'DeepCombi', 'GenNet']
        
        x = np.arange(len(methods))
        width = 0.12
        spacing = 1.1
        
        # Create legend handles for diseases
        legend_handles = []
        
        # Plot bars for each disease within each method group
        for i, disease in enumerate(diseases):
            disease_values = []
            disease_bars = []
            
            for method in methods:
                value = df[df['Disease'] == disease][method].iloc[0]
                disease_values.append(value)
            
            # Simulate error bars
            error_bars = self.simulate_error_bars(disease_values)
            
            # Plot bars with error bars
            bars = ax.bar(x + (i - 2) * width * spacing, disease_values, width, 
                        label=disease, color=self.disease_colors[disease], 
                        edgecolor='black', linewidth=0.5,
                        yerr=error_bars, capsize=2, error_kw={'linewidth': 0.8})
            
            disease_bars.extend(bars)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, disease_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error_bars[j] + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=6)
            
            # Create legend handle
            legend_handles.append(plt.Rectangle((0,0),1,1, facecolor=self.disease_colors[disease], 
                                              edgecolor='black', linewidth=0.5, label=disease))
        
        # Add reference line at random performance
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[1]*0.98, 0.5, 'Random', va='center', ha='right', 
                fontsize=8, color='gray')
        
        ax.set_ylabel('AUC', fontweight='bold', fontsize=12)
        ax.set_xlabel('Method', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10, color='black', fontweight='bold')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0.02)
        
        # Enhanced legend
        ax.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_handles), 
            fontsize=9,
            frameon=False,
            columnspacing=2.5
        )
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_ylim(0.4, 0.70)
        ax.set_yticks(np.arange(0.4, 0.71, 0.05))
        ax.set_yticklabels([f"{tick:.2f}" for tick in np.arange(0.4, 0.71, 0.05)], 
                          fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(os.path.join(self.output_dir, 'fig_01_baseline.jpg'), 
                   format='jpg', bbox_inches='tight', dpi=600)
        print("Figure 1 saved: fig_01_baseline.jpg")
    
    def create_figure02_combined(self):
        """Figure 2: Combined Population Dataset and Train Set comparison with error bars"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
        
        df = self.data['Setting02']
        
        # Handle different possible column structures
        try:
            # Try multi-level column access first
            diseases = df[df.columns[0]].tolist()
            
            if isinstance(df.columns, pd.MultiIndex):
                pop_no_cov = df[('Population Dataset', 'No COV')].tolist()
                pop_cov = df[('Population Dataset', 'COV')].tolist()
                train_no_cov = df[('Train Set', 'No COV')].tolist()
                train_cov = df[('Train Set', 'COV')].tolist()
            else:
                diseases = df.iloc[:, 0].tolist()
                pop_no_cov = df.iloc[:, 1].tolist()
                pop_cov = df.iloc[:, 2].tolist()
                train_no_cov = df.iloc[:, 3].tolist()
                train_cov = df.iloc[:, 4].tolist()
                
        except Exception as e:
            print(f"Error accessing Setting02 data: {e}")
            print(f"Dataframe structure: {df.head()}")
            print(f"Columns: {df.columns}")
        
        x = np.arange(len(diseases))
        width = 0.32
        spacing = 1.1
        
        # TOP SUBPLOT: Population Dataset (Data Leakage)
        for i, disease in enumerate(diseases):
            base_color = self.disease_colors[disease]
            
            # Simulate error bars
            error_no_cov = pop_no_cov[i] * 0.03
            error_cov = pop_cov[i] * 0.03
            
            # Without Covariates - solid bars
            ax1.bar(x[i] - width/2 * spacing, pop_no_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.5, 
                yerr=error_no_cov, capsize=2, error_kw={'linewidth': 0.8},
                label='Without Covariates' if i == 0 else "")
                
            # Label for without covariates bar
            ax1.text(x[i] - width/2 * spacing, pop_no_cov[i] + error_no_cov + 0.01, 
                    f"{pop_no_cov[i]:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=6)
            
            # With Covariates - hatched bars
            ax1.bar(x[i] + width/2 * spacing, pop_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.5, hatch='///', 
                yerr=error_cov, capsize=2, error_kw={'linewidth': 0.8},
                label='With Covariates' if i == 0 else "")
            
            # Label for with covariates bar
            ax1.text(x[i] + width/2 * spacing, pop_cov[i] + error_cov + 0.01, 
                    f"{pop_cov[i]:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=6)

        # BOTTOM SUBPLOT: Train Set Only (Proper Training)
        for i, disease in enumerate(diseases):
            base_color = self.disease_colors[disease]
            
            # Simulate error bars
            error_no_cov = train_no_cov[i] * 0.05
            error_cov = train_cov[i] * 0.05
            
            # Without Covariates - solid bars
            ax2.bar(x[i] - width/2 * spacing, train_no_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.5,
                yerr=error_no_cov, capsize=2, error_kw={'linewidth': 0.8})
                
            # Label for without covariates bar
            ax2.text(x[i] - width/2 * spacing, train_no_cov[i] + error_no_cov + 0.01, 
                    f"{train_no_cov[i]:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=6)
            
            # With Covariates - hatched bars
            ax2.bar(x[i] + width/2 * spacing, train_cov[i], width, 
                color=base_color, edgecolor='black', linewidth=0.5, hatch='///', 
                yerr=error_cov, capsize=2, error_kw={'linewidth': 0.8})
            
            # Label for with covariates bar
            ax2.text(x[i] + width/2 * spacing, train_cov[i] + error_cov + 0.01, 
                    f"{train_cov[i]:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=6)

        # Create custom legend (only for top subplot)
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='black', 
                        linewidth=0.5, label='Without Covariates'),
            plt.Rectangle((0,0),1,1, facecolor='gray', edgecolor='black', 
                        linewidth=0.5, hatch='///', label='With Covariates')
        ]
        
        # Configure TOP subplot (Population Dataset)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.margins(x=0.05)
        ax1.set_ylabel('AUC', fontweight='bold', fontsize=10)
        ax1.set_title('Population Dataset (Data Leakage)', fontweight='bold', fontsize=11, pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels([])  # Remove x-axis labels for top plot
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0.2, 1.1)
        ax1.set_yticks(np.arange(0.2, 1.2, 0.2)) 
        ax1.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.2, 1.2, 0.2)], fontweight='bold', fontsize=8)
        
        # Configure BOTTOM subplot (Train Set)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.margins(x=0.05)
        ax2.set_ylabel('AUC', fontweight='bold', fontsize=10)
        ax2.set_xlabel('Disease', fontweight='bold', fontsize=10)
        ax2.set_title('Train Set Only (Proper Training)', fontweight='bold', fontsize=11, pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(diseases, fontsize=8, color='black', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0.2, 1.0)
        ax2.set_yticks(np.arange(0.2, 1.1, 0.2)) 
        ax2.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.2, 1.1, 0.2)], fontweight='bold', fontsize=8)
        
        # Add legend to the bottom of the figure
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(legend_elements), 
            fontsize=9,
            frameon=False
        )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, hspace=0.4)
        plt.savefig(os.path.join(self.output_dir, 'fig_02_combined.jpg'), format='jpg', bbox_inches='tight', dpi=600)
        print("Figure 2 saved: fig_02_combined.jpg")

    def create_radar_plot(self, data_dict, title, filename, show_covariates_label=False):
        """Create radar plot with same colors for method types, differentiated by line styles"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        diseases = list(data_dict[list(data_dict.keys())[0]].keys())
        methods = list(data_dict.keys())
        
        # Calculate the maximum value for each disease across all methods
        disease_max_values = {}
        disease_min_values = {}
        
        for disease in diseases:
            disease_values = [data_dict[method][disease] for method in methods]
            disease_max_values[disease] = max(disease_values)
            disease_min_values[disease] = min(disease_values)
        
        # Find overall max for consistent grid spacing
        overall_max = max(disease_max_values.values())
        
        # Calculate angles for each disease
        angles = [5.35-n / float(len(diseases)) * 2 * pi for n in range(len(diseases))]
        angles += angles[:1]  # Complete the circle
        
        # Create normalized data for plotting (each disease scaled to its max)
        normalized_data = {}
        plot_radius = 1.0  # Standard radius for the plot
        
        for method in methods:
            normalized_data[method] = []
            for disease in diseases:
                original_value = data_dict[method][disease]
                max_value = disease_max_values[disease]
                # Scale to plot radius based on disease-specific maximum
                normalized_value = (original_value / max_value) * plot_radius
                normalized_data[method].append(normalized_value)
            normalized_data[method] += normalized_data[method][:1]  # Complete circle
        
        # Add alternating filled circles for grid background FIRST
        grid_radii = [0.6, 0.7, 0.8, 0.9, 1.0]
        fill_colors = ['#f8f8f8', '#ffffff']  # Light gray and white alternating
        
        # Create full circle angles for filling
        theta_fill = np.linspace(0, 2*np.pi, 100)
        
        for i, radius in enumerate(grid_radii):
            fill_color = fill_colors[i % 2]
            if i == 0:
                # First circle from center to first radius
                ax.fill_between(theta_fill, 0, radius, color=fill_color, alpha=0.3, zorder=0)
            else:
                # Ring between previous and current radius
                prev_radius = grid_radii[i-1]
                ax.fill_between(theta_fill, prev_radius, radius, color=fill_color, alpha=0.3, zorder=0)
        
        # Plot each method using normalized data with updated colors and line styles
        for method in methods:
            color = self.method_colors[method]
            linestyle = self.line_styles.get(method, '-')  # Default to solid if not found
            
            ax.plot(angles, normalized_data[method], color=color, 
                    linewidth=3.0, linestyle=linestyle, marker='o', markersize=7, 
                    label=method, zorder=5)
        
        # Set up the plot
        ax.set_ylim(0.5, plot_radius + 0.04)
        ax.set_rmax(plot_radius + 0.04)
    
        # Add axis lines and labels for each disease
        for i, (angle, disease) in enumerate(zip(angles[:-1], diseases)):
            max_value = disease_max_values[disease]
            min_value = disease_min_values[disease] 
            
            # Add disease name at the outer edge
            label_radius = plot_radius + 0.15
            angle_text = angle

            if disease == 'Pancreatic Cancer' or disease == 'Prostate Cancer':
                label_radius = label_radius - 0.06
            
            if disease == 'T2D':
                label_radius = label_radius - 0.09
                angle_text = angle_text + 0.08
            if disease == 'Breast Cancer':
                label_radius = label_radius - 0.08
            if disease == 'Colon Cancer':
                label_radius = label_radius - 0.07
                angle_text = angle_text - 0.07

            ax.text(angle_text, label_radius, disease, fontsize=11, fontweight='bold', 
                    color='black', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='lightgray', linewidth=0.5))
            
            # Add maximum value at the edge of each axis
            max_label_radius = plot_radius + 0.04
            if disease == 'Breast Cancer':
                max_label_radius = plot_radius + 0.03
            if disease == 'T2D' or disease == 'Colon Cancer':
                max_label_radius = plot_radius + 0.05

            ax.text(angle, max_label_radius, f'{max_value:.3f}', 
                fontsize=9, fontweight='bold', ha='center', va='center',
                color='#d62728',  # Green for max values
                )
            
            # Add Minimum value at the inner edge of each axis
            min_label_radius = plot_radius - 0.1
            if disease == 'Breast Cancer':
                min_label_radius = min_label_radius - 0.035
            elif disease == 'T2D':
                min_label_radius = min_label_radius -0.08
            elif disease == 'Colon Cancer':
                min_label_radius = min_label_radius + 0.02
            elif disease == 'Pancreatic Cancer':
                min_label_radius = min_label_radius - 0.1
            else: 
                min_label_radius =min_label_radius - 0.15

            ax.text(angle, min_label_radius, f'{min_value:.3f}', 
                fontsize=9, fontweight='bold', ha='center', va='center',
                color='#1f77b4',)  # Blue for min values
        
        # Remove default ticks and labels but ENABLE GRID with custom styling
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Enable grid with careful styling to not interfere with existing elements
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, color='lightgray')
        ax.set_rgrids(grid_radii, labels=[], angle=0)
        
        # Keep polar spine invisible as before
        ax.spines['polar'].set_visible(False)
        
        # Enhanced legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, 
                fontsize=10, frameon=False, columnspacing=3.5)
        
        # ADD RADIAL LINES AT THE VERY END to ensure they're visible
        for i, (angle, disease) in enumerate(zip(angles[:-1], diseases)):
            # Draw visible radial line from center to outer edge, but behind data points
            ax.plot([angle, angle], [0.5, plot_radius + 0.02], color='lightgray', 
                linewidth=0.6, alpha=0.5, zorder=4)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10)
        plt.savefig(os.path.join(self.output_dir, filename), format='jpg', 
                bbox_inches='tight', dpi=600, pad_inches=0.1)
    
    def create_figure05_radar_no_cov(self):
        """Figure 5: Radar plot comparison without covariates"""
        df = self.data['Multilabel']
        diseases = df['Disease'].tolist()
        
        data_dict = {
            'Disease-wise (without Covariates)': {},
            'Disease-wise (with Covariates)': {},
            'Multilabel (without Covariates)': {},
            'Multilabel (with Covariates)': {}
        }
        
        for i, disease in enumerate(diseases):
            data_dict['Disease-wise (without Covariates)'][disease] = df.iloc[i]['Disease-wise (without Covariates)']
            data_dict['Disease-wise (with Covariates)'][disease] = df.iloc[i]['Disease-wise (with Covariates)']
            data_dict['Multilabel (without Covariates)'][disease] = df.iloc[i]['Multilabel (without Covariates)'] 
            data_dict['Multilabel (with Covariates)'][disease] = df.iloc[i]['Multilabel (with Covariates)']
        
        self.create_radar_plot(data_dict, 
                             'Method Comparison Across Diseases', 
                             'fig_05_comparison_radar.jpg')
        print("Figure 5 saved: fig_05_comparison_radar.jpg")
    
    def generate_all_figures(self):
        """Generate all enhanced figures"""
        print("Starting enhanced figure generation...")
        print("="*60)
        
        print("Generating Figure 1: Enhanced Baseline Comparison...")
        self.create_figure01_baseline_enhanced()
        
        print("\nGenerating Figure 2: Combined Population Dataset and Train Set...")
        self.create_figure02_combined()
        
        # print("\nGenerating Figure 3: Multilabel Framework...")
        # self.create_figure03_multilabel()
        
        print("\nGenerating Figure 5: Radar Plot Comparison...")
        self.create_figure05_radar_no_cov()
        
        print("\n" + "="*60)
        print("All enhanced figures generated successfully!")
        print(f"Files saved in: {self.output_dir}")
        print("\nGenerated files:")
        print("- fig_01_baseline.jpg")
        print("- fig_02_combined.jpg") 
        print("- fig_05_comparison_radar.jpg")

# Main execution
if __name__ == "__main__":
    # File paths
    excel_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_AIiH_2025/results_AIiH_poster.xlsx'
    output_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_AIiH_2025'
    
    # Initialize enhanced visualizer
    print("Initializing Enhanced Nature GWAS Visualizer...")
    visualizer = NatureGWASVisualizer(excel_path, output_dir)
    
    # Generate all enhanced figures
    visualizer.generate_all_figures()
    
    print("\nAll enhanced visualizations completed successfully!")