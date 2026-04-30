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
    'hatch.linewidth': 0.8,  # Control hatch pattern line width
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


class NatureGWASVisualizer:
    def __init__(self, excel_path, output_dir):
        """Initialize with Excel file path and output directory"""
        self.excel_path = excel_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        
        # Method colors 
        self.method_colors = {
            #'DeepCombi': '#858176',     
            #'GenNet': '#4b0082',        # Green
            'Transformer': '#1F4E9D',            
            'Mamba': '#C8202B'                
        }

        # Line styles for differentiation
        self.line_styles = {
            'Transformer': ':',     
            'GenNet': ':'
        }
        
        self.load_data()
    
    def load_data(self):
        """Load all data sheets from Excel file"""
        try:
            self.data = {}
            sheet_names = ['Comparison_no_cov', 'Comparison_cov']
            
            for sheet in sheet_names:
                self.data[sheet] = pd.read_excel(self.excel_path, sheet_name=sheet)
                print(f"Loaded {sheet}: {self.data[sheet].shape}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def create_radar_plot(self, data_dict, title, filename, show_covariates_label=False):
        """Create radar plot with independent axis scaling like the reference image"""
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
        
        # Define marker styles for each method
        marker_styles = {
            'Transformer': 'o',  # Circle/dot
            'Mamba': 's'         # Square
        }
        
        # Plot each method using normalized data
        for method in methods:
            linestyle = self.line_styles.get(method, '-')  # Default to solid if not found
            marker = marker_styles.get(method, 'o')  # Default to circle if not found
            
            ax.plot(angles, normalized_data[method], color=self.method_colors[method], linestyle=linestyle,
                    linewidth=3.0, marker=marker, markersize=7, label=method, zorder=5)
            
            # Add value labels directly on data points
            for i, (angle, disease) in enumerate(zip(angles[:-1], diseases)):
                original_value = data_dict[method][disease]
                normalized_value = normalized_data[method][i]
                
                # Position label slightly outside the data point
                label_offset = 0.04 if method == 'Mamba' else -0.04

                if title =='Method Comparison Across Diseases (With Covariates)' and disease == 'Colorectal Cancer':
                    label_offset = 0.04 if method == 'Transformer' else -0.04 # Adjust offset for Colorectal Cancer

                label_radius = normalized_value + label_offset
                
                # ax.text(angle, label_radius, f'{original_value:.3f}', 
                #         fontsize=8, fontweight='bold', ha='center', va='center',
                #         color=self.method_colors[method],
                #         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                #                 edgecolor=self.method_colors[method], linewidth=0.5, alpha=0.8),
                #         zorder=6)
                
                ax.text(angle, label_radius, f'{original_value:.3f}', 
                        fontsize=8, fontweight='bold', ha='center', va='center',
                        color=self.method_colors[method],
                        zorder=6)
        
        # Set up the plot
        ax.set_ylim(0.6, plot_radius + 0.05)
        ax.set_rmax(plot_radius + 0.05)

        # Add axis lines and labels for each disease
        for i, (angle, disease) in enumerate(zip(angles[:-1], diseases)):
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
            if disease == 'Colorectal Cancer':
                label_radius = label_radius - 0.03
                angle_text = angle_text - 0.07

            ax.text(angle_text, label_radius, disease, fontsize=11, fontweight='bold', 
                    color='black', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='lightgray', linewidth=0.5))
        
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
            ax.plot([angle, angle], [0.6, plot_radius + 0.02], color='lightgray', 
                linewidth=0.6, alpha=0.5, zorder=4)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10)
        plt.savefig(os.path.join(self.output_dir, filename), format='pdf', 
                bbox_inches='tight', dpi=600, pad_inches=0.1)
    
    def create_figure05_radar_no_cov(self):
        """Figure 5: Radar plot comparison without covariates"""
        df = self.data['Comparison_no_cov']
        diseases = df['Disease'].tolist()
        
        data_dict = {
            #'DeepCombi': {},
            #'GenNet': {},
            'Mamba': {},
            'Transformer': {},
        }
        
        for i, disease in enumerate(diseases):
           # data_dict['DeepCombi'][disease] = df.iloc[i]['DeepCombi']
           # data_dict['GenNet'][disease] = df.iloc[i]['GenNet']
            data_dict['Mamba'][disease] = df.iloc[i]['Mamba']
            data_dict['Transformer'][disease] = df.iloc[i]['Transformer']
        
        self.create_radar_plot(data_dict, 
                             'Method Comparison Across Diseases (Without Covariates)', 
                             'fig_05_comparison_no_cov_max_norm.pdf')
        print("Figure 5 saved: fig_05_comparison_no_cov_max_norm.pdf")
    
    def create_figure06_radar_with_cov(self):
        """Figure 6: Radar plot comparison with covariates"""
        df = self.data['Comparison_cov']
        diseases = df['Disease'].tolist()
        
        data_dict = {
           # 'DeepCombi': {},
           # 'GenNet': {},
            'Mamba': {},
            'Transformer': {},
        }
        
        for i, disease in enumerate(diseases):
           # data_dict['DeepCombi'][disease] = df.iloc[i]['DeepCombi']
          #  data_dict['GenNet'][disease] = df.iloc[i]['GenNet']
            data_dict['Mamba'][disease] = df.iloc[i]['Mamba']
            data_dict['Transformer'][disease] = df.iloc[i]['Transformer']

        self.create_radar_plot(data_dict, 
                             'Method Comparison Across Diseases (With Covariates)', 
                             'fig_06_comparison_cov_max_norm.pdf')
        print("Figure 6 saved: fig_06_comparison_cov_max_norm.pdf")
    
    def generate_all_figures(self):
        """Generate all enhanced figures"""
        print("Starting enhanced figure generation...")
        print("="*60)
        
        print("\nGenerating Figure 5: Radar Plot (No Covariates)...")
        self.create_figure05_radar_no_cov()
        
        print("\nGenerating Figure 6: Radar Plot (With Covariates)...")
        self.create_figure06_radar_with_cov()
        
        print("\n" + "="*60)
        print("All enhanced figures generated successfully!")
        print(f"Files saved in: {self.output_dir}")
        print("\nGenerated files:")
        print("- fig_05_comparison_no_cov.pdf")
        print("- fig_06_comparison_cov.pdf")

# Main execution
if __name__ == "__main__":
    # File paths
    excel_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Mamba/results_mamba.xlsx'
    output_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Mamba/'
    
    # Initialize enhanced visualizer
    print("Initializing Enhanced Nature GWAS Visualizer...")
    visualizer = NatureGWASVisualizer(excel_path, output_dir)
    
    # Generate all enhanced figures
    visualizer.generate_all_figures()
    
    print("\nAll enhanced visualizations completed successfully!")