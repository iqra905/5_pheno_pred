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
        
        # Method colors 
        self.method_colors = {
            'MLP': '#1f77b4',           # Blue
            'MLP - Chr': '#aec7e8',     # Light Blue
            'CNN': '#ff7f0e',           # Orange
            'CNN - Chr': '#ffbb78',     # Light Orange
            'ML - DeepCombi': '#858176',     
            'ML - GenNet': '#1f77b4',        # Green
            'Disease-wise Singlescale': '#1f77b4',            
            'Disease-wise Multiscale': '#2ca02c',
            'Multilabel Singlescale': '#ff7f0e',             
            'PGS - AI': '#d62728'                
        }

        # Line styles for differentiation
        self.line_styles = {
            'ML - DeepCombi': ':',     
            'ML - GenNet': ':'
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
    
    def create_radar_plot(self, data_dict, title, filename, show_covariates_label=False):
        """Create radar plot with global normalization to max 1.0"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        diseases = list(data_dict[list(data_dict.keys())[0]].keys())
        methods = list(data_dict.keys())
        
        # Set global maximum to 1.0 for normalization
        global_max = 1.0
        plot_radius = 1.0  # Standard radius for the plot
        
        # Calculate the actual maximum value for each disease (for axis labeling)
        disease_max_values = {}
        disease_min_values = {}
        
        for disease in diseases:
            disease_values = [data_dict[method][disease] for method in methods]
            disease_max_values[disease] = max(disease_values)
            disease_min_values[disease] = min(disease_values)
        
        # Calculate angles for each disease
        angles = [5.35-n / float(len(diseases)) * 2 * pi for n in range(len(diseases))]
        angles += angles[:1]  # Complete the circle
        
        # Create normalized data for plotting (all values scaled to global max of 1.0)
        normalized_data = {}
        
        for method in methods:
            normalized_data[method] = []
            for disease in diseases:
                original_value = data_dict[method][disease]
                # Scale to plot radius based on GLOBAL maximum of 1.0
                normalized_value = (original_value / global_max) * plot_radius
                normalized_data[method].append(normalized_value)
            normalized_data[method] += normalized_data[method][:1]  # Complete circle
        
        # Add alternating filled circles for grid background FIRST
        grid_radii = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fill_colors = ['#f8f8f8', '#ffffff']  # Light gray and white alternating
        
        # Create full circle angles for filling
        theta_fill = np.linspace(0, 2*np.pi, 100)
        
        for i, radius in enumerate(grid_radii):
            fill_color = fill_colors[i % 2]
            if i == 0:
                # First circle from center to first radius
                ax.fill_between(theta_fill, 0, radius, color=fill_color, alpha=0.4, zorder=0)
            else:
                # Ring between previous and current radius
                prev_radius = grid_radii[i-1]
                ax.fill_between(theta_fill, prev_radius, radius, color=fill_color, alpha=0.4, zorder=0)
        
        # Plot each method using normalized data
        for method in methods:
            linestyle = self.line_styles.get(method, '-')  # Default to solid if not found
            ax.plot(angles, normalized_data[method], color=self.method_colors[method], linestyle=linestyle,
                    linewidth=3.0, marker='o', markersize=7, label=method, zorder=5)
        
        # Set up the plot
        ax.set_ylim(0.4, plot_radius + 0.04)
        ax.set_rmax(plot_radius + 0.04)

        # Add axis lines and labels for each disease
        for i, (angle, disease) in enumerate(zip(angles[:-1], diseases)):
            max_value = disease_max_values[disease]
            min_value = disease_min_values[disease] 
            
            # Add disease name at the outer edge
            label_radius = plot_radius + 0.15
            angle_text = angle

            # Handle special case for Colon Cancer - display on two lines
            display_disease = disease
            if disease == 'Colon Cancer':
                display_disease = 'Colon\nCancer'

            if disease == 'Pancreatic Cancer' or disease == 'Prostate Cancer':
                label_radius = label_radius 
            
            if disease == 'T2D':
                label_radius = label_radius - 0.05
                angle_text = angle_text + 0.15
                display_disease = 'Type2\nDiabetes'
            if disease == 'Breast Cancer':
                label_radius = label_radius -0.04
            if disease == 'Colon Cancer':
                label_radius = label_radius - 0.04
                angle_text = angle_text - 0.15

            ax.text(angle_text, label_radius, display_disease, fontsize=20, fontweight='bold', 
                    color='black', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                            edgecolor='lightgray', linewidth=0.5))
            
            # Add maximum value at the edge of each axis (actual max value, not normalized)
            max_label_radius = plot_radius + 0.08 
            if disease == 'Breast Cancer':
                max_label_radius = plot_radius + 0.04
            if disease == 'Colon Cancer':
                max_label_radius = plot_radius + 0.08

            ax.text(angle, max_label_radius, f'{max_value:.2f}', 
                fontsize=20, fontweight='bold', ha='center', va='center',
                color=self.method_colors[method],  
                )
        
        # Updated grid labels - only show 0.4, 0.6, 0.8, and 1.0
        grid_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        grid_labels = []
        for val in grid_values:
            if val in [0.4, 0.6, 0.8, 1.0]:
                grid_labels.append(f'{val:.1f}')
            else:
                grid_labels.append('')  # Empty string for values we don't want to show
        
        # Remove default ticks and labels but ENABLE GRID with custom styling
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Enable grid with careful styling to not interfere with existing elements
        ax.grid(True, linestyle='-', linewidth=0.9, color='black')
        ax.set_rgrids(grid_radii, labels=grid_labels, angle=0, fontsize=15, fontweight='bold', color='black')
        
        # Keep polar spine invisible as before
        ax.spines['polar'].set_visible(False)
        
        # Enhanced legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, 
                fontsize=15, frameon=False, columnspacing=3.5)
        
        # ADD RADIAL LINES AT THE VERY END to ensure they're visible
        for i, (angle, disease) in enumerate(zip(angles[:-1], diseases)):
            # Draw visible radial line from center to outer edge, but behind data points
            ax.plot([angle, angle], [0.4, plot_radius + 0.02], color='black', 
                linewidth=0.9, zorder=4)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10)
        plt.savefig(os.path.join(self.output_dir, filename), format='pdf', 
                bbox_inches='tight', dpi=600, pad_inches=0.1)
        
    def create_figure05_radar_no_cov(self):
        """Figure 5: Radar plot comparison without covariates"""
        df = self.data['Comparison_no_cov']
        diseases = df['Disease'].tolist()
        
        data_dict = {
            #'ML - DeepCombi': {},
            'ML - GenNet': {},
            'PGS - AI': {}
        }
        
        for i, disease in enumerate(diseases):
           # data_dict['ML - DeepCombi'][disease] = df.iloc[i]['DeepCombi']
            data_dict['ML - GenNet'][disease] = df.iloc[i]['GenNet']
            data_dict['PGS - AI'][disease] = df.iloc[i]['Multilabel Multiscale']
        
        self.create_radar_plot(data_dict, 
                             'Method Comparison Across Diseases (Without Covariates)', 
                             'fig_05_comparison_no_cov_max_1.pdf')
        print("Figure 5 saved: fig_05_comparison_no_cov_max_1.pdf")
    
    def create_figure06_radar_with_cov(self):
        """Figure 6: Radar plot comparison with covariates"""
        df = self.data['Comparison_cov']
        diseases = df['Disease'].tolist()
        
        data_dict = {
            #'ML - DeepCombi': {},
            'ML - GenNet': {},
            'PGS - AI': {}
        }
        
        for i, disease in enumerate(diseases):
            #data_dict['ML - DeepCombi'][disease] = df.iloc[i]['DeepCombi']
            data_dict['ML - GenNet'][disease] = df.iloc[i]['GenNet']
            data_dict['PGS - AI'][disease] = df.iloc[i]['Multilabel Multiscale']

        self.create_radar_plot(data_dict, 
                             'Method Comparison Across Diseases (With Covariates)', 
                             'fig_06_comparison_cov_max_1.pdf')
        print("Figure 6 saved: fig_06_comparison_cov_max_1.pdf")
    
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
    excel_path = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Nature/results_nature.xlsx'
    output_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Nature/updated_figs_max_1'
    
    # Initialize enhanced visualizer
    print("Initializing Enhanced Nature GWAS Visualizer...")
    visualizer = NatureGWASVisualizer(excel_path, output_dir)
    
    # Generate all enhanced figures
    visualizer.generate_all_figures()
    
    print("\nAll enhanced visualizations completed successfully!")