import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set style for Nature Communications compliance
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_multiscale_architecture_diagram():
    """Create a comprehensive multiscale architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define colors for different components
    colors = {
        'input': '#E8F4FD',           # Light blue
        'short_range': '#FFF2CC',     # Light yellow  
        'medium_range': '#FFE6CC',    # Light orange
        'long_range': '#FFCCCC',      # Light red
        'fusion': '#E1D5E7',          # Light purple
        'dense': '#D5E8D4',           # Light green
        'output': '#F8CECC'           # Light pink
    }
    
    # Input Layer
    input_box = FancyBboxPatch((1, 9), 14, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(8, 9.75, 'Input: 5M SNPs × 37K Samples\n(Genomic Sequence Data)', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Branch Labels
    branch_y = 7.5
    ax.text(3, branch_y, 'Short-Range\nBranch', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#B8860B')
    ax.text(8, branch_y, 'Medium-Range\nBranch', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#FF8C00')
    ax.text(13, branch_y, 'Long-Range\nBranch', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#DC143C')
    
    # Three Parallel Branches
    branch_configs = [
        {'x': 1, 'color': colors['short_range'], 'kernels': 'Kernels: 3, 5', 
         'scale': 'Local LD\n(1-10 kb)', 'features': 'Fine-grained\nSNP interactions'},
        {'x': 6, 'color': colors['medium_range'], 'kernels': 'Kernels: 7, 11', 
         'scale': 'Gene-level\n(10-100 kb)', 'features': 'Regulatory\npatterns'},
        {'x': 11, 'color': colors['long_range'], 'kernels': 'Kernels: 15, 21', 
         'scale': 'Regulatory\n(100kb-1Mb)', 'features': 'Chromosomal\ndomains'}
    ]
    
    # Draw branches
    for i, config in enumerate(branch_configs):
        # Conv Layer 1
        conv1 = FancyBboxPatch((config['x'], 6), 4, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=config['color'], 
                               edgecolor='black', linewidth=1.5)
        ax.add_patch(conv1)
        ax.text(config['x'] + 2, 6.4, f"Conv1D\n{config['kernels'].split(',')[0]}", 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Conv Layer 2  
        conv2 = FancyBboxPatch((config['x'], 5), 4, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=config['color'], 
                               edgecolor='black', linewidth=1.5)
        ax.add_patch(conv2)
        ax.text(config['x'] + 2, 5.4, f"Conv1D\n{config['kernels'].split(',')[1]}", 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Pooling Layer
        pool = FancyBboxPatch((config['x'] + 0.5, 4), 3, 0.6, 
                              boxstyle="round,pad=0.05", 
                              facecolor=config['color'], 
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(pool)
        ax.text(config['x'] + 2, 4.3, "MaxPool1D", 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Scale Information
        scale_box = FancyBboxPatch((config['x'], 2.8), 4, 0.8, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='white', 
                                   edgecolor=config['color'], linewidth=2)
        ax.add_patch(scale_box)
        ax.text(config['x'] + 2, 3.2, config['scale'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Feature Information
        feature_box = FancyBboxPatch((config['x'], 1.8), 4, 0.8, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor='white', 
                                     edgecolor=config['color'], linewidth=2)
        ax.add_patch(feature_box)
        ax.text(config['x'] + 2, 2.2, config['features'], 
                ha='center', va='center', fontsize=9)
        
        # Arrows from input to branches
        arrow = ConnectionPatch((8, 9), (config['x'] + 2, 6.8), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc="black", lw=2)
        ax.add_artist(arrow)
        
        # Arrows from branches to fusion
        arrow_to_fusion = ConnectionPatch((config['x'] + 2, 1.8), (8, 0.8), "data", "data",
                                         arrowstyle="->", shrinkA=5, shrinkB=5, 
                                         mutation_scale=20, fc="black", lw=2)
        ax.add_artist(arrow_to_fusion)
    
    # Feature Fusion Layer
    fusion_box = FancyBboxPatch((4, 0.2), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['fusion'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(8, 0.7, 'Feature Fusion (Concatenation)\nMulti-Scale Representation', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Dense Layers
    dense1 = FancyBboxPatch((5, -1.2), 6, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['dense'], 
                            edgecolor='black', linewidth=1.5)
    ax.add_patch(dense1)
    ax.text(8, -0.8, 'Dense Layer (512 units)\nReLU + Dropout', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    dense2 = FancyBboxPatch((5.5, -2.2), 5, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['dense'], 
                            edgecolor='black', linewidth=1.5)
    ax.add_patch(dense2)
    ax.text(8, -1.8, 'Dense Layer (256 units)\nReLU + Dropout', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Output Layer
    output_box = FancyBboxPatch((3, -3.5), 10, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8, -3, 'Multi-Disease Classification\n5 Output Nodes (Sigmoid Activation)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Disease outputs
    diseases = ['Prostate\nCancer', 'Pancreatic\nCancer', 'Colon\nCancer', 'Breast\nCancer', 'T2D']
    disease_colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9467bd']
    
    for i, (disease, color) in enumerate(zip(diseases, disease_colors)):
        disease_box = FancyBboxPatch((1.5 + i*2.8, -5), 2.5, 0.8, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor=color, alpha=0.3,
                                     edgecolor=color, linewidth=2)
        ax.add_patch(disease_box)
        ax.text(2.75 + i*2.8, -4.6, disease, 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrow from output to disease
        arrow_disease = ConnectionPatch((8, -3.5), (2.75 + i*2.8, -5), "data", "data",
                                       arrowstyle="->", shrinkA=5, shrinkB=5, 
                                       mutation_scale=15, fc=color, lw=2)
        ax.add_artist(arrow_disease)
    
    # Arrows between dense layers
    arrow_dense = ConnectionPatch((8, -1.2), (8, -2.2), "data", "data",
                                 arrowstyle="->", shrinkA=5, shrinkB=5, 
                                 mutation_scale=20, fc="black", lw=2)
    ax.add_artist(arrow_dense)
    
    arrow_output = ConnectionPatch((8, -2.2), (8, -3.5), "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5, 
                                  mutation_scale=20, fc="black", lw=2)
    ax.add_artist(arrow_output)
    
    # Add title
    ax.text(8, 11, 'Multi-Scale Convolutional Architecture for GWAS', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Add biological scale annotations
    ax.text(0.2, 3.2, 'Biological\nScales:', ha='left', va='center', 
            fontsize=11, fontweight='bold', style='italic')
    
    # Add kernel size annotations
    ax.text(0.2, 6.4, 'Kernel\nSizes:', ha='left', va='center', 
            fontsize=11, fontweight='bold', style='italic')
    
    # Set axis properties
    ax.set_xlim(0, 16)
    ax.set_ylim(-6, 12)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_detailed_branch_diagram():
    """Create a detailed view of one branch"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Input sequence representation
    ax.text(1, 7, 'Input SNP Sequence:', fontsize=12, fontweight='bold')
    
    # Draw SNP sequence as boxes
    snp_positions = np.linspace(1, 12, 50)
    for i, pos in enumerate(snp_positions[:20]):  # Show first 20 SNPs
        color = ['#FF6B6B', '#4ECDC4', '#45B7D1'][i % 3]  # Different SNP types
        rect = patches.Rectangle((pos, 6), 0.2, 0.5, 
                               facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
    
    ax.text(6.5, 5.5, '... 5 Million SNPs ...', ha='center', fontsize=10, style='italic')
    
    # Convolutional operations for different scales
    scales = [
        {'name': 'Short-Range (k=3)', 'y': 4.5, 'kernel_size': 3, 'color': '#FFF2CC'},
        {'name': 'Medium-Range (k=7)', 'y': 3.5, 'kernel_size': 7, 'color': '#FFE6CC'},
        {'name': 'Long-Range (k=15)', 'y': 2.5, 'kernel_size': 15, 'color': '#FFCCCC'}
    ]
    
    for scale in scales:
        # Label
        ax.text(0.5, scale['y'], scale['name'], fontsize=11, fontweight='bold')
        
        # Kernel window
        kernel_width = scale['kernel_size'] * 0.2
        kernel_rect = patches.Rectangle((2, scale['y']-0.1), kernel_width, 0.2, 
                                      facecolor=scale['color'], edgecolor='black', linewidth=2,
                                      alpha=0.8)
        ax.add_patch(kernel_rect)
        
        # Arrow showing convolution
        arrow = patches.FancyArrowPatch((2 + kernel_width + 0.2, scale['y']), 
                                      (8, scale['y']),
                                      arrowstyle='->', mutation_scale=20, 
                                      color='black', linewidth=2)
        ax.add_patch(arrow)
        
        # Feature map
        feature_rect = patches.Rectangle((8.5, scale['y']-0.15), 3, 0.3, 
                                       facecolor=scale['color'], edgecolor='black', linewidth=1,
                                       alpha=0.6)
        ax.add_patch(feature_rect)
        ax.text(10, scale['y'], 'Feature Map', ha='center', va='center', fontsize=9)
    
    # Biological interpretation
    ax.text(0.5, 1.5, 'Biological Interpretation:', fontsize=12, fontweight='bold')
    interpretations = [
        'Short-Range: Local LD patterns, SNP-SNP interactions',
        'Medium-Range: Gene-level regulation, moderate interactions', 
        'Long-Range: Chromosomal domains, distant regulatory elements'
    ]
    
    for i, interp in enumerate(interpretations):
        ax.text(1, 1.2 - i*0.3, f'• {interp}', fontsize=10)
    
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Multi-Scale Convolution Detail View', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def save_architecture_diagrams(output_dir):
    """Save both diagrams"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Main architecture diagram
    fig1 = create_multiscale_architecture_diagram()
    fig1.savefig(os.path.join(output_dir, 'Multiscale_Architecture_Main.pdf'), 
                format='pdf', bbox_inches='tight', dpi=300)
    fig1.savefig(os.path.join(output_dir, 'Multiscale_Architecture_Main.png'), 
                format='png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Detailed branch diagram
    fig2 = create_detailed_branch_diagram()
    fig2.savefig(os.path.join(output_dir, 'Multiscale_Branch_Detail.pdf'), 
                format='pdf', bbox_inches='tight', dpi=300)
    fig2.savefig(os.path.join(output_dir, 'Multiscale_Branch_Detail.png'), 
                format='png', bbox_inches='tight', dpi=300)
    plt.show()
    
    print("Architecture diagrams saved!")
    print("Files created:")
    print("- Multiscale_Architecture_Main.pdf/png")
    print("- Multiscale_Branch_Detail.pdf/png")

# Usage
if __name__ == "__main__":
    # Set your output directory
    output_dir = '/vol/research/fmodal_mmmed/Codes/5_disease_experiments/GWAS_Results_Nature'
    
    # Create and save diagrams
    save_architecture_diagrams(output_dir)
    
    # Or create individual diagrams
    # fig1 = create_multiscale_architecture_diagram()
    # fig2 = create_detailed_branch_diagram()
    # plt.show()