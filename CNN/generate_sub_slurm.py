#!/usr/bin/env python3

# Define hyperparameter search space
bs = ["32"]
dropout = ["0.5"]
epochs = ["70"]
lr = ["0.001"]
peak_lr = ["1e-2"]
final_lr = ["1e-5"]
act = ["gelu"]
opt = ["adamw"]
sch = ["exponential_decay"]
wd = ["0.5"]
df = ["0.1"]

# Architecture parameters

# Model 4
kernel_sizes = ["127,31,7"]
stride = ["64,16,4"]
conv_channels = ["64,128,256"]
fc_layers = ["128,64"]


# kernel_sizes = ["127,31,7,1", "127,31,7,1", "127,31,7,1", "127,31,7,1"]
# stride = ["64,16,4,1", "64,16,4,1", "64,16,4,1", "64,16,4,1"]
# conv_channels = ["64,128,256,128", "64,128,256,64", "64,128,256,32", "64,128,256,16"]
# fc_layers = ["128,64", "128,128", "64,64", "64,32", "256,64"]

# Model 5
# kernel_sizes = ["128,64,4,1", "128,64,4,1", "128,64,4,1", "128,64,4,1"]
# stride = ["128,64,4,1", "128,64,4,1", "128,64,4,1", "128,64,4,1"]
# conv_channels = ["64,128,256,128", "64,128,256,64", "64,128,256,32", "64,128,256,16"]
# fc_layers = ["128,64", "128,128", "64,64", "64,32", "256,64"]

# # Model 6
# kernel_sizes = ["256,16,4,1", "256,16,4,1", "256,16,4,1", "256,16,4,1"]
# stride = ["128,8,2,1", "128,8,2,1", "128,8,2,1", "128,8,2,1"]
# conv_channels = ["64,128,256,128", "64,128,256,64", "64,128,256,32", "64,128,256,16"]
# fc_layers = ["128,64", "128,128", "64,64", "64,32", "256,64"]

use_pooling = ["0"]
pool_size = ["64"]
use_multi_scale = ["0"]

# Paths (update these to match your setup)
exp_dir = "/mnt/fast/nobackup/users/if00208/5_disease_experiments/CNN/results/5d_multilabel/m4/with_cov"
genotype_dir = "/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy"
phenotype_file = "/mnt/fast/datasets/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx"

# Other parameters
cov_settings = ["0", "1"]  # Without and with covariates
use_age = ["0", "1"]
use_gender = ["0", "1"]

norm_pcs = ["standard", "minmax"]
norm_age = ["standard", "minmax"]
norm_gender = ["standard", "minmax"]

# Job configuration
job_name = "cov_5d_multilabel"
partition = "3090"  # Adjust to your partition
time_limit = "03-00:00:00"  # 3 days
memory = "64G"
gpus = 1
ntasks_per_node = 4

conda_env = "/mnt/fast/nobackup/users/if00208/miniconda3/bin/activate"
env_name = "env_EPIC"

#********************************** Function to generate commands with covariate settings **********************************#
def generate_slurm_script():
    """Generate SLURM array job script with hyperparameter sweep"""
    
    # Generate all experiment combinations
    experiments = []
    experiment_count = 0
    
    for b in bs:
        for dr in dropout:
            for e in epochs:
                for a in act:
                    for s in sch:
                        for d in df:
                            for o in opt:
                                for w in wd:
                                    for l, plr, flr in zip(lr, peak_lr, final_lr):
                                        for ks, oc, st in zip(kernel_sizes, conv_channels, stride):
                                            for fc in fc_layers:
                                                for pool in use_pooling:
                                                    for ps in pool_size:
                                                        for ms in use_multi_scale:
                                                            for cov in cov_settings:
                                                                for ua in use_age:
                                                                    for ug in use_gender:
                                                                        # If no covariates are used, only one experiment with 'none' normalization
                                                                        if cov == "0" and ua == "0" and ug == "0":
                                                                            experiment_count += 1
                                                                        
                                                                            # Generate experiment ID
                                                                            experiment_id = f"{experiment_count:02d}_snps_5d_{a}_{s}_ks_{ks.replace(',','_')}_fc_{fc.replace(',','_')}_conv_channels_{oc.replace(',','_')}_no_pool_pc_{cov}_age_{ua}_gender_{ug}_none_none_none"
                                                                            
                                                                            # Create command
                                                                            command = f"""python epic_cnn_model_5D_multilabel_chkpt_npy_multiscale.py \\
-ID "{experiment_id}" \\
-bs {b} \\
-dropout {dr} \\
-epochs {e} \\
-lr {l} \\
-peak_lr {plr} \\
-final_lr {flr} \\
-act {a} \\
-sch {s} \\
-df {d} \\
-opt {o} \\
-wd {w} \\
-kernel_sizes {ks} \\
-conv_channels {oc} \\
-stride {st} \\
-fc_layers {fc} \\
-use_pooling {pool} \\
-pool_size {ps} \\
-use_multi_scale {ms} \\
-exp_dir "{exp_dir}" \\
-genotype_dir "{genotype_dir}" \\
-phenotype_file "{phenotype_file}" \\
-cov {cov} \\
-use_age {ua} \\
-use_gender {ug} \\
-norm_pcs {"none"} \\
-norm_age {"none"} \\
-norm_gender {"none"}"""
                                                                            experiments.append({'id': experiment_id,'command': command})
                                                                            continue
                                                                       
                                                                        # For covariates that are used, try their normalization options
                                                                        curr_norm_pcs = ["none"] if cov == "0" else norm_pcs
                                                                        curr_norm_age = ["none"] if ua == "0" else norm_age
                                                                        curr_norm_gender = ["none"] if ug == "0" else norm_gender
                                                                        for npc in curr_norm_pcs:
                                                                            for na in curr_norm_age:
                                                                                for ng in curr_norm_gender:
                                                                                    experiment_count += 1
                                                                                    # Generate experiment ID
                                                                                    experiment_id = f"{experiment_count:02d}_snps_5d_{a}_{s}_ks_{ks.replace(',','_')}_fc_{fc.replace(',','_')}_conv_channels_{oc.replace(',','_')}_no_pool_pc_{cov}_age_{ua}_gender_{ug}_{npc}_{na}_{ng}"
                                                                                    
                                                                                    # Create command
                                                                                    command = f"""python epic_cnn_model_5D_multilabel_chkpt_npy_multiscale.py \\
-ID "{experiment_id}" \\
-bs {b} \\
-dropout {dr} \\
-epochs {e} \\
-lr {l} \\
-peak_lr {plr} \\
-final_lr {flr} \\
-act {a} \\
-sch {s} \\
-df {d} \\
-opt {o} \\
-wd {w} \\
-kernel_sizes {ks} \\
-conv_channels {oc} \\
-stride {st} \\
-fc_layers {fc} \\
-use_pooling {pool} \\
-pool_size {ps} \\
-use_multi_scale {ms} \\
-exp_dir "{exp_dir}" \\
-genotype_dir "{genotype_dir}" \\
-phenotype_file "{phenotype_file}" \\
-cov {cov} \\
-use_age {ua} \\
-use_gender {ug} \\
-norm_pcs {npc} \\
-norm_age {na} \\
-norm_gender {ng}"""
                                                                                    experiments.append({'id': experiment_id,'command': command})
                                                                        
            
    # Generate SLURM script content
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --array=0-{len(experiments)-1}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --mem={memory}
#SBATCH --partition={partition}
#SBATCH --gpus={gpus}
#SBATCH --time={time_limit}
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source {conda_env}
conda activate {env_name}

COMMANDS=("""
    
    # Add all commands to the array
    for i, exp in enumerate(experiments):
        slurm_content += f"\n'{exp['command']}'"
    
    slurm_content += """
)

# Get the current command
CURRENT_COMMAND="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

# Extract the experiment ID from the command
EXPERIMENT_ID=$(echo "$CURRENT_COMMAND" | grep -o '\\
    -ID "[^"]*"' | sed 's/-ID "\\
    (.*\\
    )"/\\
    1/')

# Create output directory for this experiment
OUTPUT_DIR="results/5d_multilabel/m4/with_cov/${EXPERIMENT_ID}"
mkdir -p "$OUTPUT_DIR"

# Define output and error file paths
SLURM_OUT="${OUTPUT_DIR}/slurm.${SLURM_NODEID}.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
SLURM_ERR="${OUTPUT_DIR}/slurm.${SLURM_NODEID}.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Execute the command and redirect output to experiment-specific files
eval "$CURRENT_COMMAND" > "$SLURM_OUT" 2> "$SLURM_ERR"
"""
    
    return slurm_content, experiments


# def generate_slurm_script():
#     """Generate SLURM array job script with hyperparameter sweep"""
    
#     # Generate all experiment combinations
#     experiments = []
#     experiment_count = 0
    
#     for b in bs:
#         for dr in dropout:
#             for e in epochs:
#                 for a in act:
#                     for s in sch:
#                         for d in df:
#                             for o in opt:
#                                 for w in wd:
#                                     for l, plr, flr in zip(lr, peak_lr, final_lr):
#                                         for ks, oc, st in zip(kernel_sizes, conv_channels, stride):
#                                             for fc in fc_layers:
#                                                 for pool in use_pooling:
#                                                     for ps in pool_size:
#                                                         for cov in cov_settings:
#                                                             experiment_count += 1
                                                            
#                                                             # Generate experiment ID
#                                                             cov_suffix = "_cov" if cov == 1 else ""
#                                                             #experiment_id = f"{experiment_count:02d}_{a}_{s}_ks_{ks.replace(',','_')}_fc_{fc.replace(',','_')}{cov_suffix}"
#                                                             experiment_id = f"{experiment_count:02d}_snps_5d_{a}_{s}_ks_{ks.replace(',','_')}_fc_{fc.replace(',','_')}_no_pool_conv_channels_{oc.replace(',','_')}{cov_suffix}"
                                                            
#                                                             # Create command
#                                                             command = f"""python epic_cnn_model_5D_multilabel_chkpt_npy_multiscale.py \\

# -ID "{experiment_id}" \\

# -bs {b} \\

# -dropout {dr} \\

# -epochs {e} \\

# -lr {l} \\

# -peak_lr {plr} \\

# -final_lr {flr} \\

# -act {a} \\

# -sch {s} \\

# -df {d} \\

# -opt {o} \\

# -wd {w} \\

# -kernel_sizes {ks} \\

# -conv_channels {oc} \\

# -stride {st} \\

# -fc_layers {fc} \\

# -use_pooling {pool} \\

# -pool_size {ps} \\
# use_multi_scale {ms} \\

# -exp_dir "{exp_dir}" \\

# -genotype_dir "{genotype_dir}" \\

# -phenotype_file "{phenotype_file}" \\

# -cov {cov} \\

# -use_age {cov} \\

# -use_gender {cov}"""
                                                    
#                                                             experiments.append({
#                                                                 'id': experiment_id,
#                                                                 'command': command
#                                                             })
            
#     # Generate SLURM script content
#     slurm_content = f"""#!/bin/bash
# #SBATCH --job-name="{job_name}"
# #SBATCH --array=0-{len(experiments)-1}
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node={ntasks_per_node}
# #SBATCH --mem={memory}
# #SBATCH --partition={partition}
# #SBATCH --gpus={gpus}
# #SBATCH --time={time_limit}
# #SBATCH --output=/dev/null
# #SBATCH --error=/dev/null

# source {conda_env}
# conda activate {env_name}

# COMMANDS=("""
    
#     # Add all commands to the array
#     for i, exp in enumerate(experiments):
#         slurm_content += f"\n'{exp['command']}'"
    
#     slurm_content += """
# )

# # Get the current command
# CURRENT_COMMAND="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

# # Extract the experiment ID from the command
# EXPERIMENT_ID=$(echo "$CURRENT_COMMAND" | grep -o '\\
# -ID "[^"]*"' | sed 's/-ID "\\
# (.*\\
# )"/\\
# 1/')

# # Create output directory for this experiment
# OUTPUT_DIR="results/5d_multilabel/ch_no_pool/m6/${EXPERIMENT_ID}"
# mkdir -p "$OUTPUT_DIR"

# # Define output and error file paths
# SLURM_OUT="${OUTPUT_DIR}/slurm.${SLURM_NODEID}.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
# SLURM_ERR="${OUTPUT_DIR}/slurm.${SLURM_NODEID}.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# # Execute the command and redirect output to experiment-specific files
# eval "$CURRENT_COMMAND" > "$SLURM_OUT" 2> "$SLURM_ERR"
# """
    
#     return slurm_content, experiments

def main():
    """Main function to generate files"""
    print("Generating SLURM hyperparameter sweep...")
    
    # Generate SLURM script and experiment list
    slurm_content, experiments = generate_slurm_script()
    
    # Write SLURM script
    script_filename = "epic_cnn_model_5D_multilabel_npy_expo_no_pool_m4_cov.sub"
    with open(script_filename, "w") as f:
        f.write(slurm_content)
    
    print(f"Generated {len(experiments)} experiments.")
    print(f"SLURM script written to: {script_filename}")
    print(f"Array indices: 0-{len(experiments)-1}")
    print("\nTo submit:")
    print(f"sbatch {script_filename}")

if __name__ == "__main__":
    main()