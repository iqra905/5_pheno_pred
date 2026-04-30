# Define variables
bs = ["32"]
dropout = ["0.6"]

epochs = ["100"]
# lr = ["0.001","0.005"]
# peak_lr = ["1e-2","5e-2"]
# final_lr =["1e-5","5e-5"]

lr = ["0.001","0.005","0.0001","0.0005","0.00001","0.00005"]
peak_lr = ["1e-2","5e-2","1e-3","5e-3","1e-4","5e-4"]
final_lr =["1e-5","5e-5","1e-6","5e-6","1e-7","5e-7"]

act = ["tanh","relu","gelu"]

#sch = ["none","plateau"]
sch = ["none","plateau","warmup_exponential","exponential_decay"]

df =["0.3"]
opt = ["adamw"]
#opt = ["adamw","adam"]


#wd = ["0.5"]
#wd = ["1","0.9","0.1","0.01"]
wd = ["0.01","0.1","0.5","0.9"]



#hidden_sizes = ["16384,512,64","8192,256,32","4096,256,32"]

hidden_sizes = ["128,128,128","256,64,32","128,64,32","64,64,64"]

# Generate the HTCondor submit file content
submit_content = """
##################
#
# Example Job for HTCondor
#
####################

# --------------------------------------
# Executable and its arguments
executable    = epic_mlp_model_pros_chr_lrp_res.sh
#executable    = epic_mlp_model_pros_res.sh

arguments = $(ID) $(bs) $(dropout) $(epochs) $(lr) $(peak_lr) $(final_lr) $(act) $(sch) $(df) $(opt) $(wd) $(hidden_sizes) $(exp_dir) $(genotype_dir) $(phenotype_file) $(chr_file) $(label_col)
#arguments = $(ID) $(bs) $(dropout) $(epochs) $(lr) $(peak_lr) $(final_lr) $(act) $(sch) $(df) $(opt) $(wd) $(hidden_sizes) $(exp_dir) $(genotype_dir) $(phenotype_file) $(label_col)

# -----------------------------------
# Universe (vanilla, docker)
universe         = vanilla
 
# -------------------------------------
# Event, out, and error logs
log    = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/results/pros/chr_wise/logs/c$(Cluster).$(Process).log
output = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/results/pros/chr_wise/$(ID)/c$(Cluster).$(Process).out
error  = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/results/pros/chr_wise/logs/c$(Cluster).$(Process).error

should_transfer_files = YES
environment = "mount=$ENV(PWD),/vol/research/fmodal_mmmed/,/vol/vssp/SF_ucdatasets/gwas"

# -------------------------------------
# Requirements for the Job (Requirements are explained in further detail in example09.submit_file)

requirements = (HasStornext) && (Machine != "dwalin.eps.surrey.ac.uk") && (Machine != "fili.eps.surrey.ac.uk") && (Machine != "bofur.eps.surrey.ac.uk") && (Machine != "creative01.eps.surrey.ac.uk") && (Machine != "oin.eps.surrey.ac.uk") && (Machine != "cvsspgpu01.eps.surrey.ac.uk") && (Machine != "cvsspgpu02.eps.surrey.ac.uk") && (Machine != "cvsspgpu03.eps.surrey.ac.uk")
#&& (Machine=="sounds01.eps.surrey.ac.uk")

# Resources
request_GPUs     = 1
request_CPUs     = 4
request_memory = 20G

+CanCheckpoint = true
+GPUMem = 11000
+JobRunTime = 72

#-------------------
"""

# Generate queue statements for each combination of batch_size and lr
experiment_count = 0
for b in bs:
    for dr in dropout:      
        for e in epochs:
            for a in act:
                for s in sch:
                    for d in df:
                        for o in opt:
                            for w in wd:
                                for hs in hidden_sizes:
                                    for l, plr, flr in zip(lr,peak_lr,final_lr):
                                        experiment_count += 1
                                        experiment_id = f"4_{experiment_count}_{b}_{dr}_{e}_{l}_{a}_{s}_{d}_{o}_{w}"
                                        submit_content += f"""
ID = {experiment_id}
bs = {b}
dropout = {dr}
epochs = {e}
lr = {l}
peak_lr = {plr}
final_lr = {flr}
act = {a}
sch = {s}
df = {d}
opt = {o}
wd = {w}
hidden_sizes = {hs}
exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/results/pros/chr_wise/
genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_pruned
phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can_res.xlsx
chr_file = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_pros_can_pruned.gen.gz
label_col = pros01_res
queue 1
"""
#chr_file = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_pros_0.1.gen.gz

# exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/pros_0.1/chr_wise/
# genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_0.1
# phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros.xlsx
#chr_file = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_pros_can_pruned.gen.gz


# Write the submit content to a file
with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/epic_mlp_model_pros_chr_lrp_res.submit_file", "w") as f:
#with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/exp_residuals/epic_mlp_model_pros_res.submit_file", "w") as f:

    f.write(submit_content)

print(f"Generated {experiment_count} experiments.")

