# Define variables

bs = ["32"]
dropout = ["0.5"]

epochs = ["100"]

lr = ["0.001","0.005"]
peak_lr = ["1e-2","5e-2"]
final_lr =["1e-5","5e-5"]

act = ["tanh","relu"]
opt = ["adamw"]
sch = ["plateau","warmup_exponential","exponential_decay"]
wd = ["0.5"]
df =["0.1"]

conv_channels = ["64,128,256","64,128,256","64,128,256","64,128,256"]
fc_layers = ["4096,128","2048,128","1024,64","512,64"]



# Generate the HTCondor submit file content
submit_content = """
##################
#
# Example Job for HTCondor
#
####################

# --------------------------------------
# Executable and its arguments
executable    = epic_cnn_model_pros_chr_res.sh

arguments = $(ID) $(bs) $(dropout) $(epochs) $(lr) $(peak_lr) $(final_lr) $(act) $(sch) $(df) $(opt) $(wd) $(oc) $(fc) $(exp_dir) $(genotype_dir) $(phenotype_file) $(snp_file) $(label_col)

# -----------------------------------
# Universe (vanilla, docker)
universe         = vanilla
 
# -------------------------------------
# Event, out, and error logs
log    =/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/exp_residuals/results/pros/chr_wise/logs/c$(Cluster).$(Process).log
output =/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/exp_residuals/results/pros/chr_wise/$(ID)/c$(Cluster).$(Process).out
error  =/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/exp_residuals/results/pros/chr_wise/logs/c$(Cluster).$(Process).error

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
                                for l, plr, flr in zip(lr,peak_lr,final_lr):
                                    for oc, fc in zip(conv_channels, fc_layers):
                                        experiment_count += 1
                                        experiment_id = f"01_{experiment_count}_{b}_{dr}_{e}_{l}_{a}_{s}_{d}_{o}_{w}"
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
oc = {oc}
fc = {fc}
exp_dir =/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/exp_residuals/results/pros/chr_wise/
genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_pruned
phenotype_file =  /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can_res.xlsx
snp_file = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_pros_can_pruned.gen
label_col = pros01_res
queue 1
"""

# Write the submit content to a file
with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/exp_residuals/epic_cnn_model_pros_chr_res.submit_file", "w") as f:
    f.write(submit_content)

print(f"Generated {experiment_count} experiments.")

