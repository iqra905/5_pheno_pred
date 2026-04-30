# Define variables

bs = ["32"]
dropout = ["0.5"]

epochs = ["100"]

lr = ["0.001","0.005"]
peak_lr = ["1e-2","5e-2"]
final_lr =["1e-5","5e-5"]

act = ["tanh","relu"]
opt = ["adamw"]
sch = ["warmup_exponential","exponential_decay"]
wd = ["0.5"]
df =["0.1"]

conv_channels = ["64,128,256"]
fc_layers = ["2048,128","1024,64","512,64"]
#fc_layers = ["4096,128","2048,128","1024,64","512,64"]




# Generate the HTCondor submit file content
submit_content = """
##################
#
# Example Job for HTCondor
#
####################

# --------------------------------------
# Executable and its arguments
executable    = epic_cnn_model_5D_chr.sh

arguments = $(ID) $(bs) $(dropout) $(epochs) $(lr) $(peak_lr) $(final_lr) $(act) $(sch) $(df) $(opt) $(wd) $(oc) $(fc) $(exp_dir) $(genotype_dir) $(phenotype_file) $(snp_file) $(cov) $(label_col)

# -----------------------------------
# Universe (vanilla, docker)
universe         = vanilla
 
# -------------------------------------
# Event, out, and error logs
log    = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.2/chr_wise/logs/c$(Cluster).$(Process).log
output = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.2/chr_wise/$(ID)/c$(Cluster).$(Process).out
error  = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.2/chr_wise/logs/c$(Cluster).$(Process).error

should_transfer_files = YES
environment = "mount=$ENV(PWD),/vol/research/fmodal_mmmed/,/vol/vssp/SF_ucdatasets/gwas"

# -------------------------------------
# Requirements for the Job (Requirements are explained in further detail in example09.submit_file)
requirements = (HasStornext) && (Machine != "dwalin.eps.surrey.ac.uk") && (Machine != "fili.eps.surrey.ac.uk") && (Machine != "bifur.eps.surrey.ac.uk") && (Machine != "bofur.eps.surrey.ac.uk") && (Machine != "creative01.eps.surrey.ac.uk") && (Machine != "oin.eps.surrey.ac.uk") && (Machine != "cvsspgpu01.eps.surrey.ac.uk") && (Machine != "cvsspgpu02.eps.surrey.ac.uk") && (Machine != "cvsspgpu03.eps.surrey.ac.uk")
#&& (Machine=="sounds01.eps.surrey.ac.uk")

# Resources
request_GPUs     = 2
request_CPUs     = 2
request_memory = 25G

+CanCheckpoint = true
+GPUMem = 15000
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
                                    for oc in conv_channels:
                                        for fc in fc_layers:
                                            experiment_count += 1
                                            experiment_id = f"{experiment_count}_{b}_{dr}_{e}_{l}_{a}_{s}_{d}_{o}_{w}"
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
exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.2/chr_wise/
genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_0.2
phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/t2d.xlsx
snp_file = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d_0.2.gen
cov = 1
label_col = t2dm
queue 1
"""

# Write the submit content to a file
with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/epic_cnn_model_t2d_chr_0.2.submit_file", "w") as f:
    f.write(submit_content)

print(f"Generated {experiment_count} experiments.")

