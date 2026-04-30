# Define variables
bs = ["32"]
dropout = ["0.5"]

epochs = ["100"]

lr = ["0.001","0.005"]
peak_lr = ["1e-2","5e-2"]
final_lr =["1e-5","5e-5"]

act = ["tanh","relu","gelu"]
opt = ["adamw"]
sch = ["warmup_exponential","exponential_decay"]
wd = ["0.5"]
df =["0.1"]

# kernel_sizes = ["4096,128"]
# conv_channels = ["32,128"]
# stride = ["4096,16"]
# fc_layers = ["256,64"]

kernel_sizes = ["4096,1,1","2048,1,1"]
stride = ["2048,1,1","1024,1,1"]
conv_channels = ["64,128,256","64,128,256"]
fc_layers = ["64,128","128,16"]

# kernel_sizes = ["4096,1,1","2048,1,1","1024,1,1","512,1,1"]
# stride = ["2048,1,1","1024,1,1","512,1,1","256,1,1"]
# conv_channels = ["64,128,256","64,128,256","64,128,256","64,128,256"]
# fc_layers = ["64,128","128,16"]

# kernel_sizes = ["4096,32,8","2048,32,8","1024,32,8","512,32,8","256,32,8"]
# stride = ["2048,16,4","1024,16,4","512,16,4","256,16,4","128,16,4"]
# conv_channels = ["64,128,256","64,128,256","64,128,256","64,128,256","64,128,256"]
# fc_layers = ["64,128","128,16","1024,128","1024,128","2048,256"]


# kernel_sizes = ["512,128,64",
#                 "4096,1","2048,1","1024,1","512,1","256,1",
#                 "4096,32,1","2048,32,1","1024,32,1","512,32,1","256,32,1"]
# stride = ["32,16,8",
#           "4096,1","2048,1","1024,1","512,1","256,1",
#           "4096,8,1","2048,8,1","1024,8,1","512,8,1","256,8,1"]
# conv_channels = ["128,128,128",
#                  "16,32","16,32","16,32","16,32","16,32",
#                  "16,32,64","16,32,64","16,32,64","16,32,64","16,32,64"]
# fc_layers = ["1024,128",
#              "1024,128","2048,512,128","2048,512,128","4096,1024,128","4096,1024,128",
#              "256,128","512,128","512,128","1024,128","2048,512,128"]

# Generate the HTCondor submit file content
submit_content = """
##################
#
# Example Job for HTCondor
#
####################

# --------------------------------------
# Executable and its arguments
#executable    = epic_cnn_model_5D.sh
executable    = epic_cnn_model_5D_skip.sh

arguments = $(ID) $(bs) $(dropout) $(epochs) $(lr) $(peak_lr) $(final_lr) $(act) $(sch) $(df) $(opt) $(wd) $(ks) $(oc) $(st) $(fc) $(exp_dir) $(genotype_dir) $(phenotype_file) $(cov) $(label_col)

# -----------------------------------
# Universe (vanilla, docker)
universe         = vanilla
 
# -------------------------------------
# Event, out, and error logs
log    = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/brea/brea_0.2/residual/logs/c$(Cluster).$(Process).log
output = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/brea/brea_0.2/residual/$(ID)/c$(Cluster).$(Process).out
error  = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/brea/brea_0.2/residual/logs/c$(Cluster).$(Process).error

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
                                    for ks, oc, st in zip(kernel_sizes, conv_channels, stride):
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
ks = {ks}
oc = {oc}
st = {st}
fc = {fc}
exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/brea/brea_0.2/residual/
genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/brea_can_0.2
phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/brea_can.xlsx
cov = 1
label_col = breacancer
queue 1
"""

# Write the submit content to a file
with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/epic_cnn_model_brea_skip_0.2.submit_file", "w") as f:
    f.write(submit_content)

print(f"Generated {experiment_count} experiments.")

