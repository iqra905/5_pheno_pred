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

kernel_sizes = ["4096,1,1"]
stride = ["2048,1,1"]
conv_channels = ["64,128,384"]

num_transformer_layers = ["2","4"]
nhead = ["4","6",]
d_model = ["384"]
fc_layers = ["64,128"]

pooling = ["mean","cls"]


# Generate the HTCondor submit file content
submit_content = """
##################
#
# Example Job for HTCondor
#
####################

# --------------------------------------
# Executable and its arguments
executable    = epic_cnn_model_5D_transformer_scratch.sh

arguments = $(ID) $(exp_dir) $(genotype_dir) $(phenotype_file) $(num_transformer_layers) $(nhead) $(d_model) $(fc_layers) $(pooling) $(label_col) $(bs) $(dropout) $(epochs) $(act) $(opt) $(sch) $(wd) $(df) $(lr) $(peak_lr) $(final_lr) $(kernel_sizes) $(stride) $(conv_channels)

# -----------------------------------
# Universe (vanilla, docker)
universe         = vanilla
 
# -------------------------------------
# Event, out, and error logs
log    = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/Transformer/results_scratch/pros/full/logs/c$(Cluster).$(Process).log
output = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/Transformer/results_scratch/pros/full/$(ID)/c$(Cluster).$(Process).out
error  = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/Transformer/results_scratch/pros/full/logs/c$(Cluster).$(Process).error

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
+GPUMem = 11000
+JobRunTime = 72

#-------------------
"""

# Generate queue statements for each combination of batch_size and lr
experiment_count = 0
for nlayers in num_transformer_layers:
    for heads in nhead:
        for dim in d_model:
            for fc in fc_layers:
                for pool in pooling:
                    for b in bs:
                        for d in dropout:      
                            for e in epochs:
                                for a in act:
                                    for o in opt:
                                        for s in sch:
                                            for w in wd:
                                                for d in df:
                                                    for l, plr, flr in zip(lr,peak_lr,final_lr):
                                                        for ks, st, oc in zip(kernel_sizes, stride, conv_channels):
                                                            experiment_count += 1
                                                            experiment_id = f"{experiment_count}_nlayers_{nlayers}_nhead{heads}_dmodel_{dim}_pool_{pool}_{b}_{d}_{e}_{a}_{o}_{s}_{w}_{d}_{l}"
                                                            submit_content += f"""
ID = {experiment_id}
exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/Transformer/results_scratch/pros/full/
genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can
phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx
num_transformer_layers = {nlayers}
nhead = {heads}
d_model = {dim}
fc_layers = {fc}
pooling = {pool}
label_col = pros01
bs = {b}
dropout = {d}
epochs = {e}
act = {a}
opt = {o}
sch = {s}
wd = {w}
df = {d}
lr = {l}
peak_lr = {plr}
final_lr = {flr}
kernel_sizes = {ks}
stride = {st}
conv_channels = {oc}
queue 1
"""

# Write the submit content to a file
with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/epic_cnn_model_5D_transformer_scratch_pros.submit_file", "w") as f:
    f.write(submit_content)

print(f"Generated {experiment_count} experiments.")

