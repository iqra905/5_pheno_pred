#************************************************************* General *************************************************************************#

# Define variables
#bs = ["32"]
bs = ["64"]

dropout = ["0.5"]

epochs = ["100"]
# lr = ["0.0005"]
# peak_lr = ["5e-3"]
# final_lr =["5e-6"]

# lr = ["0.01","0.05","0.001","0.005","0.0001","0.0005"]
# peak_lr = ["1e-1","5e-1","1e-2","5e-2","1e-3","5e-3"]
# final_lr =["1e-4","5e-4","1e-5","5e-5","1e-6","5e-6"]

# lr = ["0.001","0.005","0.0001","0.0005","0.00001","0.00005"]
# peak_lr = ["1e-2","5e-2","1e-3","5e-3","1e-4","5e-4"]
# final_lr =["5e-4","1e-5","5e-5","1e-6","5e-6","1e-7","5e-7"]

# lr = ["0.001","0.005","0.0001","0.0005"]
# peak_lr = ["1e-2","5e-2","1e-3","5e-3"]
# final_lr =["1e-4","5e-4","1e-6","5e-6"]

lr = ["0.001","0.005"]
peak_lr = ["1e-2","5e-2"]
final_lr =["1e-4","5e-4"]

#act = ["gelu"]
act = ["relu","gelu"]


#sch = ["explr"]
#sch = ["explr","step","multistep","cosine"]
sch = ["warmup_exponential","exponential_decay","plateau","explr","cosine"]
#sch = ["exponential_decay","explr"]



df =["0.1"]
opt = ["adamw"]

#wd = ["0.5"]
#wd = ["0.0001","0.0005","0.001","0.005","0.01","0.05","0.1","0.5","0.7"]
#wd = ["0.05","0.1","0.3","0.5","0.7"]
wd = ["0.1","0.3","0.5","0.7"]





#hidden_sizes = ["16384,512,64","8192,256,32","4096,256,32"]

#hidden_sizes = ["256,64,32"]
#hidden_sizes = ["128,128,128","128,64,32","64,32,16","64,32","64,16"]
#hidden_sizes = ["128,64,32"]
#hidden_sizes = ["128,64,32","64,32,16","64,32","64,16"]
hidden_sizes = ["64,32,16","64,32","64,16"]



# cov = ["0","1"]
# use_age = ["0","1"]
# use_gender = ["0","1"]
cov = ["0"]
use_age = ["0"]
use_gender = ["0"]

# Generate the HTCondor submit file content
submit_content = """
##################
#
# Example Job for HTCondor
#
####################

# --------------------------------------
# Executable and its arguments
executable    = epic_mlp_model_5D_cov_layer_last_ipca_precompute.sh
arguments = $(ID) $(bs) $(dropout) $(epochs) $(lr) $(peak_lr) $(final_lr) $(act) $(sch) $(df) $(opt) $(wd) $(hidden_sizes) $(exp_dir) $(genotype_dir) $(phenotype_file) $(cov) $(use_age) $(use_gender) $(label_col) $(model_type) $(pca_features_dir)

# -----------------------------------

# Universe (vanilla, docker)
universe         = vanilla
 
# -------------------------------------
# Event, out, and error logs
log    = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/logs/c$(Cluster).$(Process).log
output = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/$(ID)/c$(Cluster).$(Process).out
error  = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/logs/c$(Cluster).$(Process).error

should_transfer_files = YES
environment = "mount=$ENV(PWD),/vol/research/fmodal_mmmed/,/vol/vssp/SF_ucdatasets/gwas"

# -------------------------------------
# Requirements for the Job (Requirements are explained in further detail in example09.submit_file)
requirements = (HasStornext) && (Machine != "dwalin.eps.surrey.ac.uk") && (Machine != "fili.eps.surrey.ac.uk") && (Machine != "bifur.eps.surrey.ac.uk") && (Machine != "bofur.eps.surrey.ac.uk")  && (Machine != "creative01.eps.surrey.ac.uk") 

#requirements = (HasStornext) && (Machine != "dwalin.eps.surrey.ac.uk") && (Machine != "fili.eps.surrey.ac.uk") && (Machine != "bifur.eps.surrey.ac.uk") && (Machine != "cogvis3.eps.surrey.ac.uk") && (Machine != "bofur.eps.surrey.ac.uk") && (Machine != "creative01.eps.surrey.ac.uk") && (Machine != "oin.eps.surrey.ac.uk") && (Machine != "cvsspgpu01.eps.#surrey.ac.uk") && (Machine != "cvsspgpu02.eps.surrey.ac.uk") && (Machine != "cvsspgpu03.eps.surrey.ac.uk")
#&& (Machine=="sounds01.eps.surrey.ac.uk")

# Resources
request_GPUs     = 1
request_CPUs     = 2
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
                                    for c in cov:
                                        for ua in use_age:
                                            for ug in use_gender:
                                                for l, plr, flr in zip(lr,peak_lr,final_lr):
                                                    experiment_count += 1
                                                    experiment_id = f"{experiment_count}_{b}_{dr}_{e}_{l}_{a}_{s}_{d}_{o}_{w}"
                                                    #experiment_id = f"{experiment_count}_{b}_{dr}_{e}_{l}_{a}_{s}_{d}_{o}_{w}_patience_100_pc_{c}_age_{ua}_gender_{ug}"
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
exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/
genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/col_can_pruned
phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/col_can.xlsx
cov = {c}
use_age = {ua}
use_gender = {ug}
label_col = crc
model_type = snps_only
pca_features_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/col_can_pruned_pca
queue 1
"""

# exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_0.1/chr_wise/
# genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/col_0.1
# phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/col.xlsx
#chr_file = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_col_0.1.gen.gz

# Write the submit content to a file
with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/epic_mlp_model_5D_cov_layer_last_ipca_precompute_col_pruned.submit_file", "w") as f:
    f.write(submit_content)

print(f"Generated {experiment_count} experiments.")


# ##************************************************************* Normalization Techniques *************************************************************************#

# def generate_queue_entry(exp_id, bs, dropout, epochs, lr, peak_lr, final_lr, act, sch, df, opt, wd, hidden_sizes,
#                         cov, use_age, use_gender, norm_pcs, norm_age, norm_gender):
#     """Generate a queue entry for HTCondor submit file"""
#     return f"""
# ID = {exp_id}
# bs = {bs}
# dropout = {dropout}
# epochs = {epochs}
# lr = {lr}
# peak_lr = {peak_lr}
# final_lr = {final_lr}
# act = {act}
# sch = {sch}
# df = {df}
# opt = {opt}
# wd = {wd}
# hidden_sizes = {hidden_sizes}
# exp_dir = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/exp_cov_norm/
# genotype_dir = /vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/col_can
# phenotype_file = /vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/col_can.xlsx
# cov = {cov}
# use_age = {use_age}
# use_gender = {use_gender}
# norm_pcs = {norm_pcs}
# norm_age = {norm_age}
# norm_gender = {norm_gender}
# label_col = col01
# queue 1
# """


# # Define variables
# bs = ["32"]
# dropout = ["0.5"]
# epochs = ["100"]
# lr = ["0.0001"]
# peak_lr = ["1e-3"]
# final_lr =["1e-6"]
# act = ["gelu"]
# sch = ["explr"]
# df =["0.1"]
# opt = ["adamw"]
# wd = ["0.5"]
# hidden_sizes = ["256,64,32"]

# # Feature usage flags
# # cov = ["0","1"]
# # use_age = ["0","1"]
# # use_gender = ["0","1"]

# cov = ["1"]
# use_age = ["1"]
# use_gender = ["1"]

# # Normalization options
# norm_pcs = ["none", "standard", "minmax", "robust"]
# norm_age = ["none", "standard", "minmax", "robust"]
# norm_gender = ["none", "minmax"]


# # Generate the HTCondor submit file content
# submit_content = """
# ##################
# #
# # Example Job for HTCondor
# #
# ####################

# # --------------------------------------
# # Executable and its arguments
# executable    = epic_mlp_model_5D_cov_sep_norm.sh
# arguments = $(ID) $(bs) $(dropout) $(epochs) $(lr) $(peak_lr) $(final_lr) $(act) $(sch) $(df) $(opt) $(wd) $(hidden_sizes) $(exp_dir) $(genotype_dir) $(phenotype_file) $(cov) $(use_age) $(use_gender) $(label_col) $(norm_pcs) $(norm_age) $(norm_gender)

# # -----------------------------------
# # Universe (vanilla, docker)
# universe         = vanilla
 
# # -------------------------------------
# # Event, out, and error logs
# log    = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/exp_cov_norm/logs/c$(Cluster).$(Process).log
# output = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/exp_cov_norm/$(ID)/c$(Cluster).$(Process).out
# error  = /vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_pruned_pca/exp_cov_norm/logs/c$(Cluster).$(Process).error

# should_transfer_files = YES
# environment = "mount=$ENV(PWD),/vol/research/fmodal_mmmed/,/vol/vssp/SF_ucdatasets/gwas"

# # -------------------------------------
# # Requirements for the Job (Requirements are explained in further detail in example09.submit_file)
# requirements = (HasStornext) && (Machine != "dwalin.eps.surrey.ac.uk") && (Machine != "fili.eps.surrey.ac.uk") && (Machine != "bifur.eps.surrey.ac.uk") && (Machine != "bofur.eps.surrey.ac.uk") 

# #requirements = (HasStornext) && (Machine != "dwalin.eps.surrey.ac.uk") && (Machine != "fili.eps.surrey.ac.uk") && (Machine != "bifur.eps.surrey.ac.uk") && (Machine != "cogvis3.#eps.surrey.ac.uk") && (Machine != "bofur.eps.surrey.ac.uk") && (Machine != "creative01.eps.surrey.ac.uk") && (Machine != "oin.eps.surrey.ac.uk") && (Machine != "cvsspgpu01.eps.#surrey.ac.uk") && (Machine != "cvsspgpu02.eps.surrey.ac.uk") && (Machine != "cvsspgpu03.eps.surrey.ac.uk")
# #&& (Machine=="sounds01.eps.surrey.ac.uk")

# # Resources
# request_GPUs     = 1
# request_CPUs     = 2
# request_memory = 20G

# +CanCheckpoint = true
# +GPUMem = 11000
# +JobRunTime = 72

# #-------------------
# """

# # Generate queue statements for each combination of batch_size and lr
# experiment_count = 0
# for b in bs:
#     for dr in dropout:      
#         for e in epochs:
#             for a in act:
#                 for s in sch:
#                     for d in df:
#                         for o in opt:
#                             for w in wd:
#                                 for hs in hidden_sizes:
#                                     for l, plr, flr in zip(lr, peak_lr, final_lr):
#                                         for c in cov:
#                                             for ua in use_age:
#                                                 for ug in use_gender:
#                                                     # If no covariates are used, only one experiment with 'none' normalization
#                                                     if c == "0" and ua == "0" and ug == "0":
#                                                         experiment_count += 1
#                                                         experiment_id = f"{experiment_count}_{b}_{dr}_{e}_{l}_{a}_{s}_{d}_{o}_{w}_cov_{c}_age_{ua}_gender_{ug}_none_none_none"
#                                                         submit_content += generate_queue_entry(
#                                                             experiment_id, b, dr, e, l, plr, flr, a, s, d, o, w, hs,
#                                                             c, ua, ug, "none", "none", "none"
#                                                         )
#                                                         continue

#                                                     # For covariates that are used, try their normalization options
#                                                     curr_norm_pcs = ["none"] if c == "0" else norm_pcs
#                                                     curr_norm_age = ["none"] if ua == "0" else norm_age
#                                                     curr_norm_gender = ["none"] if ug == "0" else norm_gender

#                                                     for npc in curr_norm_pcs:
#                                                         for na in curr_norm_age:
#                                                             for ng in curr_norm_gender:
#                                                                 experiment_count += 1
#                                                                 experiment_id = f"{experiment_count}_{b}_{dr}_{e}_{l}_{a}_{s}_{d}_{o}_{w}_cov_{c}_age_{ua}_gender_{ug}_{npc}_{na}_{ng}"
#                                                                 submit_content += generate_queue_entry(
#                                                                     experiment_id, b, dr, e, l, plr, flr, a, s, d, o, w, hs,
#                                                                     c, ua, ug, npc, na, ng
#                                                                 )

# # Write the submit content to a file
# with open("/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/epic_mlp_model_5D_cov_sep_norm_col.submit_file", "w") as f:
#     f.write(submit_content)

# print(f"Generated {experiment_count} experiments.")

