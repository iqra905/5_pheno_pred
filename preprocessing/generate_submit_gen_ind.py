# Generate the HTCondor submit file content
submit_content = """
# --------------------------------------
# Executable and its arguments
executable    = create_ind_files.sh
#arguments = $(start_idx) $(sample_idx)
arguments = $(file_name) 


# -----------------------------------
# Universe (vanilla, docker)
universe         = vanilla
 
# -------------------------------------
# Event, out, and error logs
log    = logs/c$(Cluster).$(Process).log
output = logs/c$(Cluster).$(Process).out
error  = logs/c$(Cluster).$(Process).error

should_transfer_files = YES
environment = "mount=$ENV(PWD),/vol/research/fmodal_mmmed/"

# -------------------------------------
# Requirements for the Job (Requirements are explained in further detail in example09.submit_file)
requirements = (HasStornext) 
#&& (Machine=="sounds01.eps.surrey.ac.uk")

# Resources
request_GPUs     = 1
request_CPUs     = 2
request_memory = 15G

+CanCheckpoint = true
+GPUMem = 17000
+JobRunTime = 72

#-------------------
"""

num_samples = 1000
for i in range(num_samples):
    sample_idx = i + 1
    file_name = "sample_"+ str(sample_idx) +".gen.gz"
    submit_content += f"file_name = {file_name} \n queue 1 \n\n"

with open("rm_dup1.submit_file", "w") as f:
    f.write(submit_content)

# num_samples = 37663
# for i in range(num_samples):
#     start_idx = 5 + i * 3
#     sample_idx = i + 1
#     submit_content += f"start_idx = {start_idx} \n sample_idx = {sample_idx} \n queue 1 \n\n"

# with open("create_ind_files.submit_file", "w") as f:
#     f.write(submit_content)



