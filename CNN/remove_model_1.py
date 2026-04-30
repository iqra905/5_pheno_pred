
import os

def remove_model_file(folder):
    """Remove all checkpoint files matching the pattern checkpoint_epoch_*.pt"""
    files_removed = 0
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Check if file starts with "checkpoint_epoch_" and ends with ".pt"
            if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
            #if file.startswith("best_model") and file.endswith(".pt"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file} from {root}")
                    files_removed += 1
                except OSError as e:
                    print(f"Error removing {file} from {root}: {e}")
            
            if file.startswith("best_model") and file.endswith(".pt"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file} from {root}")
                    files_removed += 1
                except OSError as e:
                    print(f"Error removing {file} from {root}: {e}")
    
    print(f"Total files removed: {files_removed}")

def main():
    folder_path = "/mnt/fast/nobackup/scratch4weeks/if00208/disease_wise_multiscale/parallel/no_pool"
    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return
    
    remove_model_file(folder_path)

if __name__ == "__main__":
    main()

# import os

# def remove_model_file(folder):
#     file_to_remove = "trained_genotype_model.pth"
#     for root, dirs, files in os.walk(folder):
#         if file_to_remove in files:
#             os.remove(os.path.join(root, file_to_remove))
#             print(f"Removed {file_to_remove} from {root}")

# def main():
#     folder_path = "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/02"
#     if not os.path.exists(folder_path):
#         print("Folder path does not exist.")
#         return
    
#     remove_model_file(folder_path)

# if __name__ == "__main__":
#     main()
