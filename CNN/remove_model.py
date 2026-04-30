import os
import argparse

def remove_model_file(folder):
    """Remove all checkpoint files matching the pattern checkpoint_epoch_*.pt"""
    if not os.path.exists(folder):
        print("Folder path does not exist.")
        return
    
    files_removed = 0
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            #Check if file starts with "checkpoint_epoch_" and ends with ".pt"
            if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
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

            if file.startswith("model") and file.endswith(".pt"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file} from {root}")
                    files_removed += 1
                except OSError as e:
                    print(f"Error removing {file} from {root}: {e}")
            
            if file.startswith("trained") and file.endswith(".p"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed {file} from {root}")
                    files_removed += 1
                except OSError as e:
                    print(f"Error removing {file} from {root}: {e}")
    
    print(f"Total files removed: {files_removed}")

def main():
    parser = argparse.ArgumentParser(description="Remove model epoch checkpoints.") 
    parser.add_argument('-folder_path', type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results', help="Path to the directory containing experiment folders")

    args = parser.parse_args()    
    
    remove_model_file(args.folder_path)

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
