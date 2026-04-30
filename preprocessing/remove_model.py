import os

def remove_model_file(folder):
    file_to_remove = "trained_model.pth"
    for root, dirs, files in os.walk(folder):
        if file_to_remove in files:
            os.remove(os.path.join(root, file_to_remove))
            print(f"Removed {file_to_remove} from {root}")

def main():
    folder_path = "/vol/research/fmodal_mmmed/Codes/GenNet_MLP/results_updated_noval_opt"
    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return
    
    remove_model_file(folder_path)

if __name__ == "__main__":
    main()
