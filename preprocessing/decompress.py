import shutil
import os
import gzip

def decompress_gz_files(folder_path):
    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return
    files = os.listdir(folder_path)
    print(f"List of Files is: \n {files}")
    for file in files:
        file_path = os.path.join(folder_path, file)
        if file.endswith(".gz"):
            output_file_path = file_path[:-3]
            with open(output_file_path, 'wb') as f_out, gzip.open(file_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed: {output_file_path}")
if __name__ == "__main__":
    folder_path = "/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen"
    decompress_gz_files(folder_path)

# def decompress_gz_file(gz_file_path):
#     if not os.path.exists(gz_file_path):
#         print("File path does not exist.")
#         return
#     if not gz_file_path.endswith(".gz"):
#         print("Not a .gz file.")
#         return
#     output_file_path = gz_file_path[:-3]
#     with open(output_file_path, 'wb') as f_out, gzip.open(gz_file_path, 'rb') as f_in:
#         shutil.copyfileobj(f_in, f_out)
#     print(f"Decompressed: {output_file_path}")
# if __name__ == "__main__":
#     gz_file_path = "/vol/research/ucdatasets/gwas/chr15_3d.gen.gz"
#     decompress_gz_file(gz_file_path)

