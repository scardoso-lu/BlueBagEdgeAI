import os
import shutil
import kagglehub

os.makedirs("dataset", exist_ok=True)
# Download latest version
path = kagglehub.dataset_download("viswaprakash1990/garbage-detection")

print("Path to dataset files:", path)
shutil.move(path, "dataset")
