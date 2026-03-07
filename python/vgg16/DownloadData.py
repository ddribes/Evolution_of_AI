import os
import shutil
import subprocess
import zipfile
from pathlib import Path

# Install kaggle if not installed
subprocess.run(["pip", "install", "kaggle"], check=True)

# Set permissions (important for Kaggle API)
os.chmod("kaggle.json", 0o600)

# Download dataset
subprocess.run(
    ["kaggle", "datasets", "download", "-d", "alessiocorrado99/animals10"],
    cwd=".",
    check=True
)

# Unzip dataset
zip_path = "animals10.zip"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(".")

print("Dataset ready in:", ".")
