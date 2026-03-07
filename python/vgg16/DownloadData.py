import os
import shutil
import subprocess
import zipfile
from pathlib import Path

# Install kaggle if not installed
subprocess.run(["pip", "install", "kaggle"], check=True)

# Paths
home = Path.home()
kaggle_dir = home / ".kaggle"
project_dir = home / "Documents" / "EvolutionAI" / "data"

# Create folders
kaggle_dir.mkdir(parents=True, exist_ok=True)
project_dir.mkdir(parents=True, exist_ok=True)

# Move kaggle.json to ~/.kaggle
shutil.move("kaggle.json", kaggle_dir / "kaggle.json")

# Set permissions (important for Kaggle API)
os.chmod(kaggle_dir / "kaggle.json", 0o600)

# Download dataset
subprocess.run(
    ["kaggle", "datasets", "download", "-d", "alessiocorrado99/animals10"],
    cwd=project_dir,
    check=True
)

# Unzip dataset
zip_path = project_dir / "animals10.zip"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(project_dir)

print("Dataset ready in:", project_dir)
