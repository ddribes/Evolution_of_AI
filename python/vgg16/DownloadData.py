# Upload kaggle.json
from google.colab import files
uploaded = files.upload()

# Create ~/.kaggle folder
import os
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Move kaggle.json into ~/.kaggle
import shutil
shutil.move("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))

# Set permissions
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Download dataset
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "alessiocorrado99/animals10"
])

# Unzip dataset
import zipfile

with zipfile.ZipFile("animals10.zip", "r") as zip_ref:
    zip_ref.extractall("animals10")