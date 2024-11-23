import os
import tarfile
import urllib.request
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_voc2012():
    # URLs for the VOC dataset
    VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    
    # Create directories
    root_dir = "datasets"
    voc_dir = os.path.join(root_dir, "VOCdevkit")
    os.makedirs(root_dir, exist_ok=True)

    # Download VOC2012
    filename = os.path.join(root_dir, "VOCtrainval_11-May-2012.tar")
    if not os.path.exists(filename):
        print("Downloading VOC2012 dataset...")
        with DownloadProgressBar(unit='B', unit_scale=True,
                               miniters=1, desc="VOC2012") as t:
            urllib.request.urlretrieve(VOC_URL, filename=filename,
                                     reporthook=t.update_to)
    
    # Extract VOC2012
    if not os.path.exists(voc_dir):
        print("Extracting VOC2012...")
        with tarfile.open(filename) as tar:
            tar.extractall(path=root_dir)
        print("Extraction completed!")

    # Remove the tar file to save space (optional)
    os.remove(filename)
    
    print("\nDataset structure:")
    print(f"Dataset is located at: {os.path.abspath(voc_dir)}")
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(os.path.join(voc_dir, "VOC2012")):
        level = root.replace(voc_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        if level < 2:  # Limit depth of files shown
            for f in files[:3]:  # Show only first 3 files
                print(f"{subindent}{f}")
            if len(files) > 3:
                print(f"{subindent}...")

if __name__ == "__main__":
    download_voc2012()
