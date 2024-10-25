import os
from sklearn.model_selection import KFold
from glob import glob
import shutil

# Set up directories for cross-validation splits
dataset_path = '../data'
output = '../cross_splitted'
subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

# Parameters
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Process each subfolder
for subfolder in subfolders:
    file_paths = glob(os.path.join(subfolder, '*'))
    for fold, (train_idx, test_idx) in enumerate(kf.split(file_paths)):
        train_paths = [file_paths[i] for i in train_idx]
        test_paths = [file_paths[i] for i in test_idx]
        
        # Create directories for this fold
        fold_train_dir = os.path.join(output, f'fold_{fold+1}', 'train', os.path.basename(subfolder))
        fold_test_dir = os.path.join(output, f'fold_{fold+1}', 'test', os.path.basename(subfolder))
        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_test_dir, exist_ok=True)

        # Copy files to respective fold directories
        for train_file in train_paths:
            shutil.copy(train_file, fold_train_dir)
        for test_file in test_paths:
            shutil.copy(test_file, fold_test_dir)
