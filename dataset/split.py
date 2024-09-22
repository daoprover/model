import os
import random
import shutil

# Paths for your data and folders
source_folder = '/home/sempai/Desktop/Projects/validation-model/assets/graphs'
train_folder = '/home/sempai/Desktop/Projects/validation-model/assets/train'
valid_folder = '/home/sempai/Desktop/Projects/validation-model/assets/valid'
test_folder = '/home/sempai/Desktop/Projects/validation-model/assets/test'

# Create directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Load all files from the source folder
all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Shuffle the files
random.shuffle(all_files)

# Define split ratios
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# Split the dataset
train_split = int(len(all_files) * train_ratio)
valid_split = train_split + int(len(all_files) * valid_ratio)

train_files = all_files[:train_split]
valid_files = all_files[train_split:valid_split]
test_files = all_files[valid_split:]

# Move files to train folder
for file_name in train_files:
    shutil.move(os.path.join(source_folder, file_name), os.path.join(train_folder, file_name))

# Move files to validation folder
for file_name in valid_files:
    shutil.move(os.path.join(source_folder, file_name), os.path.join(valid_folder, file_name))

# Move files to test folder
for file_name in test_files:
    shutil.move(os.path.join(source_folder, file_name), os.path.join(test_folder, file_name))

print(f"Files moved to: \nTrain: {len(train_files)} \nValid: {len(valid_files)} \nTest: {len(test_files)}")
