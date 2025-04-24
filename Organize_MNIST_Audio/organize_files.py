import os
import shutil

# Adjust these paths based on where your dataset currently is

#https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist/data
SOURCE_DIR = r"" #ENTER THE DIRECTORY OF YOUR MNIST AUDIO FILES FROM HERE
DEST_TRAIN = "MNIST_Audio_Train"
DEST_TEST = "MNIST_Audio_Test"

# Create output directories if they don't exist
os.makedirs(DEST_TRAIN, exist_ok=True)
os.makedirs(DEST_TEST, exist_ok=True)

# Get all folders
person_folders = os.listdir(SOURCE_DIR)

# Split: first 54 for training, last 6 for testing
train_folders = person_folders[:54]
test_folders = person_folders[54:]

# Copy .wav files from each folder to the respective destination
def copy_files(folders, dest_root):
    for folder in folders:
        src_folder = os.path.join(SOURCE_DIR, folder)
        for file in os.listdir(src_folder):
            if file.endswith(".wav"):
                src_path = os.path.join(src_folder, file)
                dest_path = os.path.join(dest_root, file)
                shutil.copy(src_path, dest_path)

print("Copying training data...")
copy_files(train_folders, DEST_TRAIN)
print("Copying testing data...")
copy_files(test_folders, DEST_TEST)

print("Done.")