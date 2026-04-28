import os
import numpy as np
import librosa

WAV_FOLDER = "snore_library/fixed"
OUTPUT = "output/mfcc_dataset.npz"

mfcc_list = []
filenames = []

for filename in sorted(os.listdir(WAV_FOLDER)):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(WAV_FOLDER, filename)
        print(f"Processing {filepath}...")
        
        y, sr = librosa.load(filepath, sr = 22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        mfcc_mean = np.mean(mfcc, axis=1)
        
        mfcc_list.append(mfcc_mean)
        filenames.append(filename)
        
mfcc_dataset = np.stack(mfcc_list)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
np.savez(OUTPUT, mfccs=mfcc_dataset, files=filenames)

print(f"MFCC dataset saved to {OUTPUT}")
print("Shape:", mfcc_dataset.shape)

