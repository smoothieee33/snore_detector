import librosa
import numpy as np

audio_path = "snore_library/fixed/onee.wav"

y, sr = librosa.load(audio_path, sr=16000)

print("Sample rate:", sr)
print("Audio length (samples)", len(y))

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print("MFCC shape:", mfcc.shape)
