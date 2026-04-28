import numpy as np
import librosa
import sounddevice as sd
import os
import RPi.GPIO as gpio
import board
import busio
import adafruit_ssd1306
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

gpio.setmode(gpio.BCM)
gpio.setup(17, gpio.IN, pull_up_down=gpio.PUD_UP)
i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)

data = np.load("output/mfcc_dataset.npz", allow_pickle = True)
dataset_mfccs = data["mfccs"]
dataset_files = data["files"]

while True:
    print("Recording...")
    sample = sd.rec(int(2*22050), samplerate=22050, channels=1)
    sd.wait()
    sample = sample.flatten()

    mfcc_new = librosa.feature.mfcc(y=sample, sr=22050, n_mfcc=13)
    mfcc_new_mean = np.mean(mfcc_new, axis=1)

    distances = np.linalg.norm(dataset_mfccs - mfcc_new_mean, axis=1)
    closest_index= np.argmin(distances)
    closest_file = dataset_files[closest_index]
    closest_distance = distances[closest_index]

    print(f"closest match: {closest_file}, distance: {closest_distance:.2f}")

    if closest_distance < 100:
        yay, sr = sf.read("feedback.wav", dtype='float32')
        sd.play(yay, sr)
        sd.wait()
    else:
        print("none")
        oled.fill(255)
        oled.show()
    
    if gpio.input(17)==0:
        oled.fill(0)
        oled.show()
        break
    