import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os

from PIL import Image
import matplotlib.pyplot as plt

def scale_minmax(X, min=0.0, max=1.0):
	X_std = (X - X.min()) / (X.max() - X.min())
	X_scaled = X_std * (max - min) + min
	return X_scaled

os.makedirs("birds", exist_ok=True)

dataset = pd.read_csv("dataset/bird_songs_metadata.csv")[["id", "name", "filename"]]


for bird in set(dataset["name"]):
	os.makedirs(f"birds/{bird.replace(' ', '_')}", exist_ok=True)

for i, row in dataset.iterrows():
	# Open wav file
	sample_rate, samples = wavfile.read(f'dataset/wavfiles/{row["filename"]}')
	frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

	spectrogram = np.log(spectrogram + 1e-9) # Add to avoid divide by 0

	spectrogram = 255 - scale_minmax(spectrogram, 0, 255).astype(np.uint8)

	# print(spectrogram)

	spec_im = Image.fromarray(spectrogram)
	spec_im.save(f"birds/{row['name'].replace(' ', '_')}/{row['filename'].replace('wav', 'png')}")