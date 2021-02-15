import os
import librosa
import numpy as np

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

def read_audio(filepath, sr=22050):
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    audio = 0.2 * audio / np.max(np.abs(audio))

    return np.array(audio, dtype=np.float32)