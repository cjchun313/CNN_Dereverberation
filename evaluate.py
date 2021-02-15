import numpy as np
import librosa
from pesq import pesq

if __name__ == '__main__':
    ref, sr = librosa.load('../db/sample/audio.wav', sr=16000, mono=True)
    ref = 0.2 * ref / np.max(np.abs(ref))

    deg, sr = librosa.load('../db/sample/audio_rir.wav', sr=16000, mono=True)
    deg = 0.2 * deg / np.max(np.abs(deg))

    print(pesq(sr, ref, deg, 'wb'))
    print(pesq(sr, ref, deg, 'nb'))