import numpy as np
from scipy import sparse
from scipy.signal import fftconvolve

from utils import read_audio
#from soundfile import write
from scipy.io.wavfile import write

class ImageMethod:
    def __init__(self, mic, n, r, rm, src, c=343, fs=22050):
        self.c = c
        self.fs = fs
        self.mic = mic
        self.n = n
        self.r = r
        self.rm = rm
        self.src = src

        self.gen_rir()

    def set_mic(self, mic):
        self.mic = mic
        self.gen_rir()

    def set_src(self, src):
        self.src = src
        self.gen_rir()

    def gen_rir(self):
        nn = np.linspace(-self.n, self.n, 2 * self.n + 1)
        rms = nn + 0.5 - 0.5 * np.power(-1.0, nn)
        srcs = np.power(-1.0, nn)

        xi = srcs * self.src[0] + rms * self.rm[0] - self.mic[0]
        yj = srcs * self.src[1] + rms * self.rm[1] - self.mic[1]
        zk = srcs * self.src[2] + rms * self.rm[2] - self.mic[2]

        i, j, k = np.meshgrid(xi, yj, zk)
        d = np.sqrt(np.square(i) + np.square(j) + np.square(k))
        t = np.round(self.fs * d / self.c) + 1

        e, f, g = np.meshgrid(nn, nn, nn)
        cc = np.power(self.r, np.abs(e) + np.abs(f) + np.abs(g))
        ee = np.divide(cc, d)

        t = t.reshape([-1])
        ee = ee.reshape([-1])
        one = np.ones(np.size(t, 0)).reshape([-1])

        self.length = np.max(t).astype(int) + 1

        hh = sparse.coo_matrix((ee, (one, t)), shape=(self.length, self.length)).toarray()
        self.h = np.sum(hh, axis=0).reshape([-1, 1])
        self.h = self.h / np.max(np.abs(self.h))
        self.h = self.h.reshape(-1,)

        idx = np.argmax(self.h)
        self.h = self.h[idx-1:]
        self.length = self.length - idx

    def get_rir(self):
        if self.h is None:
            self.gen_rir()

        return self.h, self.length

    def conv(self, x):
        length = x.shape[0]
        y = fftconvolve(self.h, x, mode='full')
        y = y[:length]

        return np.array(y, dtype=np.float32)


if __name__ == "__main__":
    mic = [3.0, 2.5, 1.7]
    n = 15
    r = 0.85
    rm = [10, 8, 4.2]
    src = [6.0, 3.0, 1.8]

    audio = read_audio('./audio.wav')

    image_method = ImageMethod(mic=mic, n=n, r=r, rm=rm, src=src, fs=22050)
    h, len = image_method.get_rir()

    y = image_method.conv(audio)
    print(len, y.shape, y.dtype)

    rate = 22050
    write('audio_rir.wav', rate, 0.4 * y)

    print('done!')

