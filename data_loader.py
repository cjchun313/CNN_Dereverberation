import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import glob
import numpy as np
import librosa

from audio_processing import dynamic_range_compression
from image_method import ImageMethod

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


class WavDatasetForDereverb(Dataset):
    def __init__(self, mode='train', ='waveglow', sr=22050, wav_sec=11.88, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
        self.mode = mode
        self.sr = sr
        self.sample_len = int(self.sr * wav_sec)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

        #ImageMethod(mic=mic, n=n, r=r, rm=rm, src=src, fs=22050)
        # for train
        self.rir0 = ImageMethod(mic=[4.0, 1.0, 1.5], n=12, r=0.70, rm=[8.0, 6.0, 3.5], src=[4.0, 2.5, 1.5])
        self.rir1 = ImageMethod(mic=[3.5, 2.0, 1.4], n=15, r=0.80, rm=[7.0, 5.0, 3.3], src=[3.8, 3.5, 1.5])
        # for dev
        self.rir2 = ImageMethod(mic=[6.0, 1.5, 1.6], n=13, r=0.85, rm=[12.0, 6.0, 3.8], src=[6.2, 3.1, 1.5])
        # for test
        self.rir3 = ImageMethod(mic=[5.0, 2.5, 1.7], n=15, r=0.72, rm=[10.0, 8.0, 4.2], src=[5.1, 4.2, 1.6])

        filepath = '../db/LibriSpeech/'
        if self.mode == 'train':
            filepath += 'train-clean-100/'
        elif self.mode == 'val':
            filepath += 'dev-clean/'
        elif self.mode == 'test':
            filepath += 'test-clean/'

        if type == 'waveglow':
            self.fmin = 0.0
            self.fmax = 8000.0
        else:
            self.fmin = 0.0
            self.fmax = None

        self.wavfiles = sorted(glob.glob(filepath + '*/*/*.flac'))
        self.data_len = len(self.wavfiles)
        print('data len:', self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        wavfile = self.wavfiles[idx]
        audio = self.read_audio(wavfile)
        if self.mode == 'train':
            if np.random.random() < 0.5:
                audio_rir = self.rir0.conv(audio)
            else:
                audio_rir = self.rir1.conv(audio)
        elif self.mode == 'val':
            audio_rir = self.rir2.conv(audio)
        elif self.mode == 'test':
            audio_rir = self.rir3.conv(audio)
        #print(audio.shape, audio_rir.shape)

        mel = librosa.feature.melspectrogram(y=audio,
                                             sr=self.sr,
                                             n_fft=self.n_fft,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length,
                                             n_mels=self.n_mels,
                                             fmin=self.fmin,
                                             fmax=self.fmax
                                             ).reshape(1, self.n_mels, -1)
        mel = torch.from_numpy(mel).to(device=DEVICE, dtype=torch.float32)
        mel = dynamic_range_compression(mel)

        mel_rir = librosa.feature.melspectrogram(y=audio_rir,
                                             sr=self.sr,
                                             n_fft=self.n_fft,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length,
                                             n_mels=self.n_mels,
                                             fmin=self.fmin,
                                             fmax=self.fmax
                                             ).reshape(1, self.n_mels, -1)
        mel_rir = torch.from_numpy(mel_rir).to(device=DEVICE, dtype=torch.float32)
        mel_rir = dynamic_range_compression(mel_rir)

        return mel_rir, mel

    def read_audio(self, filepath):
        audio, _ = librosa.load(filepath, sr=self.sr, mono=True)
        audio = 0.2 * audio / np.max(np.abs(audio))

        audio_len = len(audio)
        if audio_len != self.sample_len:
            if audio_len > self.sample_len:
                audio = audio[:self.sample_len]
            else:
                zero_sample = np.zeros([self.sample_len - audio_len])
                audio = np.concatenate((audio, zero_sample), axis=-1)

        return audio





if __name__ == "__main__":
    val_dataset = WavDatasetForDereverb(mode='val')
    print(val_dataset)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    cnt = 0
    for batch_idx, samples in enumerate(val_loader):
        data, target  = samples
        print(data.shape, target.shape)

        cnt += 1
        if cnt == 4:
            break