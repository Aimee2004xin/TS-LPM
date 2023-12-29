import os
import librosa
import numpy as np
from scipy import io

class MFCCExtractor(object):
    def __init__(self, sr=16000, n_mfcc=13, n_fft=400, hop_length=160, wav_length=104640):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.wav_length = wav_length

    def read_audio(self, path):
        wav, sr = librosa.load(path, sr=self.sr)

        # Data Augmentation
        wav = self.add_noise(wav)
        wav = self.time_stretch(wav)

        # Make the audio file of equal length by padding zeros
        if len(wav) < self.wav_length:
            wav = np.pad(wav, (0, self.wav_length - len(wav)))

        return wav[:self.wav_length]

    def add_noise(self, wav, noise_level=0.005):
        return wav + noise_level * np.random.randn(len(wav))

    def time_stretch(self, wav, rate=0.8):
        return librosa.effects.time_stretch(wav, rate)

    def get_feats(self, path):
        x = self.read_audio(path)
        mfccs = librosa.feature.mfcc(x, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        return mfccs.T

def extract_mfcc(model: MFCCExtractor, wavfile, savefile):
    fea = model.get_feats(wavfile)
    dict = {'mfcc': fea}
    io.savemat(savefile, dict)

def handle_dataset(model: MFCCExtractor):
    wavroot = '/home/xrl/speech/tess/wav'
    matroot = '/home/xrl/speech/tess/wav_wav2vec_mat'
    save_dir = '/home/xrl/speech/tess/enmfcc'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wav_files = [f for f in os.listdir(wavroot) if f.endswith('.wav')]
    mat_files = [f.replace('.wav', '.mat') for f in wav_files]
    
    # 计算已存在的.mat文件数量
    existing_mat_files = os.listdir(save_dir)
    # 计算需要处理的文件数量
    files_to_process = [f for f in mat_files if f not in existing_mat_files]

    print(f'We need to process {len(files_to_process)} files.')

    for i, mat_file in enumerate(files_to_process):
        filename = mat_file.split('.')[0]
        wavfile = os.path.join(wavroot, filename + '.wav')
        savefile = os.path.join(save_dir, mat_file)
        if not os.path.exists(savefile):
            extract_mfcc(model, wavfile, savefile)
            print(f'\r{i+1}/{len(files_to_process)}, {savefile}', end=' ')


if __name__ == '__main__':
    extractor = MFCCExtractor()
    # datasets = ['iemocap', 'cremad', 'savee', 'meld']
    # for dataset_name in datasets:
    handle_dataset(extractor)
