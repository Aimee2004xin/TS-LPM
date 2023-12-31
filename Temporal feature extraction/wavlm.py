import torch
import soundfile as sf
import scipy.signal as signal
from scipy import io
import numpy as np
from WavLM import WavLM, WavLMConfig
import os
class WavLMFeatureExtractor:
    def __init__(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.cfg = cfg

    def read_audio(self, path):
        wav, sr = sf.read(path)
        if sr != 16000: 
            wav = signal.resample(wav, int(wav.shape[0] / sr * 16000))
        if wav.ndim == 2:
            wav = wav.mean(-1)
        return wav

    def extract_features(self, wavfile):
        wav = self.read_audio(wavfile)
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        if self.cfg.normalize:
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        with torch.no_grad():
            rep = self.model.extract_features(wav)[0]
        return rep

def handle_dataset(extractor, dataset_name):
    wavroot = f'/home/xrl/speech/{dataset_name}/wav'
    save_root = f'/home/xrl/speech/{dataset_name}/wavlm'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    wavs = os.listdir(wavroot)
    print(f'We have {len(wavs)} samples in total.')
    for i, wav in enumerate(wavs):
        wavfile = os.path.join(wavroot, wav)
        savefile = os.path.join(save_root, wav.replace('.wav', '.mat'))
        if not os.path.exists(savefile):
            features = extractor.extract_features(wavfile).cpu().numpy()
            io.savemat(savefile, {'features': features})
            print(f'\r {i+1}/{len(wavs)}, {savefile}', end=' ')

if __name__ == '__main__':
    ckpt_path = "/home/xrl/speech/WavLM-Large.pt"
    extractor = WavLMFeatureExtractor(ckpt_path)
    handle_dataset(extractor, 'savee')
