
import fairseq
import soundfile as sf
import scipy.signal as signal
from scipy import io
import torch
import torch.nn.functional as F
import os
import numpy as np
from fairseq.models.wav2vec import Wav2VecModel

def get_receptive_field(k: list, s: list):
    k.reverse()
    s.reverse()

    output_1 = 1
    output_2 = 2
    for _k, _s in zip(k, s):
        recept_1 = (output_1 - 1) * _s + _k
        output_1 = recept_1
        recept_2 = (output_2 - 1) * _s + _k
        output_2 = recept_2

    print('After the convolutional waveform encoder in HuBERT, the feature')
    print('receptive field is:', recept_1, 'points (/sr -> second)')
    print('hop is:', recept_2 - recept_1, 'points (/sr -> second)')
class Hubert(object):
    def __init__(self, ckpt_path, max_chunk=1600000, wav_length=104640):


        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval()

        self.task = task
        self.max_chunk = max_chunk
        self.wav_length = wav_length  # (326 + 1) * 0.02 * 16000 = 104640

    def read_audio(self, path):
        wav, sr = sf.read(path)
        
        if sr != self.task.cfg.sample_rate:
            num = int((wav.shape[0]) / sr * self.task.cfg.sample_rate)
            wav = signal.resample(wav, num)
            # print(f'Resample {sr} to {self.task.cfg.sample_rate}')
        
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim

        return wav
    
    def read_audio_batch(self, path_batch):
        x = []
        for path in path_batch:
            _x = self.read_audio(path)
            _x = np.pad(_x, (0, self.wav_length - _x.shape[0]), constant_values=(0, 0)) if _x.shape[0] < self.wav_length else _x[:self.wav_length]
            x.append(_x)

        x = np.stack(x, axis=0)
        return x

    def get_feats(self, path, layer):
        '''Layer index starts from 1. (e.g. 1-24)
        '''
        if isinstance(path, str):
            B = 1
            x = self.read_audio(path)
        else:
            B = len(path)
            x = self.read_audio_batch(path)

        x = torch.from_numpy(x).float()
        if self.task.cfg.normalize:
            x = F.layer_norm(x, x.shape)
        x = x.view(B, -1)

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start: start + self.max_chunk]
            feat_chunk, _ = self.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=layer,
            )
            feat.append(feat_chunk)
        return torch.cat(feat, 1)

def extract_hubert(model: Hubert, layer, wavfile, savefile):
    with torch.no_grad():
        fea = model.get_feats(wavfile, layer=layer).squeeze(0)

    fea = fea.cpu().detach().numpy()   # (t, 768)  / (t, 1024)
    dict = {'hubert': fea}
    io.savemat(savefile, dict)
    
    # print(savefile, '->', fea.shape)

def handle_cremad(model: Hubert):
    matroot = '/home/xrl/speech/cremad/wav_wav2vec_mat' 
    save_L12 = '/home/xrl/speech/cremad/hubert_large_L12_mat' 

    matroot_s = matroot
    save_L12_s = save_L12

    if not os.path.exists(save_L12_s):
        os.makedirs(save_L12_s)

    mats = os.listdir(matroot_s)
    print(f'We have {len(mats)} samples in total.')
    for i, mat in enumerate(mats):
        mat_name = mat.split('.')[0]
        wavfile = '/home/xrl/speech/cremad/wav/%s.wav' % (mat_name)
        savefile_L12 = os.path.join(save_L12_s, mat)
        if os.path.exists(savefile_L12):
            continue
        extract_hubert(model, 12, wavfile, savefile_L12)
        print('\r %s/%s, %s' % (i, len(mats), savefile_L12), end=' ')
def handle_heartbeat(model: Hubert):
    matroot = '/home/xrl/speech/heartbeat/wav_wav2vec_mat' 
    save_L12 = '/home/xrl/speech/heartbeat/hubert_large_L12_mat' 

    matroot_s = matroot
    save_L12_s = save_L12

    if not os.path.exists(save_L12_s):
        os.makedirs(save_L12_s)

    mats = os.listdir(matroot_s)
    print(f'We have {len(mats)} samples in total.')
    for i, mat in enumerate(mats):
        mat_name = mat.split('.')[0]
        wavfile = '/home/xrl/speech/heartbeat/wav/%s.wav' % (mat_name)
        savefile_L12 = os.path.join(save_L12_s, mat)
        if os.path.exists(savefile_L12):
            continue
        extract_hubert(model, 12, wavfile, savefile_L12)
        print('\r %s/%s, %s' % (i, len(mats), savefile_L12), end=' ')
        
def handle_savee(model: Hubert):
    matroot = '/home/xrl/speech/savee/wav_wav2vec_mat'
    save_L12 = '/home/xrl/speech/savee/hubert_large_L12_mat'

    matroot_s = matroot
    save_L12_s = save_L12

    if not os.path.exists(save_L12_s):
        os.makedirs(save_L12_s)

    mats = os.listdir(matroot_s)
    print(f'We have {len(mats)} samples in total.')
    for i, mat in enumerate(mats):
        mat_name = mat.split('.')[0]
        wavfile = '/home/xrl/speech/savee/wav/%s.wav' % (mat_name)
        savefile_L12 = os.path.join(save_L12_s, mat)
        if os.path.exists(savefile_L12):
            continue
        extract_hubert(model, 12, wavfile, savefile_L12)
        print('\r %s/%s, %s' % (i, len(mats), savefile_L12), end=' ')

def handle_tess(model: Hubert):
    matroot = '/home/xrl/speech/tess/wav'
    save_L12 = '/home/xrl/speech/tess/hubert_large_L12_mat'

    if not os.path.exists(save_L12):
        os.makedirs(save_L12)

    # 获取所有的 wav 文件
    wav_files = [f for f in os.listdir(matroot) if f.endswith('.wav')]
    # 获取所有已经处理过的 mat 文件
    processed_files = os.listdir(save_L12)

    # 过滤出那些还没有处理过的 wav 文件
    files_to_process = [f for f in wav_files if f.replace('.wav', '.mat') not in processed_files]

    print(f'We have {len(files_to_process)} samples to process.')

    for i, wavfile in enumerate(files_to_process):
        mat_name = wavfile.split('.')[0]
        savefile_L12 = os.path.join(save_L12, mat_name + '.mat')
        
        try:
            extract_hubert(model, 12, os.path.join(matroot, wavfile), savefile_L12)
        except Exception as error:
            print('error: ', error, '  ', wavfile)
        
        print(f'\r {i+1}/{len(files_to_process)}, {savefile_L12}', end=' ')

        
def handle_emovo(model: Hubert):
    matroot = '/home/xrl/speech/emovo/wav_wav2vec_mat' 
    save_L12 = '/home/xrl/speech/emovo/hubert_large_L12_mat' 

    matroot_s = matroot
    save_L12_s = save_L12

    if not os.path.exists(save_L12_s):
        os.makedirs(save_L12_s)

    mats = os.listdir(matroot_s)
    print(f'We have {len(mats)} samples in total.')
    for i, mat in enumerate(mats):
        mat_name = mat.split('.')[0]
        wavfile = '/home/xrl/speech/emovo/wav/%s.wav' % (mat_name)
        savefile_L12 = os.path.join(save_L12_s, mat)
        if os.path.exists(savefile_L12):
            continue
        extract_hubert(model, 12, wavfile, savefile_L12)
        print('\r %s/%s, %s' % (i, len(mats), savefile_L12), end=' ')  
        
def handle_iemocap(model: Hubert):
    matroot = '/home/xrl/speech/iemocap/wav_wav2vec_mat'
    save_L12 = '/home/xrl/speech/iemocap/hubert_large_L12_mat'

    matroot_s = matroot
    save_L12_s = save_L12

    if not os.path.exists(save_L12_s):
        os.makedirs(save_L12_s)

    mats = os.listdir(matroot_s)
    print(f'We have {len(mats)} samples in total.')
    for i, mat in enumerate(mats):
        mat_name = mat.split('.')[0]
        wavfile = '/home/xrl/speech/iemocap/wav/%s.wav' % (mat_name)
        savefile_L12 = os.path.join(save_L12_s, mat)
        if os.path.exists(savefile_L12):
            continue
        try:
            extract_hubert(model, 12, wavfile, savefile_L12)
        except Exception as error:
            print('error: ', error, '  ', wavfile)
        print('\r %s/%s, %s' % (i, len(mats), savefile_L12), end=' ')

if __name__ == '__main__':
    # mode = 'train'

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    get_receptive_field(k=[10, 3, 3, 3, 3, 2, 2], s=[5, 2, 2, 2, 2, 2, 2])
    
    ckpt_path = ".../hubert_base_ls960.pt"
    # hubert_large_ll60k, hubert_base_ls960 
    model = Hubert(ckpt_path)
    # for mode in ['train', 'test']:
    #handle_heartbeat(model)
    # handle_cremad(model)
    # handle_savee(model)
    handle_tess(model)
    
