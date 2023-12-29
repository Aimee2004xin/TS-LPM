
from scipy import io
import soundfile
import torch
import numpy as np
import scipy.signal as signal
from fairseq.models.wav2vec import Wav2VecModel
import fairseq
import os

def extract_wav2vec(wavfile, savefile):
    '''
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    wavs, fs = soundfile.read(wavfile)
    
    if fs != sample_rate:
        num = int((wavs.shape[0]) / fs * sample_rate)
        wavs = signal.resample(wavs, num)

    if wavs.ndim > 1:
        wavs = np.mean(wavs, axis=1)

    wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)    # (B, S)
    
    z = wav2vec.feature_extractor(wavs)
    # z = wav2vec.vector_quantizer(z)['x']    # vq-wav2vec
    feature_wav = wav2vec.feature_aggregator(z)
    feature_wav = feature_wav.transpose(1,2).squeeze().detach().numpy()   # (t, 512)
    dict = {'wav': feature_wav}
    io.savemat(savefile, dict)
    
    # print(savefile, feature_wav.shape)


def extract_spec_batch(input_dir, output_dir, fun):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = os.listdir(input_dir)
    # Filter out files that already have a processed counterpart
    files_to_process = [file for file in all_files if not os.path.exists(os.path.join(output_dir, file.split('.')[0] + '.mat')) and file.endswith('.wav')]

    total = len(files_to_process)
    print(f'Total files to process: {total}')

    for i, file in enumerate(files_to_process):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file.split('.')[0] + '.mat')

        try:
            fun(input_file, output_file)
            print('\r%s/%s' % (i, total), end=' ')
        except:
            print('error: ', input_file)



if __name__ == '__main__':
    '''
    Pre-trained wav2vec model is available at https://github.com/pytorch/fairseq/blob/main/examples/wav2vec.
    Download model and save at model_path.
    '''
    model_path = '/home/xrl/speech/wav2vec_large.pt'
    # wav2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    # wav2vec = wav2vec[0]
    # wav2vec.eval()

    sample_rate = 16000    # input should be resampled to 16kHz!
    from fairseq.checkpoint_utils import load_model_ensemble_and_task
    models, cfg, task = load_model_ensemble_and_task([model_path])
    wav2vec = models[0]  # 取出单个模型（如果是集成模型）
    wav2vec.eval()
    #### use extract_wav2vec
    # wavfile = xxx
    # savefile = xxx
    # extract_wav2vec(wavfile, savefile)
    input_dir = '/home/xrl/speech/tess/wav' 
    output_dir = '/home/xrl/speech/tess/wav_wav2vec_mat'
    extract_spec_batch(input_dir, output_dir, extract_wav2vec)
    
    
