import soundfile as sf
import torch
import numpy as np
import os
import scipy.signal as signal
from scipy import io as sio  # Import scipy.io for savemat
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.wav2vec import Wav2Vec2Model

from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.wav2vec import Wav2Vec2Model

from fairseq.models.wav2vec import Wav2Vec2Model
from transformers import Wav2Vec2Processor, Wav2Vec2Model
# Function to load the model
def load_transformers_model(model_path):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    model.eval()
    return processor, model


def extract_wav2vec_transformers(wavfile, savefile, processor, model, sample_rate=16000):
    try:
        # Read and preprocess audio file
        waveform, original_sample_rate = sf.read(wavfile)

        # Resample if necessary
        if original_sample_rate != sample_rate:
            waveform = signal.resample_poly(waveform, sample_rate, original_sample_rate)

        # Convert to mono if stereo
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Process with model
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model(**inputs).last_hidden_state

        # Save features in .mat file with key 'wav2'
        sio.savemat(savefile, {'wav2': features.numpy()})

        print(f"Saved features to {savefile}, shape: {features.shape}")

    except Exception as e:
        print(f"Error processing {wavfile}: {e}")

def extract_spec_batch(input_dir, output_dir, model, sample_rate=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    total = len(files)
    for i, file in enumerate(files):
        if not file.endswith('.wav'):
            continue
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file.split('.')[0] + '.mat')
        try:
            if not os.path.exists(output_file):
                extract_wav2vec_transformers(input_file, output_file, model, sample_rate)
            print(f'\rProcessed {i+1}/{total}', end=' ')
        except Exception as e:
            print(f'\nError processing {input_file}: {e}')

if __name__ == '__main__':
    model_path = '/home/xrl/speech/wav2vec2/model'  # Update this path
    processor, model = load_transformers_model(model_path)

    input_dir = '/home/xrl/speech/savee/wav'  # Update this path
    output_dir = '/home/xrl/speech/savee/wav_wav2vec2_mat'  # Update this path
    extract_spec_batch(input_dir, output_dir, processor, model)
