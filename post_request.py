from preprocess import resample_single
import requests
from convert import get_batch_test_data, load_wav
import numpy as np
import librosa

sampling_rate, num_mcep, frame_period = 16000, 36, 5
# Read a batch of testdata
filename = 'p0_999_2.wav'
resample_single(filename, 'temp.wav')
test_wavfiles = get_batch_test_data('temp.wav')
test_wavs = [list(load_wav(wavfile, sampling_rate).astype(float)) for wavfile in test_wavfiles]

datadict = {'test_wavs': test_wavs, 'src_spk': 'p0', 'trg_spk': 'p3'}
r = requests.post('http://103.137.4.6:4000/VoiceConversion/v1', json=datadict)
wav_transformed = r.json()['converted']
wav_transformed = np.array(wav_transformed, dtype="float32")
librosa.output.write_wav('test.wav', wav_transformed, sampling_rate)
print('done')


