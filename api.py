import argparse
from model import Generator
import torch
from os.path import join
from data_loader import to_categorical
from utils import *
import numpy as np
from flask import Flask, jsonify
from flask import request
import json
from gevent.pywsgi import WSGIServer
import time

app = Flask(__name__)

# Below is the accent info for the used 10 speakers.

spk2acc = {'0': 'baomoi',  # F
           '1': 'sontung', #M
           '2': 'zalo',  # M
           '3': 'baomoi',  # F
           '4': 'fpt',  # M
           '5': 'fpt',  # F
           '6': 'truyen',  # M
           '7': 'truyen',  # F
           '8': 'truyen',  # M
           '9': 'truyen',  # F
           '10': 'truyen'}  # M
speakers = ['p0','p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
# speakers = ['p0', 'p3', 'p5', 'p7', 'p9']
spk2idx = dict(zip(speakers, range(len(speakers))))
resume_iters = 400000
train_data_dir = './data/mc/train'
model_save_dir = './models'

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def convert(G):
    if request.method == "POST":
        dataDict = json.loads(request.data.decode('utf-8'))
        test_wavs = np.array(dataDict.get('test_wavs', None), dtype="float32")
        src_spk = dataDict.get('src_spk', None)
        trg_spk = dataDict.get('trg_spk', None)

    start = time.time()
    sampling_rate, num_mcep, frame_period = 16000, 36, 5

    src_spk_stats = np.load(join(train_data_dir, f'{src_spk}_stats.npz'))

    trg_spk_stats = np.load(join(train_data_dir, f'{trg_spk}_stats.npz'))

    logf0s_mean_src = src_spk_stats['log_f0s_mean']
    logf0s_std_src = src_spk_stats['log_f0s_std']
    logf0s_mean_trg = trg_spk_stats['log_f0s_mean']
    logf0s_std_trg = trg_spk_stats['log_f0s_std']
    mcep_mean_src = src_spk_stats['coded_sps_mean']
    mcep_std_src = src_spk_stats['coded_sps_std']
    mcep_mean_trg = trg_spk_stats['coded_sps_mean']
    mcep_std_trg = trg_spk_stats['coded_sps_std']

    spk_idx = spk2idx[trg_spk]
    spk_cat = to_categorical([spk_idx], num_classes=len(speakers))
    spk_c_trg = spk_cat

    with torch.no_grad():
        for idx, wav in enumerate(test_wavs):
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0,
                                            mean_log_src=logf0s_mean_src,
                                            std_log_src=logf0s_std_src,
                                            mean_log_target=logf0s_mean_trg,
                                            std_log_target=logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            print("Before being fed into G: ", coded_sp.shape)
            coded_sp_norm = (coded_sp - mcep_mean_src) / mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
            spk_conds = torch.FloatTensor(spk_c_trg).to(device)
            coded_sp_converted_norm = G(coded_sp_norm_tensor, spk_conds).data.cpu().numpy()
            coded_sp_converted = np.squeeze(
                coded_sp_converted_norm).T * mcep_std_trg + mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            print("After being fed into G: ", coded_sp_converted.shape)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                     ap=ap, fs=sampling_rate, frame_period=frame_period)
            # remove noise
            wav_transformed = smooth(wav_transformed, 5)
            # wav_transformed = np.array(wav_transformed, dtype="float32")
    end = time.time() - start

    response = jsonify({"converted": wav_transformed.tolist(), \
                        "time": end, "status_code": 200})
    response.status_code = 200
    response.status = 'OK'
    return response, 200

@app.route('/VoiceConversion/v1', methods=['POST'])
def convert_():
    return convert(G)

if __name__ == '__main__':
    # Restore model
    print(f'Loading the trained models from step {resume_iters}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    G_path = join(model_save_dir, f'{resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    http_server = WSGIServer(('', 4000), app)
    http_server.serve_forever()

