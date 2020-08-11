import argparse
from model import Generator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
import glob
from preprocess import resample_single

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


class TestDataset(object):
    """Dataset for testing."""

    def __init__(self, config):
        assert config.trg_spk in speakers, f'The trg_spk should be chosen from {speakers}, but you choose {trg_spk}.'
        # Source speaker
        self.src_spk = config.src_spk
        self.trg_spk = config.trg_spk

        # self.mc_files = sorted(glob.glob(join(config.test_data_dir, f'{config.src_spk}*.npy')))
        self.src_spk_stats = np.load(join(config.train_data_dir, f'{config.src_spk}_stats.npz'))
        self.src_wav_dir = f'{config.wav_dir}/{config.src_spk}'

        self.trg_spk_stats = np.load(join(config.train_data_dir, f'{config.trg_spk}_stats.npz'))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']

        self.spk_idx = spk2idx[config.trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, filename):
        # filename la file npy
        batch_data = []
        batch_data.append(filename)
        return batch_data


def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple=4)  # TODO
    # return wav

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def test(config):
    os.makedirs(join(config.convert_dir, str(config.resume_iters)), exist_ok=True)
    sampling_rate, num_mcep, frame_period = 16000, 36, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator().to(device)
    test_loader = TestDataset(config)
    # Restore model
    print(f'Loading the trained models from step {config.resume_iters}...')
    G_path = join(config.model_save_dir, f'{config.resume_iters}-G.ckpt')
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    # Read a batch of testdata
    resample_single(config.filename, 'temp.wav')
    test_wavfiles = test_loader.get_batch_test_data('temp.wav')
    test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in test_wavfiles]

    with torch.no_grad():
        for idx, wav in enumerate(test_wavs):
            wav_name = basename(test_wavfiles[idx])
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = pitch_conversion(f0=f0,
                                            mean_log_src=test_loader.logf0s_mean_src,
                                            std_log_src=test_loader.logf0s_std_src,
                                            mean_log_target=test_loader.logf0s_mean_trg,
                                            std_log_target=test_loader.logf0s_std_trg)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            print("Before being fed into G: ", coded_sp.shape)
            coded_sp_norm = (coded_sp - test_loader.mcep_mean_src) / test_loader.mcep_std_src
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
            spk_conds = torch.FloatTensor(test_loader.spk_c_trg).to(device)
            # print(spk_conds.size())
            coded_sp_converted_norm = G(coded_sp_norm_tensor, spk_conds).data.cpu().numpy()
            coded_sp_converted = np.squeeze(
                coded_sp_converted_norm).T * test_loader.mcep_std_trg + test_loader.mcep_mean_trg
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            print("After being fed into G: ", coded_sp_converted.shape)
            wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                     ap=ap, fs=sampling_rate, frame_period=frame_period)
            wav_id = wav_name.split('.')[0]
            librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters),
                                          f'{wav_id}-vcto-{test_loader.trg_spk}.wav'), wav_transformed, sampling_rate)
            if [True, False][0]:
                wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
                                                   ap=ap, fs=sampling_rate, frame_period=frame_period)
                librosa.output.write_wav(join(config.convert_dir, str(config.resume_iters), f'cpsyn-{wav_name}'),
                                         wav_cpsyn, sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_speakers', type=int, default=11, help='dimension of speaker labels')
    # parser.add_argument('--num_converted_wavs', type=int, default=2, help='number of wavs to convert.')
    parser.add_argument('--resume_iters', type=int, default=None, help='step to resume for testing.')
    parser.add_argument('--src_spk', type=str, default='p0', help='target speaker.')
    parser.add_argument('--trg_spk', type=str, default='p2', help='target speaker.')
    parser.add_argument('--filename', type=str, help='voice')

    parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')

    parser.add_argument('--wav_dir', type=str, default="./data/sontung/wav16")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--convert_dir', type=str, default='./converted')

    config = parser.parse_args()

    print(config)
    if config.resume_iters is None:
        raise RuntimeError("Please specify the step number for resuming.")
    test(config)
