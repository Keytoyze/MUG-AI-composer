import os
import random
import time

import librosa
import numpy as np


def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum()
    den = exist.sum()
    return num / den


def from_osu_json(config, osu_json, augment=False):
    base = 4000 / osu_json['dt']
    # t = time.time()
    if os.path.exists(osu_json['bgm_path'] + "_transform.wav"):
        os.remove(osu_json['bgm_path'] + "_transform.wav")
    cache_array = osu_json['bgm_path'] + ".npz"
    if os.path.exists(cache_array):
        data = np.load(cache_array)
        y, sr = data['y'], data['sr']
        sr = sr[0]
    else:
        y, sr = librosa.load(osu_json['bgm_path'], offset=osu_json['offset'] / 1000,
                             sr=int(int(44100 / base) * base))
        y = librosa.feature.melspectrogram(y, sr,
                                           n_mels=config['wav_feature'],
                                           hop_length=int(osu_json['dt'] / 1000 * sr / 4),
                                           n_fft=int(config['wav_window'] * sr)
                                           )
        np.savez(cache_array, y=y, sr=[sr])
    # print(time.time() - t)
    # t = time.time()
    if augment:
        augment_type = random.randint(0, 2)
        if augment_type == 0:
            # noise augment
            max_noise = non_zero_mean(y) / 20  # around 5% noise
            noise = random.uniform(0, max_noise)
            wn = np.random.randn(*y.shape)
            y = np.where(y > 1e-4, y + abs(noise * wn), 0.0)
        elif augment_type == 1:
            # pitch augment
            nb_cols = y.shape[0]
            max_shifts = nb_cols // 20  # around 5% shift
            nb_shifts = np.random.randint(-max_shifts, max_shifts)
            y = np.roll(y, nb_shifts, axis=0)
    # print(time.time() - t)
    y = np.swapaxes(y, 0, 1)
    return y


def gen_input(config, osu_json, augment=False):
    x = from_osu_json(config, osu_json, augment)
    extra = np.zeros((x.shape[0], 3))
    extra[:, 0] = osu_json['old_star']
    extra[:, 1] = (osu_json['bpm'] - config['max_bpm'] / 2) / (config['max_bpm'] / 2)
    extra[:, 2] = 1 if osu_json['key'] == 4 else 0
    return np.concatenate((x, extra), 1)
