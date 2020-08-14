import numpy as np
import tensorflow as tf
import random
import math
from data import audio
from utils import OsuUtils


# Input: (key, 2)
# Output: index
def note_to_index(array, key):
    num = 0
    base = 1
    for c in range(key):
        if array[c, 1] != 0:  # LN holding
            num += base * 2
        elif array[c, 0] != 0:  # normal note
            num += base * 1
        base *= 3
    return num


# Input: index
# Output: (key, 2)
def index_to_note(index, key):
    array = np.zeros((key, 2))
    for c in range(key):
        if index % 3 == 2:  # LN holding
            array[c, 1] = 1
        elif index % 3 == 1:  # normal nte
            array[c, 0] = 1
        index //= 3
    return array


# Input: (T, key, 2)
# Output: (T, 1) | (T, 3 ** key)
def array_to_index(array, key, one_hot=False):
    output = np.zeros((array.shape[0], 3 ** key if one_hot else 1))
    for t in range(array.shape[0]):
        index = note_to_index(array[t, :, :], key)
        if one_hot:
            output[t, index] = 1
        else:
            output[t, 0] = index
    return output


# return: [ strength [0, 1] ] * 3 ** key
def get_index_to_strength_dict(key, ln_ratio=0.3, bias=0.5):
    result = np.zeros((3 ** key,))
    for c in range(3 ** key):
        data = sum(index_to_note(c, key)[:, 0]) / key
        if data != 0:
            data = bias + (1 - bias) * data
        data = data + ln_ratio * sum(index_to_note(c, key)[:, 1]) / key
        result[c] = data
    return tf.constant(result)


# Input: (T, K, 2)
# return (T, 1)
def get_beatmap_base_rhythm(config, array):
    num = config['wav_dt_split']
    backet = [0] * num
    rhythm_array = np.sum(array, axis=1)[:, 0]  # (T, )
    for i, strength in enumerate(rhythm_array):
        backet[i % num] += strength
    base_index = np.argmax(backet)
    result = np.zeros(rhythm_array.shape[0])
    # noinspection PyTypeChecker
    weights = [1, -1, -1 / 3, -1, 1 / 3, -1, -1 / 3, -1]
    for i in range(result.shape[0]):
        result[i] = weights[(i - base_index) % 8]
    return np.expand_dims(result, axis=1)


def find_map_interval(note_data, window_size):
    splitter = np.zeros((window_size, note_data.shape[1], 2))
    indices = []
    i = 0
    T = note_data.shape[0]
    while i < T - window_size:
        # print(note_data[i:(i+8), :, :])
        # print(splitter)
        if (note_data[i:(i + window_size), :, :] == splitter).all():
            indices.append(i)
            i += window_size
        else:
            i += 1
    indices = [indices[i] for i in range(len(indices)) if
               i == 0 or indices[i] - indices[i - 1] > window_size]
    for _ in range(2):
        r = random.randint(1, T - 2)
        if r not in indices:
            indices.append(r)
    indices = sorted(indices)
    return indices


# Input / Output: (T, K, 2)
def augment_map(note_data, reorder=False):
    if random.randint(0, 1) == 0:
        # mirror
        note_data = note_data[:, ::-1, :]
    if reorder:
        indices = find_map_interval(note_data, 16)
        if len(indices) <= 5:
            indices = find_map_interval(note_data, 8)
        arrays = np.split(note_data, indices, axis=0)
        random.shuffle(arrays)
        note_data = np.concatenate(arrays, axis=0)

    return note_data


# return (T, K, 2)
# T: timestamp from offset, dt = 60 / bpm / 4 (seconds)
# K: keys in [t, t + dt)
# 2: channel for no LN / LN
def beatmap_to_numpy(config, osu_json):
    dt = osu_json['dt']
    t = osu_json['offset']
    notes = osu_json['notes']
    key = int(osu_json['key'])
    length = osu_json['length']
    holding_end = [0] * key
    data = []
    index = 0
    while t < length:
        current_note = [0] * key
        current_ln = [1 if holding_end[k] > t else 0 for k in range(key)]
        if index < len(notes):
            for i in range(index, len(notes)):
                note = notes[i]
                start = note['start']
                if start >= t + dt:
                    break
                elif start >= t:
                    column = note['column']
                    current_note[column] = 1
                    holding_end[column] = max(holding_end[column], note['end'])
                    index = i + 1
                else:
                    assert False  # impossible!!
        t += dt
        data.append((current_note, current_ln))
    data = np.asarray(data, dtype=np.int8)  # (T, 2, K)
    data = np.swapaxes(data, 1, 2)  # (T, K, 2)

    return data


def numpy_to_beatmap(array, dt, offset):
    K = array.shape[1]
    T = array.shape[0]
    notes = []
    holding = [-1] * K
    t = 0
    for i in range(T):
        t = i * dt + offset
        for j in range(K):
            if array[i][j][0] == 1:  # start
                if holding[j] != -1:
                    notes.append({
                        'start': holding[j],
                        'column': j,
                        'end': t
                    })
                    holding[j] = -1
                holding[j] = t
            elif array[i][j][1] != 1:  # no LN hold, no start
                if holding[j] != -1:
                    notes.append({
                        'start': holding[j],
                        'column': j,
                        'end': t
                    })
                    holding[j] = -1
    for j in range(K):
        if holding[j] != -1:
            notes.append({
                'start': holding[j],
                'column': j,
                'end': t
            })
    return notes


def format_length(np_array, length):
    if np_array.shape[0] > length:
        if len(np_array.shape) == 3:
            return np_array[:length, :, :]
        else:
            return np_array[:length, :]
    else:
        shape = list(np_array.shape)
        shape[0] = length - shape[0]
        zeros = np.zeros(shape)
        return np.concatenate([np_array, zeros], axis=0)


def compose(vae_model, config, base_js, audio_js):
    base_input_length = config['vae_input_length']
    raw_base_data = OsuUtils.beatmap_to_numpy(config, base_js)  # (T, K, 2)
    raw_audio_input = audio.from_osu_json(config, audio_js, False)  # (T * 4, 128)
    raw_audio_note = OsuUtils.beatmap_to_numpy(config, audio_js)
    input_length = raw_audio_input.shape[0] // 4

    base_note = format_length(raw_base_data, base_input_length)
    audio_input = format_length(raw_audio_input, input_length * 4)
    audio_note = format_length(raw_audio_note, input_length)
    rhythm_base = get_beatmap_base_rhythm(config, audio_note)

    indices = vae_model.predict(base_note, audio_input, rhythm_base)
    result = []
    for index in indices:
        result.append(index_to_note(index, vae_model.key))
    result = np.array(result)
    return result


if __name__ == "__main__":
    for i in range(3 ** 4):
        array = index_to_note(i, 4)
        print(i, array, note_to_index(array, 4))
    # print(get_index_to_strength_dict(4))
    # print(get_index_to_strength_dict(4))
