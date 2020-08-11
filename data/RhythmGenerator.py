from tensorflow.keras.utils import Sequence
from models import rhythm
from data import audio
from utils import OsuUtils, Logger
import numpy as np
import traceback
import random
import os
import multiprocessing
from tqdm import tqdm
import json

def transform_output(note_data, channel, key, min_density):
    x = note_data[:, :, channel].sum(axis=1) / key # (T, )
    x = np.reshape(x, (-1, 1)) # (T, 1)
    x = np.where(x == 0, x, x * (1 - min_density) + min_density)
    x = x * 2 - 1 # scale to [-1, 1]
    return x

def run(processes):
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def write_item(star_to_paths, config, paths):
    for path in paths:
        x, y = get_item(star_to_paths, config)
        np.savez(os.path.join("temp_data", path), x=x, y=y)

def get_item(star_to_paths, config):
    batch_size = config["rhythm_batch_size"]
    input_size = config["rhythm_input_length"]
    min_density = config["rhythm_min_density"]
    x = np.zeros((batch_size, input_size * 4, rhythm.get_input_channel(config)))
    y = np.zeros((batch_size, input_size, 2))
    sample_id = 0

    while True:
        try:
            star = random.choice(list(star_to_paths.keys()))
            osu_path = random.choice(star_to_paths[star])
            # Logger.log("select: " + osu_path)
            # TODO: check valid

            js = OsuUtils.parse_beatmap(osu_path, config)
            if not os.path.exists(js['bgm_path']) or (js['key'] != 4 and js['key'] != 7) or js['old_star'] <= 3 or not js['is_mania']:
                # Logger.log("Pass...")
                continue
            else:
                # Logger.log("choose!! " + osu_path)
                pass
            note_data = OsuUtils.beatmap_to_numpy(config, js) # (T, K, 2)
            raw_in = audio.gen_input(config, js, augment=True) # (T * 4, C)
            y0 = transform_output(note_data, 0, js['key'], min_density)
            y1 = transform_output(note_data, 1, js['key'], min_density)
            raw_out = np.concatenate((y0, y1), axis=1) # (T, 2)

            max_start = min(raw_out.shape[0] - input_size, raw_in.shape[0] // 4 - input_size)
            if max_start < 0:
                # Logger.log("too short (%d): %s" % (raw_out.shape[0], osu_path))
                continue

            start = np.random.randint(max_start + 1)
            x[sample_id] = raw_in[start * 4 : start * 4 + input_size * 4, :]
            y[sample_id] = raw_out[start : start + input_size, :]

            sample_id += 1
            if sample_id == batch_size:
                return x, y
        except:
            traceback.print_exc()
            pass

class RhythmTrainGenerator(Sequence):
    def __init__(self, osu_paths, config):
        self.osu_paths = osu_paths
        self.config = config
        if not os.path.exists("temp_data"):
            os.mkdir("temp_data")
        self.data_cache = os.listdir("temp_data")

        self.star_to_paths = {}
        for path in self.osu_paths:
            old_star, _, _ = OsuUtils.get_mania_star(path, os.path.abspath(os.path.join("out", "star_%d.txt") % random.randint(0, 1000000)))
            old_star = int(old_star)
            if old_star >= 5:
                old_star = 5
            if old_star in self.star_to_paths:
                self.star_to_paths[old_star].append(path)
            else:
                self.star_to_paths[old_star] = [path]
        for star, paths in self.star_to_paths.items():
            print("star: %d (%d)" % (star, len(paths)))

    def __len__(self):
        return len(self.osu_paths) // self.config["rhythm_batch_size"]

    def __getitem__(self, index):
        if len(self.data_cache) == 0:
            processes = []
            current_ids = "%d.npz" % (len(os.listdir("temp_data")))
            self.data_cache.append(current_ids)
            write_item(self.star_to_paths, self.config, [current_ids])
        npz_file = self.data_cache.pop()
        f = open(os.path.join("temp_data", npz_file), "rb")
        data = np.load(f)
        x = data["x"]
        y = data["y"]
        f.close()
        return x, y

    def __iter__(self):
        for i in range(len(self)):
            item = self[i]
            yield item

class RhythmValGenerator(Sequence):
    def __init__(self, osu_paths, config):
        # TODO: check valid
        self.osu_paths = osu_paths
        self.config = config

    def __len__(self):
        return len(self.osu_paths)

    def __getitem__(self, index):
        
        min_density = self.config["rhythm_min_density"]

        path = self.osu_paths[index]
        try:
            js = OsuUtils.parse_beatmap(path, self.config)
            note_data = OsuUtils.beatmap_to_numpy(self.config, js) # (T, K, 2)
            raw_in = audio.gen_input(self.config, js, augment=False) # (T * 4, C)
            y0 = transform_output(note_data, 0, js['key'], min_density)
            y1 = transform_output(note_data, 1, js['key'], min_density)
            raw_out = np.concatenate((y0, y1), axis=1) # (T, 2)

            size = min(raw_in.shape[0] // 4, raw_out.shape[0])
            raw_in = np.expand_dims(raw_in[:size * 4], axis=0)
            raw_out = np.expand_dims(raw_out[:size], axis=0)

            Logger.log("val: %s (%s) (%s)" % (path, str(raw_in.shape), str(raw_out.shape)))

            return [raw_in, raw_out]
        except:
            Logger.log("Val error in index %d (%s), traceback: " % (index, path) + traceback.format_exc())
            if index + 1 == len(self):
                return self[0]
            else:
                return self[index + 1]
            

    def __iter__(self):
        for i in range(len(self)):
            item = self[i]
            yield item