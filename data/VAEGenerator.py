import os
import pickle
import random
import traceback

import numpy as np
from func_timeout import func_set_timeout
from tensorflow.keras.utils import Sequence

from data import audio
from utils import OsuUtils, MapUtils, Logger


@func_set_timeout(60)
def get_item_by_path(js, config, key, kl_loss_weight=0, strength_teacher_p=1.0, is_train=True):
    min_density = config["rhythm_min_density"]
    input_length = config['vae_input_length']
    # strength_weight = np.expand_dims(MapUtils.get_index_to_strength_dict(key, bias=min_density),
    #                                  axis=0)
    kl_loss_weight = np.reshape(kl_loss_weight, (1, 1))
    strength_teacher_p = np.reshape(strength_teacher_p, (1, 1))

    raw_note_data = OsuUtils.beatmap_to_numpy(config, js)  # (T, K, 2)
    raw_audio_input = audio.from_osu_json(config, js, is_train)  # (T * 4, 128)
    note_data = MapUtils.format_length(raw_note_data, input_length)
    audio_input = MapUtils.format_length(raw_audio_input, input_length * 4)
    # Logger.log("%d -> %d, %d -> %d" % (raw_note_data.shape[0], note_data.shape[0], raw_audio_input.shape[0], audio_input.shape[0]))
    # note_data = note_data[:t, :, :]
    # audio_input = audio_input[:t * 4, :]
    note_data_augment = MapUtils.augment_map(note_data, reorder=False)

    y_true = MapUtils.array_to_index(note_data_augment, key, one_hot=True)
    strength_true = np.mean(note_data, axis=1)  # (T, 2)
    strength_true = np.where(strength_true == 0, -1, strength_true)
    pre_notes_input = MapUtils.array_to_index(note_data_augment, key, one_hot=False)
    pre_notes_input[1:, 0] = pre_notes_input[:input_length - 1, 0]  # right shift
    pre_notes_input[0, 0] = 0
    rhythm_base = MapUtils.get_beatmap_base_rhythm(config, note_data)
    style_input = MapUtils.augment_map(note_data, reorder=True)

    # debug only

    np.savetxt("out\\train\\style_input.txt",
               style_input[:, :, 0] + style_input[:, :, 1] * 10, "%.3lf")
    np.savetxt("out\\train\\audio_input.txt", audio_input, "%.3lf")
    np.savetxt("out\\train\\rhythm_base.txt", rhythm_base, "%.3lf")
    np.savetxt("out\\train\\pre_notes_input.txt", pre_notes_input, "%.3lf")
    np.savetxt("out\\train\\y_true.txt", y_true, "%.3lf")
    np.savetxt("out\\train\\strength_true.txt", strength_true, "%.3lf")

    # reshape
    style_input = np.expand_dims(style_input, axis=0)
    audio_input = np.expand_dims(audio_input, axis=0)
    rhythm_base = np.expand_dims(rhythm_base, axis=0)
    pre_notes_input = np.expand_dims(pre_notes_input, axis=0)
    y_true = np.expand_dims(y_true, axis=0)
    strength_true = np.expand_dims(strength_true, axis=0)

    return [style_input,
            audio_input,
            rhythm_base,
            kl_loss_weight,
            pre_notes_input,
            y_true,
            strength_true,
            strength_teacher_p
            ], y_true

def get_item(star_to_paths, config, key, kl_loss_weight=0, teacher_forcing_p=1.0):
    osu_path = ""
    while True:
        try:
            star = random.choice(list(star_to_paths.keys()))
            osu_path = random.choice(star_to_paths[star])
            # Logger.log("select: " + osu_path)

            js = OsuUtils.parse_beatmap(osu_path, config)
            if not os.path.exists(js['bgm_path']) or js['key'] != key \
                    or js['old_star'] <= 3 or not js['is_mania']:
                # Logger.log("Pass...")
                continue
            else:
                # Logger.log("choose!! " + osu_path)
                pass

            return get_item_by_path(js, config, key, kl_loss_weight, teacher_forcing_p)
        except Exception as exc:
            Logger.log("Error: " + str(exc.__class__))
            Logger.log("Error in path: " + osu_path)
            pass


def write_item(star_to_paths, config, paths, key, kl_loss_weight):
    for path in paths:
        x, y = get_item(star_to_paths, config, key, kl_loss_weight)
        with open(path, 'wb') as fo:
            pickle.dump((x, y), fo)


class VAETrainGenerator(Sequence):
    def __init__(self, osu_paths, config, key, init_step=0):
        self.osu_paths = osu_paths
        self.config = config
        self.total_step = config['vae_step_per_epoch'] * config['vae_epoch']
        self.current_step = init_step
        self.key = key

        self.star_to_paths = {}
        for path, old_star in self.osu_paths:
            # old_star, _, _ = OsuUtils.get_mania_star(path, os.path.abspath(
            #     os.path.join("out", "star_%d.txt") % random.randint(0, 1000000)))
            old_star = int(old_star)
            if old_star >= 6:
                old_star = 6
            if old_star in self.star_to_paths:
                self.star_to_paths[old_star].append(path)
            else:
                self.star_to_paths[old_star] = [path]
        for star, paths in self.star_to_paths.items():
            print("star: %d (%d)" % (star, len(paths)))

    def __len__(self):
        return self.config['vae_step_per_epoch']

    def __getitem__(self, index):
        # decay teacher forcing probability
        p = np.clip(1 - self.current_step / float(self.total_step) * 2, 0, 1)
        # p = 0
        # Logger.log("teacher forcing decay: %lf (%d / %d)" % (p, self.current_step, self.total_step))
        self.current_step += 1

        x, y = get_item(self.star_to_paths, self.config, self.key, teacher_forcing_p=p)
        return x, None

    def __iter__(self):
        for i in range(len(self)):
            item = self[i]
            yield item


class VAETestGenerator(Sequence):
    def __init__(self, osu_paths, config, key):
        self.osu_paths = osu_paths
        self.config = config
        self.key = key

    def __len__(self):
        return len(self.osu_paths)

    def __getitem__(self, index):
        path = self.osu_paths[index][0]
        js = OsuUtils.parse_beatmap(path, self.config)
        x, y = get_item_by_path(js, self.config, self.key, is_train=False, strength_teacher_p=0.0)
        return x, None

    def __iter__(self):
        for i in range(len(self)):
            item = self[i]
            yield item


if __name__ == "__main__":
    import json

    path = {4: ["G:\\E\osu!\\Songs\\324992 ARM feat Nanahira - BakunanaTestroyer\\"
                "ARM feat. Nanahira - BakunanaTestroyer (Rinzler) [BEAST].osu"]}
    config = json.load(open("conf\\base_config.json"))
    get_item(path, config, 4)
