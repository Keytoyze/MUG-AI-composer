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
def get_item_by_path(js, config, key, kl_loss_weight=0, teacher_forcing_p=1.0, is_train=True):
    min_density = config["rhythm_min_density"]
    input_length = config['vae_input_length']
    strength_weight = np.expand_dims(MapUtils.get_index_to_strength_dict(key, bias=min_density),
                                     axis=0)
    kl_loss_weight = np.reshape(kl_loss_weight, (1, 1))

    raw_note_data = OsuUtils.beatmap_to_numpy(config, js)  # (T, K, 2)
    raw_audio_input = audio.from_osu_json(config, js, is_train)  # (T * 4, 128)
    note_data = MapUtils.format_length(raw_note_data, input_length)
    audio_input = MapUtils.format_length(raw_audio_input, input_length * 4)
    # Logger.log("%d -> %d, %d -> %d" % (raw_note_data.shape[0], note_data.shape[0], raw_audio_input.shape[0], audio_input.shape[0]))
    # note_data = note_data[:t, :, :]
    # audio_input = audio_input[:t * 4, :]
    note_data_decoder = MapUtils.augment_map(note_data, reorder=False)

    y_true = MapUtils.array_to_index(note_data_decoder, key, one_hot=True)
    decoder_input = MapUtils.array_to_index(note_data_decoder, key, one_hot=False)
    if is_train:
        decoder_input[1:, 0] = decoder_input[:input_length - 1, 0]  # right shift
        decoder_input[0, 0] = 0
        decoder_mask = np.random.choice([0, 1], decoder_input.shape,
                                        p=[1 - teacher_forcing_p, teacher_forcing_p])
        decoder_input = decoder_input * decoder_mask
    else:
        decoder_input = np.zeros_like(decoder_input)
    rhythm_base = MapUtils.get_beatmap_base_rhythm(config, note_data)
    encoder_input = MapUtils.augment_map(note_data, reorder=True)

    # debug only
    # np.savetxt("out\\train\\encoder_input.txt",
    #            encoder_input[:, :, 0] + encoder_input[:, :, 1] * 10, "%.3lf")
    # np.savetxt("out\\train\\audio_input.txt", audio_input, "%.3lf")
    # np.savetxt("out\\train\\rhythm_base.txt", rhythm_base, "%.3lf")
    # np.savetxt("out\\train\\decoder_input.txt", decoder_input, "%.3lf")
    # np.savetxt("out\\train\\y_true.txt", y_true, "%.3lf")
    # np.savetxt("out\\train\\strength_weight.txt", strength_weight, "%.3lf")
    # np.savetxt("out\\train\\kl_loss_weight.txt", kl_loss_weight, "%.3lf")

    # reshape
    encoder_input = np.expand_dims(encoder_input, axis=0)
    audio_input = np.expand_dims(audio_input, axis=0)
    rhythm_base = np.expand_dims(rhythm_base, axis=0)
    decoder_input = np.expand_dims(decoder_input, axis=0)
    y_true = np.expand_dims(y_true, axis=0)

    return [encoder_input,
            audio_input,
            rhythm_base,
            kl_loss_weight,
            decoder_input,
            strength_weight,
            y_true
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
        p = 1 - self.current_step / float(self.total_step)
        Logger.log("teacher forcing decay: %lf (%d / %d)" % (p, self.current_step, self.total_step))
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
        x, y = get_item_by_path(js, self.config, self.key, is_train=False)
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
