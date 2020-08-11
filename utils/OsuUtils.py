import os, json
# from filelock import Timeout, FileLock
import random, traceback
import numpy as np
from utils import MapUtils

_temp_file = os.path.abspath(os.path.join("out", "star_%d.txt") % random.randint(0, 1000000))
# print("use temp file: " + temp_file)
old_mania_command = 'java -jar "%s"' % (
    os.path.abspath(os.path.join("library", "old_mania_diff", "beatmap_analyzer.jar")))
new_mania_command = 'python "%s"' % (
    os.path.abspath(os.path.join("library", "new_mania_diff", "main.py")))


def read_item(line):
    return line.split(":")[-1].strip()


def parse_beatmap(osu_path, config, meta_only=False):
    data = open(osu_path, 'r', encoding='utf-8').read().split("\n")
    length = -1
    key = -1
    parsing_context = ""
    bpm = offset = -1
    notes = []  # [{ timestamp, column, duration }]
    column_width = 0
    is_mania = False
    for line in data:
        line = line.strip()

        if parsing_context == "[HitObjects]" and "," in line and not meta_only:
            assert key != -1
            params = line.split(",")
            column = int(int(float(params[0])) / column_width)
            timestamp = int(params[2])
            end = timestamp if int(params[3]) != 128 else (int(params[5].split(":")[0]))
            notes.append({
                'start': timestamp,
                'column': column,
                'end': end
            })
            length = max(length, end)

        elif parsing_context == "[General]":
            if line.startswith("AudioFilename"):
                bgm_path = os.path.join(os.path.dirname(os.path.realpath(osu_path)),
                                        read_item(line))
                # print(bgm_path)
                # assert os.path.exists(bgm_path)
            elif line.startswith("Mode"):
                is_mania = int(read_item(line)) == 3

        elif parsing_context == "[Metadata]":
            if line.startswith("BeatmapID"):
                beatmap_id = read_item(line)
            elif line.startswith("BeatmapSetID"):
                beatmap_set_id = read_item(line)

        elif parsing_context == "[Difficulty]":
            if line.startswith("CircleSize"):
                key = float(read_item(line))
                column_width = int(512 / key)

        elif parsing_context == "[TimingPoints]":
            if "," in line and bpm == -1:
                params = line.split(",")
                offset = int(float(params[0]))
                bpm = 60000 / float(params[1])
                if bpm > 0:
                    # restrict bpm to [max_bpm / 2, max_bpm]
                    max_bpm = config['max_bpm']
                    while bpm > max_bpm:
                        bpm = bpm / 2
                    while bpm < max_bpm / 2:
                        bpm = bpm * 2

                    dt = 60000 / bpm / config['wav_dt_split']
                else:
                    bpm = -1

        if line.startswith("["):
            parsing_context = line

    if is_mania:
        old, new, _ = get_mania_star(osu_path, os.path.abspath(
            os.path.join("out", "star_%d.txt") % random.randint(0, 1000000)))
    else:
        old = new = -1
    notes = sorted(notes, key=lambda x: x['start'])

    if not meta_only:
        offset = min(offset, notes[0]['start'])
        while offset >= dt:
            offset -= dt

    return {
        "length": length,
        "key": key,
        "is_mania": is_mania,
        "bpm": bpm,
        "dt": dt,
        "offset": offset,
        "beatmap_id": beatmap_id,
        "beatmap_set_id": beatmap_set_id,
        "bgm_path": bgm_path,
        "old_star": old,
        "new_star": new,
        "notes": notes
    }


def beatmap_to_numpy(config, osu_json):
    return MapUtils.beatmap_to_numpy(config, osu_json)


# return: (old star, new star, cache) or (-1, -1, cache) if it's not a valid mania map or too long/short
def get_mania_star(osu_path, temp_file=None, force_cal_new=False):
    if temp_file is None:
        temp_file = _temp_file
    dir_name = os.path.abspath(os.path.dirname(osu_path))
    map_name = os.path.basename(osu_path)
    js_name = os.path.join(dir_name, "data.json")
    data = {}
    if os.path.exists(js_name):
        data = json.load(open(js_name))
    if map_name not in data:
        data[map_name] = {}
    if 'old_star' in data[map_name] and 'new_star' in data[map_name]:
        return data[map_name]['old_star'], data[map_name]['new_star'], True
    old_star = _get_mania_star(osu_path, old_mania_command, temp_file)
    if old_star == -1 or not force_cal_new:
        new_star = -1
    else:
        if os.path.exists(osu_path + "_lock_"):
            return (-1, -1, False)
        with open(osu_path + "_lock_", "w") as f:
            f.write(" ")
        try:
            new_star = _get_mania_star(osu_path, new_mania_command, temp_file)
        finally:
            if os.path.exists(osu_path + "_lock_"):
                os.remove(osu_path + "_lock_")
    data = {}
    if os.path.exists(js_name):
        data = json.load(open(js_name))
    if map_name not in data:
        data[map_name] = {}
    data[map_name]['old_star'] = old_star
    data[map_name]['new_star'] = new_star
    # lock = FileLock(js_name + ".lock")
    # lock.acquire()
    with open(js_name, "w") as ff:
        # fcntl.flock(ff, fcntl.LOCK_EX)
        try:
            json.dump(data, ff)
        finally:
            pass
            # fcntl.flock(ff, fcntl.LOCK_UN)
    if old_star == -1:
        raise ValueError("")
    return old_star, new_star, False


def _get_mania_star(osu_path, command, temp_file):
    if os.path.exists(temp_file):
        os.remove(temp_file)
    os.system(command + ' "%s" "%s"' % (osu_path, temp_file))
    if os.path.exists(temp_file):
        result = float(open(temp_file).read())
        os.remove(temp_file)
    else:
        # raise ValueError("Not found star in " + osu_path)
        return -1
    return result


if __name__ == "__main__":
    data = (parse_beatmap(
        "out\\songs\\89535 sun3 - Messier 333\\sun3 - Messier 333 (MOONWOLF) [2D's 6K Normal].osu"))

    # json.dump(data, open(os.path.join("out", "test_beatmap.json"), "w"))

    note_data = beatmap_to_numpy(data)
    np.savetxt(os.path.join("out", "test_beatmap_note.txt"), note_data[:, :, 0], fmt="%d")
    np.savetxt(os.path.join("out", "test_beatmap_note_ln.txt"), note_data[:, :, 1], fmt="%d")
    np.savetxt(os.path.join("out", "test_beatmap_note_sun.txt"),
               note_data[:, :, 0].sum(axis=1) / data['key'], fmt="%.3f")
