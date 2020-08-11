
def traverse_all_diff(process_id):
    base_path = os.path.join("out", "songs")
    total = 0
    paths = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith(".osu"):
                paths.append(os.path.join(root, f))
                total += 1
    import random
    random.shuffle(paths)
    temp = os.path.abspath(os.path.join("out", "star_%d.txt") % random.randint(0, 1000000))
    # print(("[%d] " % process_id) + "temp: " + temp)
    finish = 0
    recent_time = []
    
    from timeit import default_timer as timer
    import datetime
    for full_path in paths:
        # print(("[%d] " % process_id) + full_path)
        try:
            t = timer()
            old, new, cache = get_mania_star(full_path, temp)
            d = timer() - t
            finish += 1
            recent_time.append(d)
            total_time = sum(recent_time)
            total_times = len(recent_time)
            if len(recent_time) >= 100:
                recent_time.pop(0)
            if not cache and old != -1:
                # print(("[%d] " % process_id) + str(old) + " " + str(new))
                # print(("[%d] " % process_id) + "total = %d, finish = %d, progress = %.2lf%%" % (total, finish, finish / total * 100))
                # print(("[%d] " % process_id) + "total_time = %.2lf s, total_times = %d" % (total_time, total_times))
                remain = (total - finish) * (total_time / total_times)
                now = datetime.datetime.now()
                delta = datetime.timedelta(seconds=remain)
                n_days = now + delta
                end_at = n_days.strftime('%Y-%m-%d %H:%M:%S')
                # print(("[%d] " % process_id) + "duration = %.2lf s, remain = %.2lf s, end at: %s" % (d, remain, end_at))

                
        except KeyboardInterrupt:
            exit()
        except:
            print("Error occurs in " + full_path)
            traceback.print_exc()
            print("============================================")

def new_star_statistics(base_path):
    x = []
    y = []
    beatmap = []
    beatset = []
    key = []
    sx = 0
    sy = 0
    color_map = ["r", "tan", "m", "b", "g", "y"]
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f == "data.json":
                full_path = os.path.join(root, f)
                js = json.load(open(full_path))
                for map, d in js.items():
                    if d['new_star'] != -1:
                        p = os.path.join(root, map)
                        if os.path.exists(p):
                            info = parse_beatmap(p)
                            if info['is_mania']:
                                x.append(d['old_star'])
                                y.append(d['new_star'])
                                key.append(int(info['key']))
                                beatmap.append(map)
                                beatset.append(os.path.basename(root))
                                sx += d['new_star']
                                sy += d['old_star']
                    # if d['new_star'] > 8:
                    #     print(full_path, d, map)
                # print(full_path)
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches
    plt.xlabel("ppy's algo")
    plt.xlim(0, 11)
    plt.ylim(0, 11)
    plt.ylabel("xxy's algo")
    patches = [mpatches.Patch(color=color_map[i-4], label="{:s}K".format(str(i))) for i in range(4, 10)]
    ax=plt.gca()
    ax.legend(handles=patches)
    rate = sy / sx
    print(sy, sx, rate)
    diag_line, = plt.plot([0, 11], [0, 11], ls="--", c=".3")

    plt.scatter(x, y, s=3, color=[color_map[x - 4] for x in key])
    print(len(x))

    import csv
    with open(os.path.join("out", "test.csv"), "w", newline='') as f:
        fcsv = csv.writer(f)
        fcsv.writerow(["dir", "file", "key", "old_star", "new_star"])
        for i in range(len(x)):
            fcsv.writerow([beatset[i], beatmap[i], str(key[i]), str(x[i]), str(y[i])])
    plt.show()


if __name__ == "__main__":
    # print(parse_beatmap("G:\\E\\osu!\\Songs\\429911  - Sendan Life\\Remo Prototype[CV Hanamori Yumiri] - Sendan Life (Hestiaaa) [Insane].osu"))
    # traverse_all_diff()
    # from multiprocessing import Process
    # for i in range(40):
    #     p = Process(target=traverse_all_diff, args=(i,))
    #     p.start()
    # import shutil
    # def copy(path):
    #     dir_name = os.path.dirname(path).replace("G:\\E\\osu!\\Songs\\", "")
    #     out_dir = os.path.join("out", "songs", dir_name)
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    #     shutil.copy(path, os.path.join(out_dir, os.path.basename(path)))
    # for root, dirs, files in os.walk("G:\\E\\osu!\\Songs"):
    #     for f in files:
    #         if f.endswith(".osu"):
    #             js_name = os.path.join(root, "data.json")
    #             full_path = os.path.join(root, f)
    #             if not os.path.exists(js_name):
    #                 copy(full_path)
    #             else:
    #                 data = json.load(open(js_name))
    #                 if f not in data:
    #                     copy(full_path)
    #                     copy(js_name)

    new_star_statistics("G:\\E\\osu!\\Songs")

# error: G:\E\osu!\Songs\338726  - Gems Pack 2 - Disaster\gems - Gems Pack 2 - Disaster (gemboyong) [16 - sa10, banana man].osu

# Groove Coverage Feat Fantasy T - God Is A Girl (Trance Remix)\Groove Coverage Feat. Fantasy T - God Is A Girl (Trance Remix) (Mr.Azaness007) [7k - hard lvl 30].osu
# out/songs/535277 Ariabl'eyeS - Kegare Naki Bara Juuji (Short ver)/Ariabl'eyeS - Kegare Naki Bara Juuji (Short ver.) (My Angel RangE) [Left's Insane].os

# out/songs/535277 Ariabl'eyeS - Kegare Naki Bara Juuji (Short ver)/Ariabl'eyeS - Kegare Naki Bara Juuji (Short ver.) (My Angel RangE) [Left's Insane].osTraceback (most recent call last):

# out/songs/482869 KISIDA KYODAN & THE AKEBOSI ROCKETS - GATE II -Sekai o Koete- (TV-size)/KISIDA KYODAN & THE AKEBOSI ROCKETS - GATE II ~Sekai o Koete~ (TV-size) (-Sh1n1-) [EZ].osu

#  out/songs/531713 AKINO with bless4 - Yuki no Youni/AKINO with bless4 - Yuki no Youni (Maxus) [CS' 7K Normal].osu

#   out/songs/243027 Various - Yolomania Vol 3A/Various - Yolomania Vol. 3A (Fullerene-) [Japanese People - Do you even squeak [BilliumMoto] 15].osu

# [39] out/songs/424743 dj TAKA feat.AiMEE - True Blue/dj TAKA feat.AiMEE - True Blue (ZZHBOY) [ADVANCED].osu