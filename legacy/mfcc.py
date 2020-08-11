if __name__ == "__main__":
    import librosa
    from utils import OsuUtils, MapUtils
    from data import audio
    import numpy as np
    import json
    import os
    import time
    from tensorflow.keras.layers import Input, MaxPool1D
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import plot_model
    import pydotplus
    from models.VAE import VAE
    from models import rhythm

    config = json.load(open("../conf/base_config.json"))

    #
    name = "G:\\E\osu!\\Songs\\324992 ARM feat Nanahira - BakunanaTestroyer\\ARM feat. Nanahira - BakunanaTestroyer (Rinzler) [BEAST].osu"
    file_name = os.path.basename(name)

    js = OsuUtils.parse_beatmap(name, config)
    #
    # print("osu: " + str(time.time() - t))
    json.dump(js, open("out\\%s.json" % file_name, "w"))
    note_data = OsuUtils.beatmap_to_numpy(config, js)
    np.savetxt("out\\raw_out_%s.txt" % file_name, note_data[:, :, 0], "%d")
    t = time.time()
    aug = MapUtils.augment_map(note_data, True)
    print("time: " + str(time.time() - t))
    np.savetxt("out\\augment_%s.txt" % file_name, aug[:, :, 0], "%d")
    # input = audio.from_osu_json(config, js, False)
    # np.savetxt("out\\input_%s.txt" % file_name, input, "%.2lf")

    # inpt = Input((None, 128))
    # x = MaxPool1D(2)(inpt)
    # model = Model(inpt, x)
    # pool = model.predict(np.expand_dims(input, 0))
    # np.savetxt("out\\pool_%s.txt" % file_name, pool[0, :, :], "%.2lf")
