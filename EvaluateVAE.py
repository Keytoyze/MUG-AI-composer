if __name__ == "__main__":

    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    import tensorflow as tf

    if tf.__version__.startswith('1.'):  # tensorflow 1
        config = tf.ConfigProto()  # allow_soft_placement=True
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:  # tensorflow 2
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    import json
    import random

    from models.VAE import VAE
    from utils import MapUtils, OsuUtils
    config = json.load(open(os.path.join("conf", "base_config.json")))

    map_base = "G:\\E\\osu!\\Songs\\575053 Camellia - Exit This Earth's Atomosphere\\Camellia - Exit This Earth's Atomosphere (Protastic101) [11.186 kms].osu"
    map_audio = "G:\\E\\osu!\\Songs\\575053 Camellia - Exit This Earth's Atomosphere\\Camellia - Exit This Earth's Atomosphere (Protastic101) [11.186 kms].osu"
    weight = "out\\models\\vae_000022_0.5546_0.6064_0.7446_0.8736.hdf5"
    map_base_js = OsuUtils.parse_beatmap(map_base, config)
    map_audio_js = OsuUtils.parse_beatmap(map_audio, config)

    vae = VAE(config, 4, is_training=False)
    vae.build_encoder()
    vae.build_decoder()
    vae.load_weight(weight)
    vae.decoder_model.save_weights("test1.hdf5", "h5")

    MapUtils.compose(vae, config, map_base_js, map_audio_js)
    # import h5py
    #
    # with h5py.File("test1.hdf5", 'r') as f:
    #     if 'layer_names' not in f.attrs and 'model_weights' in f:
    #         f = f['model_weights']
    #     for x in f:
    #         print(x)
