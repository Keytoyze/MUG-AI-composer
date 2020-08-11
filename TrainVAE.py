if __name__ == "__main__":

    import multiprocessing as mp

    mp.set_start_method("spawn")

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
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    from models.VAE import VAE
    import traceback

    from data.VAEGenerator import VAETrainGenerator, VAETestGenerator
    from pathlib import Path
    from tensorflow.keras.callbacks import ModelCheckpoint

    from models.scheduler import WarmUpCosineDecayScheduler
    # from models import LRRangeTest
    from models.RAdam import RAdam
    from utils import OsuUtils
    import time
    from tqdm import tqdm
    import pickle

    config = json.load(open(os.path.join("conf", "base_config.json")))
    train_dir = config['base_dir']
    start_epoch = 0

    print("Build model ...")
    key = 4

    vae = VAE(config, key)
    vae.build_encoder()
    vae.build_decoder()
    vae.build_train_vae()
    vae.load_weight("out\\models\\pre_vae_000046_0.6206_0.8339_0.6925_0.8869.hdf5")

    model = vae.vae_model
    model.compile(optimizer=RAdam(lr=config['vae_lr']))
    print("Success ...")

    t = time.time()

    path_cache = "out\\path_cache"
    if os.path.exists(path_cache):
        with open(path_cache, "rb") as f:
            osu_paths = pickle.load(f)
    else:
        osu_paths_ = [str(p) for p in Path(train_dir).glob("**/*") if p.suffix.lower() == ".osu"]
        osu_paths = []

        for path in tqdm(osu_paths_):
            try:
                js = OsuUtils.parse_beatmap(path, config, meta_only=True)
            except:
                continue
            if os.path.exists(js['bgm_path']) and js['key'] == key \
                    and js['old_star'] > 3 and js['old_star'] < 8 and js['is_mania']:
                osu_paths.append((path, js['old_star']))
        with open(path_cache, "wb") as f:
            pickle.dump(osu_paths, f)

    random.shuffle(osu_paths)
    # val_paths = osu_paths[:5]
    # train_paths = osu_paths[5:]

    print("Find map: %d in %d seconds" % (len(osu_paths), time.time() - t))

    train_paths = [x for x in osu_paths if hash(x[0]) % 40 != 5]
    test_paths = [x for x in osu_paths if hash(x[0]) % 40 == 5]
    generator = VAETrainGenerator(train_paths, config, key)
    test_generater = VAETestGenerator(test_paths, config, key)
    print("Train: %d, Test: %d" % (len(train_paths), len(test_paths)))
    print(test_paths)

    if os.path.exists("logs/log.log"):
        os.remove("logs/log.log")

    checkpoint = ModelCheckpoint("out\\models\\teach_vae_{epoch:06d}_{acc:.4f}_"
                                 "{top5_acc:.4f}_{overmap_acc:.4f}_{lostnote_acc:.4f}.hdf5",
                                 verbose=0, save_best_only=False, save_weights_only=True)
    warm_up_cosine_decay_scheduler = WarmUpCosineDecayScheduler(
        config["vae_lr"],
        config["vae_epoch"] * config["vae_step_per_epoch"],
        warmup_learning_rate=config['dense_lr'] / 10,
        warmup_steps=0,
        verbose=1,
        global_step_init=start_epoch * config['vae_step_per_epoch']
    )
    callbacks_list = [warm_up_cosine_decay_scheduler, checkpoint
                      ]

    # val_data_paths = [str(p) for p in Path("val_data\\rhythm").glob("**/*") if
    #                   p.suffix.lower() == ".npz"]
    # val_x = []
    # val_y = []
    # for val_data_path in val_data_paths:
    #     data = np.load(val_data_path)
    #     val_x.append(data['x'])
    #     val_y.append(data['y'])
    # val_x = np.concatenate(val_x)
    # val_y = np.concatenate(val_y)

    model.fit(x=generator,
              steps_per_epoch=config['vae_step_per_epoch'],
              epochs=config['vae_epoch'],
              validation_data=test_generater,
              # validation_data=(val_x, val_y),
              verbose=2,
              callbacks=callbacks_list,
              initial_epoch=start_epoch)
