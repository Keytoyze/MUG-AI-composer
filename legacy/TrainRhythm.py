if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn")
    import json
    import sys
    from models import rhythm
    import random
    import os
    from data import RhythmGenerator
    from pathlib import Path
    from tensorflow.keras.optimizers import Nadam
    from tensorflow.keras.callbacks import ModelCheckpoint
    import numpy as np
    from models.scheduler import WarmUpCosineDecayScheduler

    # from models.LRRangeTest import callbacks_list

    config = json.load(open(os.path.join("conf", "base_config.json")))
    train_dir = config['base_dir']

    print("Build model ...")
    model = rhythm.build_rhythm_model(config, weights_path="../out/models/0_rhythm_0040_0.4076_0.3840_0.2047.hdf5")
    model.compile(loss=rhythm.rhythm_loss, optimizer=Nadam(lr=config['dense_lr']), metrics=[rhythm.overmap, rhythm.lostnote])
    print("Success ...")

    osu_paths = [str(p) for p in Path(train_dir).glob("**/*") if p.suffix.lower() == ".osu"]
    random.shuffle(osu_paths)
    val_paths = osu_paths[:5]
    train_paths = osu_paths[5:]

    generator = RhythmGenerator.RhythmTrainGenerator(train_paths, config)
    val_generator = RhythmGenerator.RhythmValGenerator(val_paths, config)

    if os.path.exists("../logs/log.log"):
        os.remove("../logs/log.log")
    
    checkpoint = ModelCheckpoint("out\\models\\lstm_rhythm_{epoch:04d}_{loss:.4f}_{overmap:.4f}_{lostnote:.4f}.hdf5", verbose=0, save_best_only=False, save_weights_only=False)
    callbacks_list = [checkpoint, WarmUpCosineDecayScheduler(
        config["dense_lr"],
        config["rhythm_epoch"] * config["rhythm_step_per_epoch"] // config["rhythm_batch_size"],
        warmup_learning_rate=config['dense_lr'] / 10,
        warmup_steps=20,
        verbose=1
    )]

    val_data_paths = [str(p) for p in Path("val_data\\rhythm").glob("**/*") if p.suffix.lower() == ".npz"]
    val_x = []
    val_y = []
    for val_data_path in val_data_paths:
        data = np.load(val_data_path)
        val_x.append(data['x'])
        val_y.append(data['y'])
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)

    model.fit_generator(generator=generator,
        steps_per_epoch=config['rhythm_step_per_epoch'],
        epochs=config['rhythm_epoch'],
        validation_data=(val_x, val_y),
        verbose=1, 
        callbacks=callbacks_list)