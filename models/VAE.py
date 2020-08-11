import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GRU, LSTM, Conv2D, TimeDistributed, \
    Flatten, Lambda, Dense, concatenate, Embedding, Reshape, RepeatVector, PReLU, MaxPooling1D
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.models import Model

from models import densenet, rhythm, common_model
from models.LayerNormalization import LayerNormalization
from models.PositionEmbedding import PositionEmbedding
from models.losses import CCELoss, StrengthLoss, KLLoss
import numpy as np


class VAE:

    def __init__(self, config, key, is_training=True):
        self.config = config
        self.key = key
        self.encoder_built = False
        self.decoder_built = False
        self.is_training = is_training

    def build_encoder(self):
        self.encoder_built = True
        growth_rate = self.config['dense_growth_rate']
        drop_rate = self.config['dense_drop_rate']
        nb_layers = self.config['dense_nb_layers']
        latent_dim = self.config['vae_latent_dim']
        filter_size = self.config['dense_filter_size']
        intermediate_dim = self.config['vae_intermediate_dim']
        batch_size = self.config['rhythm_batch_size']
        input_length = self.config['vae_input_length']

        self.encoder_input = Input(shape=(input_length, self.key, 2), name="encoder_input")
        # x = Conv2D(growth_rate * 2, (1, self.key), padding='same', name="encoder_entry")(
        #     self.encoder_input)
        # x = LayerNormalization(name="encoder_entry_norm")(x)
        # # x = LeakyReLU(alpha=0.1)(x)
        # x = PReLU(shared_axes=[1, 2])(x)
        x = self.encoder_input
        x = PositionEmbedding(size=self.key, mode='concat')(x)

        # DenseNet to capture map features
        x = densenet.DenseBlock(x, nb_layers, growth_rate, drop_rate,
                                filter_size=(3, 3),
                                is_1d=False, prefix="encoder_1")
        # 2D -> 1D
        x = TimeDistributed(Flatten())(x)
        x = densenet.TransitionLayer(x, prefix="encoder_1_trans", is_1d=True)
        x = densenet.DenseBlock(x, nb_layers, growth_rate, drop_rate,
                                filter_size=filter_size,
                                is_1d=True, prefix="encoder_2")
        x = densenet.TransitionLayer(x, prefix="encoder_2_trans", is_1d=True)
        x = densenet.DenseBlock(x, nb_layers, growth_rate, drop_rate,
                                filter_size=filter_size,
                                is_1d=True, prefix="encoder_3")
        # x = Dense(1, name="encoder_reduct")(encoder_out)
        # reduct_t = K.mean(encoder_out, axis=1, keepdims=False)
        # reduct_c = Dense(1, name="encoder_reduct_c")(x)
        x = K.mean(x, axis=1, keepdims=False)
        # x = Flatten()(Dense(1, name="encoder_reduct_c")(x))
        # x = MaxPooling1D(pool_size=input_length // 4 // latent_dim)(x)
        # self.z_mean = Flatten()(x)
        # assert self.z_mean.shape[-1] == latent_dim

        # LSTM to reduce dim
        # with tf.device('/cpu:0'):
        #     x = LSTM(latent_dim, name="encoder_lstm")(x)
        self.z_mean = Dense(latent_dim, name="encoder_dense_mean")(x)#(Flatten()(reduct_c))
        self.z_log_var = Dense(latent_dim, name="encoder_dense_log")(x)#(Flatten()(reduct_t))

        # Sampling VAE
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim))
            return z_mean + K.exp(z_log_var / 2) * epsilon

        self.context_vector = Lambda(sampling, output_shape=(latent_dim,), name="encoder_sample")(
            [self.z_mean, self.z_log_var])

        if not self.is_training:
            self.encoder_model = Model(self.encoder_input, self.context_vector)

    def build_decoder(self):
        self.decoder_built = True
        latent_dim = self.config['vae_latent_dim']
        intermediate_dim = self.config['vae_intermediate_dim']
        embedding_dim = self.config['embedding_dim']
        drop_rate = self.config['dense_drop_rate']
        growth_rate = self.config['dense_growth_rate']
        filter_size = self.config['dense_filter_size']
        nb_layers = self.config['dense_nb_layers']

        if not self.is_training:
            context = Input(shape=(None, latent_dim))
            input_length = None
            input_length_audio = None
        else:
            assert self.encoder_built
            input_length = self.config['vae_input_length']
            input_length_audio = input_length * 4
            context = self.context_vector

        self.rhythm_model = rhythm.build_rhythm_base(self.config, input_length=input_length_audio)
        self.audio_input = self.rhythm_model.input
        self.audio_base_output = self.rhythm_model.output
        self.context_vector_repeat = common_model.RepeatLayer()([context, self.audio_base_output])
        self.rhythm_base_input = Input(shape=(input_length, 1), name="rhythm_base_input")

        concat1 = concatenate(
            [self.audio_base_output, self.context_vector_repeat, self.rhythm_base_input],
            axis=2)
        out1 = densenet.DenseBlock(concat1, nb_layers // 2, growth_rate, drop_rate,
                                   filter_size, prefix="rhy_3")
        # attention layer
        attention_weight = Dense(latent_dim, name="decoder_attention", activation="softmax")(
            out1)
        context_vector_repeat_attention = attention_weight * self.context_vector_repeat
        concat2 = concatenate([out1, context_vector_repeat_attention], axis=-1)
        self.audio_output = densenet.DenseBlock(concat2, nb_layers // 2, growth_rate, drop_rate,
                                                filter_size, prefix="rhy_4")

        with tf.device('/cpu:0'):
            self.decoder_lstm_1 = LSTM(intermediate_dim, return_sequences=True, return_state=True,
                                       name="decoder_rnn_1")
        self.embedding = Embedding(3 ** self.key, embedding_dim, name="decoder_embedding",
                                   input_length=input_length)
        self.decoder_softmax = TimeDistributed(
            Dense(3 ** self.key, name="decoder_softmax", activation="softmax"),
            name="decoder_softmax_time")

        if self.is_training:
            self.decoder_input = Input(shape=(input_length, 1))
            x = self.embedding(self.decoder_input)
            x = Reshape(target_shape=(input_length, embedding_dim))(x)
            x = concatenate([self.audio_output, x], axis=-1)
            with tf.device('/cpu:0'):
                x, _, _ = self.decoder_lstm_1(x)
            self.decoder_output = self.decoder_softmax(x)  # (B, T, 3 ** key)
            self.y_true = Input(shape=(input_length, 3 ** self.key), name="y_true")

            # calculate strength output
            self.strength_weight = Input(shape=(3 ** self.key),
                                         name="strength_weight_input")
            self.decoder_strength = rhythm.StrengthLayer(self.strength_weight, self.decoder_output,
                                                         self.key, post_fix="pred")
            self.y_true_strength = rhythm.StrengthLayer(self.strength_weight, self.y_true, self.key,
                                                        post_fix="true")
            self.y_true_notes = K.round(self.y_true_strength)
            self.true_positives = K.sum(
                K.round(K.clip(self.y_true_notes * self.decoder_strength, 0, 1)))
        else:
            self.decoder_input = Input(shape=(1, 1))
            self.audio_output_step = Input(shape=(1, self.audio_output.shape[-1]))
            x = self.embedding(self.decoder_input)
            x = Reshape(target_shape=(1, embedding_dim))(x)
            x = concatenate([self.audio_output_step, x], axis=-1)
            decoder_state_input_h = Input(shape=(intermediate_dim,))
            decoder_state_input_c = Input(shape=(intermediate_dim,))
            with tf.device('/cpu:0'):
                x, self.state_h, self.state_c = self.decoder_lstm_1(x, initial_state=[
                    decoder_state_input_h, decoder_state_input_c])
            self.decoder_output = self.decoder_softmax(x)  # (B, T, 3 ** key)

            self.rhythm_model = Model([
                context,
                self.audio_input,
                self.rhythm_base_input
            ], self.audio_output)

            self.decoder_model = Model([
                self.decoder_input,
                self.audio_output_step,
                decoder_state_input_h,
                decoder_state_input_c
            ], [self.decoder_output, self.state_h, self.state_c])

    def loss_strength(self):
        return StrengthLoss()([self.y_true_notes, self.y_true_strength, self.decoder_strength])

    def loss_cce(self):
        return CCELoss(key=self.key, config=self.config)([self.y_true, self.decoder_output])

    def loss_kl(self):
        return KLLoss()([self.z_log_var, self.z_mean])

    def loss_vae(self):
        return (
                       self.loss_cce() +
                       self.loss_kl() * K.mean(self.kl_loss_weight) +
                       self.loss_strength() * self.config['vae_strength_weight']
               ) / 3

    def acc(self):
        return categorical_accuracy(self.y_true, self.decoder_output)

    def top_5_acc(self):
        return top_k_categorical_accuracy(
            K.reshape(self.y_true, (-1, 3 ** self.key)),
            K.reshape(self.decoder_output, (-1, 3 ** self.key)), 5)

    def note_acc(self):
        return self.acc() * K.mean(K.round(K.clip(self.y_true_notes * self.decoder_strength, 0, 1)))

    def overmap_acc(self):
        predicted_positives = K.sum(K.round(K.clip(self.decoder_strength, 0, 1)))
        return self.true_positives / (predicted_positives + K.epsilon())

    def lostnote_acc(self):
        total_positives = K.sum(self.y_true_notes)
        return self.true_positives / (total_positives + K.epsilon())

    def z_log_var_aver(self):
        return K.mean(self.z_log_var)

    def build_train_vae(self):
        assert self.encoder_built
        assert self.decoder_built

        self.kl_loss_weight = Input(shape=(1,), name='kl_loss_weight')
        self.vae_model = Model([
            self.encoder_input,
            self.audio_input,
            self.rhythm_base_input,
            self.kl_loss_weight,
            self.decoder_input,
            self.strength_weight,
            self.y_true
        ], self.decoder_output)
        self.vae_model.add_loss(self.loss_vae())
        self.vae_model.add_metric(self.acc(), aggregation="mean", name="acc")
        self.vae_model.add_metric(self.top_5_acc(), aggregation="mean", name="top5_acc")
        self.vae_model.add_metric(self.note_acc(), aggregation="mean", name="note_acc")
        self.vae_model.add_metric(self.overmap_acc(), aggregation="mean", name="overmap_acc")
        self.vae_model.add_metric(self.lostnote_acc(), aggregation="mean", name="lostnote_acc")
        self.vae_model.add_metric(self.loss_cce(), aggregation="mean", name="cce_loss")
        self.vae_model.add_metric(self.loss_strength(), aggregation="mean", name="strength_loss")
        self.vae_model.add_metric(self.z_log_var_aver(), aggregation="mean", name="var")

    def load_weight(self, path=None):
        if path is not None:
            if self.is_training:
                self.vae_model.load_weights(path, by_name=True, skip_mismatch=True)
            else:
                self.encoder_model.load_weights(path, by_name=True)
                self.rhythm_model.load_weights(path, by_name=True)
                self.decoder_model.load_weights(path, by_name=True)

    def predict(self, base_map, audio, rhythm_base):
        assert not self.is_training
        assert audio.shape[0] == rhythm_base.shape[0] * 4

        context = self.encoder_model.predict(np.expand_dims(base_map, 0))  # (1, 64)
        T = rhythm_base.shape[0]
        print("context: " + str(context))
        audio_output = self.rhythm_model.predict([
            np.expand_dims(context, 0),
            np.expand_dims(audio, 0),
            np.expand_dims(rhythm_base, 0)]
        )  # (1, T, C)
        print("audio_output: " + str(audio_output.shape))
        print("out: " + str(audio_output.flatten()[:100]))
        C = audio_output.shape[-1]

        notes = []
        state_h = np.zeros((1, self.config['vae_intermediate_dim']))
        state_c = np.zeros((1, self.config['vae_intermediate_dim']))
        last_note_index = 0
        from tqdm import tqdm
        for i in range(T):
            output, state_h, state_c = self.decoder_model.predict([
                np.reshape(np.array(last_note_index), (1, 1, 1)),
                np.reshape(audio_output[0, i, :], (1, 1, C)),
                state_h,
                state_c
            ])
            last_note_index = np.argmax(output)
            print(last_note_index, end=", ")
            notes.append(last_note_index)


if __name__ == "__main__":
    import json
    from tensorflow.keras.utils import plot_model
    import os

    os.environ["PATH"] += os.pathsep + 'G:\\D\\graphviz\\bin'

    config = json.load(open("conf\\base_config.json"))
    train = True

    vae = VAE(config, 4, is_training=train)
    vae.build_encoder()
    vae.build_decoder()
    # vae.decoder_model.summary()
    if train:
        vae.build_train_vae()
        vae.vae_model.summary()
        plot_model(vae.vae_model, to_file='out\\model.png', show_shapes=True)
    else:
        vae.encoder_model.summary()
        plot_model(vae.encoder_model, to_file="out\\model_encoder.png", show_shapes=True)
        vae.rhythm_model.summary()
        plot_model(vae.rhythm_model, to_file="out\\model_rhythm.png", show_shapes=True)
        vae.decoder_model.summary()
        plot_model(vae.decoder_model, to_file="out\\model_decoder.png", show_shapes=True)

    # import h5py
    #
    # with h5py.File("out\\models\\vae_000098_0.5298_0.6102_0.7525_0.8810.hdf5", 'r') as f:
    #     if 'layer_names' not in f.attrs and 'model_weights' in f:
    #         f = f['model_weights']
    #     for x in f:
    #         print(x)
