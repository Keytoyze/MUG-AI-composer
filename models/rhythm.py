from models import densenet
from tensorflow.keras.layers import LeakyReLU, Conv1D, Input, Activation, Reshape, PReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mean_squared_error
from models.LayerNormalization import LayerNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.layers import multiply
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from models.common_model import RepeatLayer
from models.PositionEmbedding import PositionEmbedding


def get_input_channel(config):
    return config['wav_feature']


def StrengthLayer(strength_weight, output, key, post_fix=""):
    x = RepeatLayer(output_shape=(None, 3 ** key), name="repeat_weight_" + post_fix)(
        [strength_weight, output]
    )
    x = multiply([x, output])
    x = K.sum(x, axis=-1, keepdims=False)  # (B, T)
    return x


def build_rhythm_base(config, prefix="rhy", input_length=None):
    growth_rate = config['dense_growth_rate']
    drop_rate = config['dense_drop_rate']
    filter_size = config['dense_filter_size']
    nb_layers = config['dense_nb_layers']

    inpt = Input(shape=(input_length, get_input_channel(config)), name="audio_input")
    x = PositionEmbedding(size=4, mode='concat')(inpt)

    x = Conv1D(growth_rate * 2, 3, padding='same', name=prefix + "_entry")(inpt)
    x = LayerNormalization(name=prefix + "_entry_norm")(x)
    # x = LeakyReLU(alpha=0.1)(x)
    x = PReLU(shared_axes=[1], name=prefix + "_entry_prelu")(x)

    x = densenet.DenseBlock(x, nb_layers, growth_rate, drop_rate, filter_size=filter_size,
                            prefix=prefix + "1")
    x = densenet.TransitionLayer(x, prefix=prefix + "1")
    x = densenet.DenseBlock(x, nb_layers, growth_rate, drop_rate, filter_size=filter_size,
                            prefix=prefix + "2")
    x = densenet.TransitionLayer(x, prefix=prefix + "2")
    x = LayerNormalization(name="audio_output")(x)

    # return Model(inpt, x)
    return Model(inpt, x)


# check if a note generated is overmapping
def overmap(y_true, y_pred):
    y_true_notes = (y_true[:, :, 0] + 1) / 2
    y_pred_notes = (y_pred[:, :, 0] + 1) / 2 + 0.2
    y_true_1 = K.round(y_true_notes)
    true_positives = K.sum(K.round(K.clip(y_true_1 * y_pred_notes, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_notes, 0, 1)))
    return (predicted_positives - true_positives) / (predicted_positives + K.epsilon())


# check if a note given is lost
def lostnote(y_true, y_pred):
    y_true_notes = (y_true[:, :, 0] + 1) / 2
    y_pred_notes = (y_pred[:, :, 0] + 1) / 2 + 0.2
    y_true_1 = K.round(y_true_notes)
    true_positives = K.sum(K.round(K.clip(y_true_1 * y_pred_notes, 0, 1)))
    total_positives = K.sum(y_true_1)
    return (total_positives - true_positives) / (total_positives + K.epsilon())


# assign different weights
def rhythm_loss(y_true, y_pred):
    y_true_notes = K.round((y_true[:, :, 0] + 1) / 2)
    blank_weight = K.mean(y_true_notes) * 2
    note_weight = 2 - blank_weight
    sample_weight = y_true_notes * (note_weight - blank_weight) + blank_weight

    ln_loss = mean_squared_error(y_true[:, :, 1], y_pred[:, :, 1])
    note_loss = K.mean(K.square(y_true[:, :, 0] - y_pred[:, :, 0]) * sample_weight, axis=-1)

    return ln_loss * 0.2 + note_loss * 0.8


# class UpdateAnnealingParameter(Callback):
#     def __init__(self, gamma, nb_epochs, verbose=0):
#         super(UpdateAnnealingParameter, self).__init__()
#         self.gamma = gamma
#         self.nb_epochs = nb_epochs
#         self.verbose = verbose

#     def on_epoch_begin(self, epoch, logs=None):
#         new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
#         K.set_value(self.gamma, new_gamma)

#         if self.verbose > 0:
#             print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))

def get_scheduler(model):
    def scheduler(epoch):
        if epoch < 1000:
            # lr=K.get_value(model2.optimizer.lr)
            K.set_value(model.optimizer.lr, 0.00001)
            return K.get_value(model.optimizer.lr)
        else:
            K.set_value(model.optimizer.lr, 0.0000001)
            return K.get_value(model.optimizer.lr)

    return scheduler


def build_rhythm_model(config, weights_path=None):
    base_model = build_rhythm_base(config)
    input_length = config['rhythm_input_length']
    model = Sequential([
        base_model,

        # Reshape((input_length, 1, 258)),
        # Conv2D(2, (1, 1)),
        # Reshape((input_length, 2)),

        CuDNNLSTM(2, return_sequences=True),
        Activation('tanh')
    ])

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model
