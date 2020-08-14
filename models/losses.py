from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class CCELoss(Layer):
    def __init__(self, key, config, **kwargs):
        super(CCELoss, self).__init__(**kwargs)
        self.key = key
        self.input_length = config['vae_input_length']

    def label_smoothing(self, inputs, epsilon=0.1):
        c = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / c)

    def call(self, inputs, **kwargs):
        y_true, decoder_output = inputs
        note_distribution = K.mean(y_true, axis=1, keepdims=False) + K.epsilon()  # (B, 3 ** K)
        class_weight = K.clip(1 / note_distribution / (3 ** self.key), 0, 50)  # (B, 3 ** K)
        class_weight = K.repeat(class_weight, self.input_length)  # （B, T, 3 ** K)
        sample_weight = K.sum(y_true * class_weight, axis=-1)  # （B, T)
        cce_loss = K.mean(
            K.categorical_crossentropy(self.label_smoothing(y_true), decoder_output)
            * sample_weight)
        return cce_loss


class StrengthLoss(Layer):

    def call(self, inputs, **kwargs):
        y_true_notes, y_true_strength, predict_strength = inputs
        density = K.mean(y_true_notes)
        blank_weight = 2 * density# / (2 - density)
        note_weight = 2 - blank_weight
        sample_weight = y_true_notes * (note_weight - blank_weight) + blank_weight
        strength_loss = K.mean(K.square(y_true_strength - predict_strength)
                               * sample_weight)
        return strength_loss


class KLLoss(Layer):

    def call(self, inputs, **kwargs):
        z_log_var, z_mean = inputs
        return K.mean(- 0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=-1))