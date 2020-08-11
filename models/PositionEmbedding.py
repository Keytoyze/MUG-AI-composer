from __future__ import print_function
from keras import backend as K
from tensorflow.python.keras.layers import Layer


class PositionEmbedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        # batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,
                                2 * K.arange(self.size / 2, dtype='float32') / self.size)  # (S/2, )
        position_j = K.expand_dims(position_j, 0) # (1, S/2)
        if len(x.shape) == 3:
            ones = K.ones_like(x[:, :, 0])
        else:
            ones = K.ones_like(x[:, :, 0, 0])
        position_i = K.cumsum(ones, 1) - 1  # (B, T): (0, 1, 2, ...)
        position_i = K.expand_dims(position_i, 2)   # (B, T, 1)
        position_ij = K.dot(position_i, position_j) # (B, T, S/2)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)    # (B, T, S)
        if len(x.shape) == 4:
            position_ij = K.expand_dims(position_ij, -1) # (B, T, S, 1)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], -1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
