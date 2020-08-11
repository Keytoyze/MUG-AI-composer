from tensorflow.keras.layers import LeakyReLU, Conv1D, Conv2D, Dropout, concatenate, MaxPooling1D, \
    MaxPooling2D, Input, PReLU
from models.LayerNormalization import LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def DenseLayer(x, nb_filter, alpha=0.0, drop_rate=0.2, filter_size=3, is_1d=True,
               prefix=""):
    # Bottleneck layers
    x = LayerNormalization(name=prefix + "_l_bn_norm")(x)
    # x = LeakyReLU(alpha=alpha)(x)
    if is_1d:
        x = PReLU(shared_axes=[1], name=prefix + "_l_bn_prelu")(x)
        x = Conv1D(4 * nb_filter, 1, padding='same', name=prefix + "_l_bn_1d")(x)
    else:
        x = PReLU(shared_axes=[1,2], name=prefix + "_l_bn_prelu")(x)
        x = Conv2D(4 * nb_filter, (1, 1), padding='same', name=prefix + "_l_bn_2d")(x)

    # Composite function
    x = LayerNormalization(name=prefix + "_l_cs_norm")(x)
    # x = LeakyReLU(alpha=alpha)(x)
    if is_1d:
        x = PReLU(shared_axes=[1], name=prefix + "_l_cs_prelu")(x)
        x = Conv1D(nb_filter, filter_size, padding='same', name=prefix + "_l_cs_1d")(x)
    else:
        x = PReLU(shared_axes=[1,2], name=prefix + "_l_cs_prelu")(x)
        x = Conv2D(nb_filter, filter_size, padding='same', name=prefix + "_l_cs_2d")(x)

    if drop_rate: x = Dropout(drop_rate)(x)

    return x


def DenseBlock(raw_x, nb_layers, growth_rate, drop_rate=0.2, alpha=0.0, filter_size=3, is_1d=True,
               prefix="densenet"):
    shape = K.int_shape(raw_x)[1:]
    inpt = Input(shape=shape)
    x = inpt
    for i in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, alpha=alpha, drop_rate=drop_rate,
                          filter_size=filter_size, is_1d=is_1d, prefix="%s_%d" % (prefix, i))
        axis = 2 if is_1d else 3
        x = concatenate([x, conv], axis=axis)
    return Model(inpt, x, name=prefix + "_denseblock")(raw_x)


def TransitionLayer(raw_x, compression=0.5, alpha=0.0, is_1d=True, pool_size=2, prefix=""):
    nb_filter = int(raw_x.shape.as_list()[-1] * compression)
    shape = K.int_shape(raw_x)[1:]
    inpt = Input(shape=shape)
    x = inpt
    x = LayerNormalization(name=prefix + "_t_norm")(x)
    # x = LeakyReLU(alpha=alpha)(x)
    if is_1d:
        x = PReLU(shared_axes=[1], name=prefix + "_t_prelu")(x)
        x = Conv1D(nb_filter, 1, padding='same', name=prefix + "_t_1d")(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
    else:
        x = PReLU(shared_axes=[1,2], name=prefix + "_t_prelu")(x)
        x = Conv2D(nb_filter, (1, 1), padding='same', name=prefix + "_t_1d")(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
    return Model(inpt, x, name=prefix + "_dense_transition")(raw_x)
