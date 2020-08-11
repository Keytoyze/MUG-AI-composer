# coding=utf8
"""Recifited Adam optimizer
# Author : forin-xyz
# Created Time : Aug 24 22:02:55 2019
# Description:
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.legacy import interfaces


class RAdam(Adam):
    """RAdam optimizer, also named Recifited Adam optimizer.
    Arguments
    ---------
        lr: float >= 0. Learning rate, default 0.001.
        beta_1: float, (0, 1). Generally close to 1.
        beta_2: float, (0, 1). Generally close to 1.
        epsilon: float >= 0. Fuzz factor, a negligible value (
            e.g. 1e-8), defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    References
    ----------
      - [On the Variance of the Adaptive Learing Rate and Beyond](
         https://arxiv.org/abs/1908.03265)
    """

    # @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay:
            lr = lr * (1. / (1. + self.decay * K.cast(
                self.iterations, K.dtype(self.decay)
            )))

        t = K.cast(self.iterations, K.floatx()) + 1.
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        beta_1_t = K.pow(beta_1, t)
        beta_2_t = K.pow(beta_2, t)
        rho_inf = 2. / (1. - beta_2) - 1.
        rho_t = rho_inf - 2. * t * beta_2_t / (1. - beta_2_t)
        r_t = K.sqrt(
            K.relu(rho_t - 4.) * (rho_t - 2.) * rho_inf / (
                    K.relu(rho_inf - 4.) * (rho_inf - 2.) * rho_t)
        )
        flag = K.cast(rho_t > 4., K.floatx())

        ms = [K.zeros(K.int_shape(p)) for p in params]
        vs = [K.zeros(K.int_shape(p)) for p in params]

        self.weights = [self.iterations] + ms + vs
        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = beta_1 * m + (1. - beta_1) * g
            v_t = beta_2 * v + (1. - beta_2) * K.square(g)

            m_hat_t = m_t / (1. - beta_1_t)
            v_hat_t = K.sqrt(v_t / (1. - beta_2_t))
            new_p = p - lr * (r_t / (v_hat_t + self.epsilon) + flag - 1.) * m_hat_t

            if getattr(p, "constraint", None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
        return self.updates


del division
del print_function
del absolute_import
del unicode_literals
