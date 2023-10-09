from typing import Any
import tensorflow as tf
import keras.losses

global_loss_list = {}


def loss_reduceMean(truth, pred):
    """ Custom loss function which reduces the dimensionality of the feature array
    entries into a single number """

    print("PREDICTION START:")
    tf.print(pred)
    print(pred.shape)
    print("PREDICTION END")
    pred = tf.debugging.check_numerics(pred, "pred has nans or infs")

    out = tf.reduce_mean(pred**2)
    return out


global_loss_list['loss_reduceMean'] = loss_reduceMean


class nntr_L2_distance(keras.losses.Loss):
    """ Loss function for the reconstruction of two photons. Compares the
    accuracy of the reconstructed track with both real tracks and minimizes
    the combined distance.
    Version without norms, so len(truth) == 12
    """
    def __init__(self,
                 train_uncertainties=True,
                 epsilon=1E-3,
                 **kwargs) -> None:

        super().__init__(reduction=tf.keras.losses.Reduction.NONE,
                         name="nntr_L2_distance")
        self.train_uncertainties = train_uncertainties
        self.eps = epsilon

    def call(self,
             truth,
             prediction) -> Any:
        """ Call method can only accept two *args (truth, prediction)
        """
        self.pred1, self.pred2, self.sigma = self.prepare_inputs(prediction)
        self.truth = truth

        return self.with_uncertainties()

    def prepare_inputs(self, prediction):
        """ Expect always two vertices and permutate both to check which best
        fits the true vertex variables

        Shape of prediction (and truth) array: \n
        0   1   2    3   4   5    6   7   8    9   10  11  \n
        px1 py1 pz1  vx1 vy1 vz1  px2 py2 pz2  vx2 vy2 vz2 \n

        Uncertainties are given as ln(sigma) with indices in 'prediction': \n
        12-23
        """

        p = tf.debugging.check_numerics(prediction,
                                        "Prediction has nans or infs")
        pred1 = tf.concat([a for a in
                          [p[:, 0:3],       # p1
                           p[:, 3:6],       # v1
                           p[:, 6:9],       # p2
                           p[:, 9:12]]],    # v2
                          axis=1)
        pred2 = tf.concat([a for a in
                          [p[:, 6:9],       # p2
                           p[:, 9:12],      # v2
                           p[:, 0:3],       # p1
                           p[:, 3:6]]],     # v1
                          axis=1)
        sigma = p[:, 12:24]

        return pred1, pred2, sigma

    def with_uncertainties(self):
        r""" Allows the network to estimate the uncertainty on the L2 distance
        by training the parameter 'sigma'

        Latex formula:
        \mathcal{L} = \left( \frac{\Vec{t} - \Vec{p}}{\Vec{\sigma}} \right)^2 + \ln{(\Vec{\sigma}^2)}
        """

        # Turn off uncertainties in the loss by omitting it from d1 and d2
        # and training it to zero in 'loss_per_event'
        if self.train_uncertainties:
            sigma_zero = 0.0
            sigma = self.sigma
        else:
            sigma_zero = self.sigma
            sigma = 0.0

        # Loss function
        distance1 = tf.reduce_mean(((self.pred1 - self.truth)/sigma)**2 + tf.math.log(sigma)**2,
                                   axis=1,
                                   keepdims=True)
        distance2 = tf.reduce_mean(((self.pred2 - self.truth)/sigma)**2 + tf.math.log(sigma)**2,
                                   axis=1,
                                   keepdims=True)
        # Loss = E x min([d1, d2]) = E x 1
        loss_per_event = tf.reduce_min(tf.concat([distance1, distance2],
                                                 axis=1),
                                       axis=1) + tf.reduce_mean(sigma_zero**2)
        # res = min(E x 1) = 1
        res = tf.reduce_mean(loss_per_event)

        return res


global_loss_list["nntr_L2_distance"] = nntr_L2_distance
