import tensorflow as tf
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


class L2Distance(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        """ Loss function for the reconstruction of two photons. Compares the
        accuracy of the reconstructed track with both real tracks and minimizes
        the combined distance.
        """
        super().__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name="L2Distance")
        
    def call(self, truth, prediction):
        r""" Expect always two vertices and permutate both to check which best
        fits the true vertex variables

        Shape of prediction (and truth) array: \n
        0   1   2    3   4   5    6   7   8    9   10  11  \n
        px1 py1 pz1  vx1 vy1 vz1  px2 py2 pz2  vx2 vy2 vz2 \n
        
        
        Loss function without the uncertainty parameter 'sigma'

        Latex formula:
        \mathcal{L} = \left( \Vec{t} - \Vec{p} \right)^2
        """
        p = prediction
        pred1 = tf.concat(
            [a for a in
             [p[:, 0:3],       # p1
              p[:, 3:6],       # v1
              p[:, 6:9],       # p2
              p[:, 9:12]]],    # v2
            axis=1)
        pred2 = tf.concat(
            [a for a in
             [p[:, 6:9],       # p2
              p[:, 9:12],      # v2
              p[:, 0:3],       # p1
              p[:, 3:6]]],     # v1
            axis=1)

        tf.debugging.check_numerics(pred1, "Prediction 1 has nans or infs")
        tf.debugging.check_numerics(pred2, "Prediction 2 has nans or infs")

        # Loss function
        distance1 = tf.reduce_mean(
            (pred1 - truth)**2,
            axis=1,
            keepdims=True)
        distance2 = tf.reduce_mean(
            (pred2 - truth)**2,
            axis=1,
            keepdims=True)
        # Loss = E x min([d1, d2]) = E x 1
        loss_per_event = tf.reduce_min(
            tf.concat([distance1, distance2], axis=1),
            axis=1)
        # res = min(E x 1) = 1
        return tf.reduce_mean(loss_per_event)

global_loss_list["L2Distance"] = L2Distance


class L2DistanceWithUncertainties(tf.keras.losses.Loss):
    """ Loss function for the reconstruction of two photons. Compares the
    accuracy of the reconstructed track with both real tracks and minimizes
    the combined distance. With added uncertainties on each of the predicted
    parameters.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name="L2DistanceWithUncertainties")

    def call(self, truth, prediction):
        r""" Expect always two vertices and permutate both to check which best
        fits the true vertex variables

        Shape of prediction (and truth) array: \n
        0   1   2    3   4   5    6   7   8    9   10  11  \n
        px1 py1 pz1  vx1 vy1 vz1  px2 py2 pz2  vx2 vy2 vz2 \n

        Uncertainties are given as ln(sigma) with indices in 'prediction': \n
        12-23
        
        
        Allows the network to estimate the uncertainty on the L2 distance
        by training the parameter 'sigma'

        Latex formula:
        $\mathcal{L} = \left( \frac{\Vec{t} - \Vec{p}}{\Vec{\sigma}} \right)^2 + \ln{(\Vec{\sigma}^2)}$

        For practical reasons, exp(sigma) is used to avoid divergences.
        """
        p = prediction
        pred1 = tf.concat(
            [a for a in
             [p[:, 0:3],       # p1
              p[:, 3:6],       # v1
              p[:, 6:9],       # p2
              p[:, 9:12]]],    # v2
            axis=1)
        pred2 = tf.concat(
            [a for a in
             [p[:, 6:9],       # p2
              p[:, 9:12],      # v2
              p[:, 0:3],       # p1
              p[:, 3:6]]],     # v1
            axis=1)

        ln_sigma = p[:, 12:24]
        tf.debugging.check_numerics(pred1, "Prediction 1 has nans or infs")
        tf.debugging.check_numerics(pred2, "Prediction 2 has nans or infs")
        tf.debugging.check_numerics(ln_sigma, "Sigma has nans or infs")

        # Loss function
        distance1 = tf.reduce_mean(
            ((pred1 - truth)/(tf.exp(ln_sigma)))**2 + 2*ln_sigma,
            axis=1,
            keepdims=True)
        distance2 = tf.reduce_mean(
            ((pred2 - truth)/(tf.exp(ln_sigma)))**2 + 2*ln_sigma,
            axis=1,
            keepdims=True)
        # Loss = E x min([d1, d2]) = E x 1
        loss_per_event = tf.reduce_min(
            tf.concat([distance1, distance2], axis=1),
            axis=1)
        # res = min(E x 1) = 1
        return tf.reduce_mean(loss_per_event)


global_loss_list["L2DistanceWithUncertainties"] = L2DistanceWithUncertainties
