import tensorflow as tf
import sys
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


def _nntr_find_prediction(p):
    """
    Shape of prediction (and truth) array: \n
        0   1   2    3   4   5    6   7   8    9   10  11   12  13  14   15  16  17  \n
        A1x A1y A1z  B1x B1y B1z  A2x A2y A2z  B2x B2y B2z  V1x V1y V1z  V1x V1y V1z \n
    """
    pred1 = tf.concat(
        [a for a in
            [p[:, 0:3],       # A1
             p[:, 3:6],       # B1
             p[:, 6:9],       # A2
             p[:, 9:12],      # B2
             p[:, 12:15],     # V1
             p[:, 15:18]]],   # V2
        axis=1)
    pred2 = tf.concat(
        [a for a in
            [p[:, 6:9],      # A2
             p[:, 9:12],     # B2
             p[:, 0:3],      # A1
             p[:, 3:6],      # B1
             p[:, 15:18],    # V2
             p[:, 12:15]]],  # V1
        axis=1)

    tf.debugging.check_numerics(pred1, "Prediction 1 has nans or infs")
    tf.debugging.check_numerics(pred2, "Prediction 2 has nans or infs")
    return pred1, pred2


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
        
        Loss function without the uncertainty parameter 'sigma'

        Latex formula:
        \mathcal{L} = \left( \Vec{t} - \Vec{p} \right)^2
        """
        pred1, pred2 = _nntr_find_prediction(prediction)
        # Loss function
        distance1 = tf.reduce_sum(
            (pred1 - truth)**2,
            axis=1,
            keepdims=True)
        distance2 = tf.reduce_sum(
            (pred2 - truth)**2,
            axis=1,
            keepdims=True)
        # Loss = E x min([d1, d2]) = E x 1
        loss_per_event = tf.reduce_min(
            tf.concat([distance1, distance2], axis=1),
            axis=1)
        # res = min(E x 1) = 1
        return tf.reduce_sum(loss_per_event)

global_loss_list["L2Distance"] = L2Distance

class L1Distance(tf.keras.losses.Loss):
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
        
        Loss function without the uncertainty parameter 'sigma'

        Latex formula:
        \mathcal{L} = \left( \Vec{t} - \Vec{p} \right)^2
        """
        pred1, pred2 = _nntr_find_prediction(prediction)
        # Loss function
        distance1 = tf.math.abs(pred1 - truth)
        distance2 = tf.math.abs(pred2 - truth)
        # Loss = E x min([d1, d2]) = E x 1
        loss_per_event = tf.reduce_min(
            tf.concat([distance1, distance2], axis=1),
            axis=1)
        # res = min(E x 1) = 1
        return tf.reduce_mean(loss_per_event)

global_loss_list["L1Distance"] = L1Distance


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

        Uncertainties are given as ln(sigma) with indices in 'prediction': \n
        18-36      
        
        Allows the network to estimate the uncertainty on the L2 distance
        by training the parameter 'sigma'

        Latex formula:
        $\mathcal{L} = \left( \frac{\Vec{t} - \Vec{p}}{\Vec{\sigma}} \right)^2 + \ln{(\Vec{\sigma}^2)}$

        For practical reasons, exp(sigma) is used to avoid divergences.
        """
        pred1, pred2 = _nntr_find_prediction(prediction)
        assert prediction.shape[1] == 36

        mode = "sigma"

        if mode == "ln_sigma":
            # train ln(sigma) to improve convergence
            ln_sigma = prediction[:, 18:36]
            tf.print("\nln(sigma)", tf.reduce_mean(ln_sigma, axis=0), output_stream=sys.stdout)
            tf.print("\npred-truth", tf.reduce_mean(pred1-truth, axis=0))
            tf.debugging.check_numerics(ln_sigma, "ln(sigma) has nans or infs")

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
        
        if mode == "sigma":
            # train sigma as is
            sigma = prediction[:, 18:36]
            tf.print("\nsigma", tf.reduce_mean(sigma, axis=0), output_stream=sys.stdout)
            tf.print("\npred-truth", tf.reduce_mean(pred1-truth, axis=0))
            tf.debugging.check_numerics(sigma, "Sigma has nans or infs")

            # Loss function
            distance1 = tf.reduce_mean(
                ((pred1 - truth)/sigma)**2 + 2*tf.math.log(sigma),
                axis=1,
                keepdims=True)
            distance2 = tf.reduce_mean(
                ((pred2 - truth)/sigma)**2 + 2*tf.math.log(sigma),
                axis=1,
                keepdims=True)
            # Loss = E x min([d1, d2]) = E x 1
            loss_per_event = tf.reduce_min(
                tf.concat([distance1, distance2], axis=1),
                axis=1)
            # res = min(E x 1) = 1
            return tf.reduce_mean(loss_per_event)

        if mode == "inverse_sigma":
            inverse_sigma = prediction[:, 18:36]
            tf.print("\nsigma", 1/tf.reduce_mean(inverse_sigma, axis=0), output_stream=sys.stdout)
            tf.print("\npred-truth", tf.reduce_mean(pred1-truth, axis=0))
            tf.debugging.check_numerics(inverse_sigma, "Sigma has nans or infs")

            # Loss function
            distance1 = tf.reduce_mean(
                ((pred1 - truth)*inverse_sigma)**2 + inverse_sigma**(-2),
                axis=1,
                keepdims=True)
            distance2 = tf.reduce_mean(
                ((pred2 - truth)*inverse_sigma)**2 + inverse_sigma**(-2),
                axis=1,
                keepdims=True)
            # Loss = E x min([d1, d2]) = E x 1
            loss_per_event = tf.reduce_min(
                tf.concat([distance1, distance2], axis=1),
                axis=1)
            # res = min(E x 1) = 1
            return tf.reduce_mean(loss_per_event)



global_loss_list["L2DistanceWithUncertainties"] = L2DistanceWithUncertainties


class QuantileLoss(tf.keras.losses.Loss):
    """ Loss function for the reconstruction of two photons. Compares the
    accuracy of the reconstructed track with both real tracks and minimizes
    the combined distance. With added uncertainties on each of the predicted
    parameters.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name="QuantileLoss")

    def call(self, truth, prediction):
        r""" Expect always two vertices and permutate both to check which best
        fits the true vertex variables

        Uncertainties are given as ln(sigma) with indices in 'prediction': \n
        18-36      
        
        Allows the network to estimate the uncertainty on the L2 distance
        by training the parameter 'sigma'

        Latex formula:
        $\mathcal{L} = \left( \frac{\Vec{t} - \Vec{p}}{\Vec{\sigma}} \right)^2 + \ln{(\Vec{\sigma}^2)}$

        For practical reasons, exp(sigma) is used to avoid divergences.
        """
        pred1, pred2 = _nntr_find_prediction(prediction)

        assert prediction.shape[1] == 54
        quantile_lower = prediction[:, 18:36]
        quantile_upper = prediction[:, 36:54]
        tf.print("\nquantile_lower", tf.reduce_min(quantile_lower, axis=1), output_stream=sys.stdout)
        tf.print("\nquantile_upper", tf.reduce_min(quantile_upper, axis=1), output_stream=sys.stdout)
        tf.print("\npred-truth", tf.reduce_mean(pred1-truth, axis=1))
        tf.debugging.check_numerics(quantile_lower, "Lower quantile has nans or infs")
        tf.debugging.check_numerics(quantile_upper, "Higher quantile has nans or infs")

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
        loss_median = tf.reduce_mean(loss_per_event)

        # Quantile losses
        def _compute_quantile(quantile_value:float):
            residual1 = tf.reduce_mean(tf.abs(pred1 - truth), axis=1, keepdims=True)
            residual2 = tf.reduce_mean(tf.abs(pred2 - truth), axis=1, keepdims=True)
            residual = tf.reduce_min(tf.concat([residual1, residual2], axis=1), axis=1)
            residual = tf.reduce_mean(residual)
            loss_quantile = tf.maximum(quantile_value*residual, (quantile_value-1)*residual)
            return loss_quantile

        loss_quantile_lower = _compute_quantile(0.25)
        loss_quantile_upper = _compute_quantile(0.75)
        print("\Lower and upper Quantile losses", loss_quantile_lower, loss_quantile_upper)
        return loss_median + loss_quantile_lower + loss_quantile_upper


global_loss_list["QuantileLoss"] = QuantileLoss


