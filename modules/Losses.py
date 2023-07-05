
'''
We don't use the standard keras loss logic, keep this file empty
'''
global_loss_list = {}

import numpy as np
import tensorflow as tf

# Custom loss function which reduces the dimensionality of the feature array
# entries into a single number
def loss_reduceMean(truth, pred):

    print("PREDICTION START:")
    tf.print(pred)
    print(pred.shape)
    print("PREDICTION END")
    pred = tf.debugging.check_numerics(pred, "pred has nans or infs")

    out = tf.reduce_mean(pred**2)
    #tf.print(out)
    return out

global_loss_list['loss_reduceMean'] = loss_reduceMean

# TODO Custom loss function which compares the accuracy of the reconstructed
# track with both real tracks and minimizes the combined distance
def loss_track_distance(truth, prediction) :
    print(truth.shape)
    print(prediction.shape)

    prediction = tf.debugging.check_numerics(prediction, "Prediction has nans or infs")

    t = truth
    print("Truth array in loss function:", t)
    tf.print("Truth array in loss function (TF):", t)
    p = prediction
    distance = (t-p)**2
    return tf.reduce_mean(distance)