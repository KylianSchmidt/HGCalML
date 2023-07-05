
'''
We don't use the standard keras loss logic, keep this file empty
'''
global_loss_list = {}

import numpy as np
import tensorflow as tf

# Custom loss function which reduces the dimensionality of the feature array
# entries into a single number
def loss_reduceMean(truth, pred):
    #print(truth.shape)
    #print(pred.shape)

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

    # Write loss function here
    # What is the shape of predicted and truth?
    # keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
    t = truth#[:,:-3][0]
    print("t ", t)
    tf.print("t ", t)
    p = prediction
    distance = (t-p)**2
    return tf.reduce_mean(distance)