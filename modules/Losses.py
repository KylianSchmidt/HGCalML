global_loss_list = {}

import numpy as np
import tensorflow as tf

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

# TODO Custom loss function which 
def loss_track_distance(truth, prediction) :
    """ Loss function for the reconstruction of two photons. Compares the
    accuracy of the reconstructed track with both real tracks and minimizes
    the combined distance.
    \n 
    Truth array has the shape:
        0   1   2   3   4   5   6   7   8   9   10  11  12  13 \n
        px1 py1 pz1 n1  vx1 vy1 vz1 px2 py2 pz2 n2  vx2 vy2 vz2
    """

    p = tf.debugging.check_numerics(prediction, "Prediction has nans or infs")
    
    pred1 = tf.concat([a[...,tf.newaxis] for a in 
                      [p[:,0], p[:,1], p[:,2],       #p1
                       p[:,3],                       #norm1
                       p[:,4], p[:,5], p[:,6],       #v1
                       p[:,7], p[:,8],  p[:,9],      #p2
                       p[:,10],                      #norm2
                       p[:,11], p[:,12], p[:,13]]],  #v2
                       axis=1)
    pred2 = tf.concat([a[...,tf.newaxis] for a in 
                      [p[:,7], p[:,8], p[:,9],       #p2
                       p[:,10],                      #norm2
                       p[:,11], p[:,12], p[:,13],    #v2
                       p[:,0], p[:,1], p[:,2],       #p1
                       p[:,3],                       #norm1
                       p[:,4], p[:,5], p[:,6]]],     #v1
                       axis=1)
    
    # Loss function
    distance1 = tf.reduce_mean((pred1-truth)**2)
    distance2 = tf.reduce_mean((pred2-truth)**2)
    res = tf.minimum(distance1, distance2)

    return res

global_loss_list["loss_track_distance"] = loss_track_distance