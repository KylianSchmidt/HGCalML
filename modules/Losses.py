
'''
We don't use the standard keras loss logic, keep this file empty
'''
global_loss_list = {}


import tensorflow as tf

# Custom loss function which reduces the dimensionality of the feature array
# entries into a single number
def loss_reduceMean(pred,truth):
    print(truth.shape)
    print(pred.shape)

    return tf.reduce_mean(pred**2)

global_loss_list['loss_reduceMean'] = loss_reduceMean