'''
Track Reconstruction algorithm for the simulation of the LUXE experiment
Uses the GravNet architecture 
'''
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate
from Layers import VectorNorm
from RaggedLayers import CollapseRagged
from GravNetLayersRagged import CastRowSplits, ScaledGooeyBatchNorm2, RaggedGravNet
import tensorflow as tf
from DeepJetCore.DJCLayers import ScalarMultiply

def pretrain_model(Inputs):
    """ Network model for the BeamDumpTrackCalo two photon reconstruction

    Notes:
    -----
    Specific Model Name: 
    Original model

    v5:
    Use Concatenate(CollapseRagged('sum'), CollapseRagged('mean'))

    Parameters
    ----------
    Inputs : tuple(x ,rs)
        Ragged arrays containing the features of the detector hits in the form 
            [eventNum x hits] x properties 
        where "eventNum x hits" are separated by TensorFlow using the rowsplits rs
    
    Returns
    -------
    Model(Inputs, Outputs)
        Inputs : same as Inputs \n
        Outputs : \n
        [p1_normed, norm1, v1, p2_normed, norm2, v2]
    """
    
    x, rs = Inputs
    rs = CastRowSplits()(rs)
    x = ScaledGooeyBatchNorm2()(x)

    x_list = []
    for _ in range(5):
        x,*_ = RaggedGravNet(
            n_neighbours = 32,
            n_dimensions = 5,
            n_filters = 64,
            n_propagate = 64,
            feature_activation='elu')([x,rs])
        
        x = Dense(64, activation='elu')(x)
        x = ScaledGooeyBatchNorm2()(x)
        x_list.append(x)

    x = Concatenate(axis=1)(x_list)
    x = Dense(128, activation='elu')(x)

    # This bit has a large impact on the shape of the output distributions
    sum = CollapseRagged('sum')([x,rs])
    mean = CollapseRagged('mean')([x,rs])
    x = Concatenate(axis=1)([sum, mean])
    x = ScaledGooeyBatchNorm2()(x)
    
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)

    x = ScaledGooeyBatchNorm2()(x)

    p1 = Dense(3)(x)
    p1_normed, norm1 = VectorNorm()(p1)
    norm1 = ScalarMultiply(1000)(norm1)

    v1 = Dense(3)(x)
    v1 = ScalarMultiply(1000)(v1)

    p2 = Dense(3)(x)
    p2_normed, norm2 = VectorNorm()(p2)
    norm2 = ScalarMultiply(1000)(norm2)

    v2 = Dense(3)(x)
    v2 = ScalarMultiply(1000)(v2)

    Outputs = Concatenate(axis=1)(
    #    0, 1, 2,   3,     4, 5, 6,  7, 8, 9,   10,    11, 12, 13
        [p1_normed, norm1, v1,       p2_normed, norm2, v2        ])
    return Model(inputs=Inputs, outputs=Outputs)


train=training_base()

from Losses import loss_track_distance

if not train.modelSet():
    train.setModel(pretrain_model)
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
    
    train.compileModel(learningrate=1e-2, 
                       loss=loss_track_distance)
    
    train.keras_model.summary()
    
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
cb = [simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency= 10,
        plot_frequency = 5,
        select_metrics='*loss'
        ),]

nbatch = 500 
train.change_learning_rate(1e-2)
train.trainModel(nepochs=5, batchsize=nbatch, additional_callbacks=cb)

nbatch = 5000 
train.change_learning_rate(1e-3)
train.trainModel(nepochs=10, batchsize=nbatch, additional_callbacks=cb)

nbatch = 5000
train.change_learning_rate(1e-5)
train.trainModel(nepochs=50, batchsize=nbatch, additional_callbacks=cb)
