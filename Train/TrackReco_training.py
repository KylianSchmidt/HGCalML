'''
Track Reconstruction algorithm for the simulation of the LUXE experiment
Uses the GravNet architecture
'''

import tensorflow as tf
from DeepJetCore.training.training_base import training_base
from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import Dense, Concatenate
from Layers import RaggedGlobalExchange
from RaggedLayers import CollapseRagged
from GravNetLayersRagged import CastRowSplits, ScaledGooeyBatchNorm2, RaggedGravNet
from Losses import nntr_L2_distance
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback


def nntr_two_vertex_fitter(Inputs):
    """ Network model for the BeamDumpTrackCalo two photon reconstruction

    Notes
    -----
    Specific Model Name: 
    GravNet from the original paper https://arxiv.org/pdf/1902.07987.pdf, with
    some modifications

    Version
    --------
    1.0.2

    Date
    ----
    2023-09-28

    Parameters
    ----------
    Inputs : tuple(x ,rs)
        Ragged arrays containing the features of the detector hits in the form 
            [eventNum x hits] x properties 
        where "eventNum x hits" are separated by TensorFlow using the rowsplits
        rs.
    
    Returns
    -------
    Model(Inputs, Outputs)
        Inputs : same as Inputs \n
        Outputs : \n
        [p1, v1, p2, v2, sigma]
    """

    batchnorm_parameters = {
        "fluidity_decay": 0.1,
        "max_viscosity": 0.9999}
    
    x, rs = Inputs
    rs = CastRowSplits()(rs)

    x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)

    x_list = []
    for _ in range(5):
        x = RaggedGlobalExchange()([x, rs])

        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)

        x, *_ = RaggedGravNet(
            n_neighbours=40,
            n_dimensions=4,
            n_filters=64,
            n_propagate=64,
            feature_activation='elu')([x, rs])
   
        x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)
        x_list.append(x)

    x = Concatenate(axis=1)(x_list)
    x = Dense(512, activation='elu')(x)

    x = CollapseRagged('sum')([x, rs])
    x = ScaledGooeyBatchNorm2(**batchnorm_parameters)(x)

    features = Dense(12, activation="linear")(x)
    sigma = Dense(12, activation="relu")(x)
  
    outputs = Concatenate(axis=1)([features, sigma])
    return Model(inputs=Inputs, outputs=outputs)


if __name__ == "__main__":
    train = training_base()
    loss = nntr_L2_distance(train_uncertainties=True,
                            epsilon=0.001)

    if not train.modelSet():
        train.setModel(nntr_two_vertex_fitter)
        train.saveCheckPoint("before_training.h5")
        train.setCustomOptimizer(tf.keras.optimizers.Adam())
        train.compileModel(learningrate=1e-3,
                           loss=loss)
        train.keras_model.summary()

    cb = [simpleMetricsCallback(
            output_file=train.outputDir+'/losses.html',
            record_frequency=5,
            plot_frequency=5,
            select_metrics='*loss')]

    nbatch = 50000
    train.change_learning_rate(5e-4)
    train.trainModel(nepochs=5,
                     batchsize=nbatch,
                     additional_callbacks=cb)

    nbatch = 200000
    train.change_learning_rate(1e-4)
    train.trainModel(nepochs=20,
                     batchsize=nbatch,
                     additional_callbacks=cb)

    nbatch = 200000
    train.change_learning_rate(1e-5)
    train.trainModel(nepochs=30,
                     batchsize=nbatch,
                     additional_callbacks=cb)
