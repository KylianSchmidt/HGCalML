'''
Track Reconstruction algorithm for the simulation of the LUXE experiment
Uses the GravNet architecture 
'''
from DeepJetCore.training.training_base import training_base

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate
from RaggedLayers import CollapseRagged
from GravNetLayersRagged import CastRowSplits, ScaledGooeyBatchNorm2, RaggedGravNet
import tensorflow as tf

def pretrain_model(Inputs):
    # x are ragged arrays containing the features of the detector hits in the form 
    #    [eventNum x hits] x properties 
    # where eventNum x hits are separated by TensorFlow using the rowsplits rs
    x, rs = Inputs

    rs = CastRowSplits()(rs)
    x = ScaledGooeyBatchNorm2()(x)

    for _ in range(5):
        x,*_ = RaggedGravNet(
                 n_neighbours = 32,
                 n_dimensions = 5,
                 n_filters = 64,
                 n_propagate = 64,
                 feature_activation='elu')([x,rs])
        
        x = Dense(64, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = ScaledGooeyBatchNorm2()(x)

    x = CollapseRagged('mean')([x,rs])
    #xm = CollapseRagged('sum')([x,rs])
    #x = Concatenate()([x,xm])
    x = ScaledGooeyBatchNorm2()(x)
    
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = Dense(64, activation='elu')(x)
    x = ScaledGooeyBatchNorm2()(x)

    outputs = Dense(3)(x)
    
    return Model(inputs=Inputs, outputs=outputs)



train=training_base()
# Custom loss function
from Losses import loss_reduceMean
from Losses import loss_track_distance

if not train.modelSet():
    train.setModel(pretrain_model)
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
    
    train.compileModel(learningrate=1e-8,
                   loss=loss_track_distance)
    
    train.keras_model.summary()
    
    #start somewhere
    #from model_tools import apply_weights_from_path
    #import os
    #path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
    #train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    
from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
cb = [
    simpleMetricsCallback(
        output_file=train.outputDir+'/losses.html',
        record_frequency= 10,
        plot_frequency = 5,
        select_metrics='*loss'
        ),]

nbatch = 1500 
train.change_learning_rate(5e-4)
train.trainModel(nepochs=3, batchsize=nbatch, additional_callbacks=cb)

exit()

nbatch = 150000 
train.change_learning_rate(3e-5)
train.trainModel(nepochs=10,batchsize=nbatch, additional_callbacks=cb)

print('reducing learning rate to 1e-4')
train.change_learning_rate(1e-5)
nbatch = 200000 

train.trainModel(nepochs=100,batchsize=nbatch, additional_callbacks=cb)
