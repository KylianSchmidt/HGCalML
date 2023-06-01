'''


Compatible with the dataset here:
/eos/home-j/jkiesele/ML4Reco/Gun20Part_NewMerge/train

On flatiron:
/mnt/ceph/users/jkieseler/HGCalML_data/Gun20Part_NewMerge/train

not compatible with datasets before end of Jan 2022

'''

from DeepJetCore.training.training_base import training_base

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from RaggedLayers import RaggedCollapseHitInfo

def pretrain_model(Inputs):
    x, rs = Inputs


    x = RaggedCollapseHitInfo()([x,rs])


    outputs = Dense(3)(x)
    
    return Model(inputs=Inputs, outputs=outputs)



train=training_base()
from Losses import myloss



if not train.modelSet():
    train.setModel(pretrain_model)
    
    train.saveCheckPoint("before_training.h5")
    train.setCustomOptimizer(tf.keras.optimizers.Adam())
    #
    train.compileModel(learningrate=1e-4,
                   loss=myloss)
    
    train.keras_model.summary()
    
    #start somewhere
    #from model_tools import apply_weights_from_path
    #import os
    #path_to_pretrained = os.getenv("HGCALML")+'/models/pre_selection_jan/KERAS_model.h5'
    #train.keras_model = apply_weights_from_path(path_to_pretrained,train.keras_model)
    



nbatch = 150000 
train.change_learning_rate(5e-4)
train.trainModel(nepochs=1, batchsize=nbatch,additional_callbacks=cb)

nbatch = 150000 
train.change_learning_rate(3e-5)
train.trainModel(nepochs=10,batchsize=nbatch,additional_callbacks=cb)

print('reducing learning rate to 1e-4')
train.change_learning_rate(1e-5)
nbatch = 200000 

train.trainModel(nepochs=100,batchsize=nbatch,additional_callbacks=cb)
