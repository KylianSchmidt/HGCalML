#!/bin/bash

nnTrackReco_model_directory="./nntr_models/normal_detector/${1}"

echo "Train the NN using the following model:"
echo $nnTrackReco_model_directory

# Train
echo "Commencing Training"
mkdir $nnTrackReco_model_directory
rm -rf $nnTrackReco_model_directory/Output
python3 Train/TrackReco_training.py \
        ./nntr_data/normal_detector/Training/dataCollection.djcdc \
        $nnTrackReco_model_directory/Output \
        --valdata ./nntr_data/normal_detector/Training/dataCollection.djcdc

# Predict
echo "Commencing Prediction"

rm -rf $nnTrackReco_model_directory/Predicted
predict.py  $nnTrackReco_model_directory/Output/KERAS_check_best_model.h5 \
            $nnTrackReco_model_directory/Output/trainsamples.djcdc \
            ./nntr_data/normal_detector/Testing/Testing.djctd \
            $nnTrackReco_model_directory/Predicted \

# Plotting
