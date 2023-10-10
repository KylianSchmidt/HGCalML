#!/bin/bash

detector_type=""
model_name=""
train_uncertainties=""

while getopts "d:m:u" flag; do
    case "${flag}" in
        d) detector_type=$OPTARG;;
        m) model_name=$OPTARG;;
        u) train_uncertainties="--train_uncertainties";;
    esac
done

nntr_model_dir="./nntr_models/$detector_type/$model_name"

echo "Train the NN using the following model:"
echo $nntr_model_dir

# Train
echo "Starting Training"
SECONDS=0

mkdir $nntr_model_dir
rm -rf $nntr_model_dir/Output

python3 Train/TrackReco_training.py \
        ./nntr_data/$detector_type/Training/dataCollection.djcdc \
        $nntr_model_dir/Output \
        --valdata ./nntr_data/$detector_type/Training/dataCollection.djcdc \
        $train_uncertainties

echo $SECONDS

# Prediction
echo "Starting Prediction"

rm -rf $nntr_model_dir/Predicted
python3 predict.py  $nntr_model_dir/Output/KERAS_check_best_model.h5 \
                    $nntr_model_dir/Output/trainsamples.djcdc \
                    ./nntr_data/$detector_type/Testing/Testing.djctd \
                    $nntr_model_dir/Predicted



