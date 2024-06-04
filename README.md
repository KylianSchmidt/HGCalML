**NOTE:** Forked by Kylian Schmidt from the HGCalML project


NNTR: Application to beamdump photon reconstruction
===================================================

Structure for training:
 - The main training script is `Train/training_nntr.py`
 - The input data is produced by a separate geant4 simulation. It's format is converted and the data is preprocessed by the `prepare_input.py` script. 
 - After the preprocessing step, the data should be divided into
    - Detector hits: awkward ragged arrays (here we use the excellent uproot and awkward packages ``https://zenodo.org/doi/10.5281/zenodo.4340632``, ``https://github.com/scikit-hep/awkward``) 
    - True MC information: used for the training of the model. It is in the form of a ragged array, but could also be a regular array.
   Both arrays have to be normalized before passing to the model. It is recommended to do this in the preprocessing script and keep the normalisation parameters (mean and standard deviation) on file for later use.
 - The files produced by the preprocessing script are then passed to the `modules/datastructures/TrainData_TrackReco.py` script. The TrainData class converts the awkward ragged to a tensorflow ragged tensor. Further information on how to use the script can be found in `Converting the data from ntuples`.
 - Finally, the loss functions found in `Losses.py` dictate the behaviour of the model. The L2Distance loss function works best, while the loss functions with uncertainties (L2DistanceWithUncertainties and QuantileLoss) are unstable and might diverge.

After the training:
 - The `plotting.py` file contains useful classes that show how to extract the output of the model and convert it to awkward arrays.

List of files to adapt
 - Train/training_nntr.py
 - prepare_input.py
 - modules/datastructures/TrainData_TrackReco.py
 - Losses.py
 - plotting.py (optional)

Use:
 1. Perform the setup step detailled later
 2. Run ``./runDJC.sh`` to enter the container shell
 3. Run ``python3 Train/training_nntr.py`` to run the training and prediction (specify the paths to the .djcdc files for training and predicting) 

HGCalML
===============================================================================

Requirements
  * DeepJetCore 3.X (``https://github.com/DL4Jets/DeepJetCore``)
  * DeepJetCore 3.X container (or latest version in general)
  
For CERN (or any machine with cvmfs mounted), a script to start the latest container use this script:
```
#!/bin/bash

gpuopt=""
files=$(ls -l /dev/nvidia* 2> /dev/null | egrep -c '\n')
if [[ "$files" != "0" ]]
then
gpuopt="--nv"
fi

#this is a singularity problem only fixed recently
unset LD_LIBRARY_PATH
unset PYTHONPATH
sing=`which singularity`
unset PATH
cd

$sing run -B /eos -B /afs $gpuopt /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
```

The package follows the structure and logic of all DeepJetCore subpackages (also the example in DeepJetCore). So as a fresh starting point, it can be a good idea to follow the DeepJetCore example first.

Setup
===========

```
git clone  --recurse-submodules  https://github.com/cms-pepr/HGCalML
cd HGCalML
source env.sh #every time
./setup.sh #just once, compiles custom kernels
```


When developing custom CUDA kernels
===========

The kernels are located in 
``modules/compiled``
The naming scheme should be obvious and must be followed. Compile with make.



Converting the data from ntuples
===========

``convertFromSource.py -i <text file listing all training input files> -o <output dir> -c TrainData_NanoML``
The conversion rule itself is located here:
``modules/datastructures/TrainData_NanoML.py``

The training files (see next section) usually also contain a comment in the beginning pointing to the latest data set at CERN and flatiron.

Standard training and inference
===========
Go to the `Train` folder and then use the following command to start training. The file has code for running plots and more. That can be adapted according to needs.


```
cd Train
```
Look at the first lines of the file `std_training.py` containing a short description and where to find the dataset compatible with that training file. Then execute the following command to run a training.

```
python3 std_training.py <path_to_dataset>/training_data.djcdc <training_output_path>
```
Please notice that the standard configuration might or might not include writing the printout to a file in the training output directory.

For inference, the trained model can be applied to a different test dataset.  Please note that this is slightly *different* from the standard DeepJetCore procedure.

```
predict_hgcal.py <training_output_path>/KERAS_model.h5  <path_to_dataset>/testing_data.djcdc  <inference_output_folder>
```

To analyse the prediction, use the `analyse_hgcal_predictions.py` script.

