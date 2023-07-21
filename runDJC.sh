#!/bin/bash

gpuopt=""
files=$(ls -l /dev/nvidia* 2> /dev/null | egrep -c '\n')
if [[ "$files" != "0" ]]
then
gpuopt="--nv"
fi

#this is a singularity problem only fixed recently
# unset LD_LIBRARY_PATH
# unset PYTHONPATH
sing=`which singularity`
# unset PATH

$sing run $gpuopt -B /home /home/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/cvmfs/deepjetcore3_latest.sif