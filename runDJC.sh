#!/bin/bash

gpuopt=""
files=$(ls -l /dev/nvidia* 2> /dev/null | egrep -c '\n')
if [[ "$files" != "0" ]]
then
gpuopt="--nv"
fi

sing=`which singularity`

$sing run $gpuopt -B /home /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
#/home/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/cvmfs/deepjetcore3_latest.sif
