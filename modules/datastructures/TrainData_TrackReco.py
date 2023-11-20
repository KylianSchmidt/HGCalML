from DeepJetCore.TrainData import TrainData
from DeepJetCore import SimpleArray
import numpy as np
import uproot
import awkward as ak
import pandas as pd
import pickle
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainData_TrackReco(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.description = "This Class converts Root files from a Geant4 " + \
            "simulation to Awkward and Numpy arrays"

    def convertFromSourceFile(self, filename, weightobjects, istraining):
        """ Construct the feature, truth (and weightobejects) from a root file
        The data from the geant4 root files are jagged arrays of shape

            Properties x hits x eventNum

        Where properties are listed here as keys. The prefix guides uproot
        to the correct branch->event

        Truth has the shape:
            0   1   2   3   4   5   6   7   8   9   10  11
           [px1 py1 pz1 vx1 vy1 vz1 px2 py2 pz2 vx2 vy2 vz2]
        """
        self.filename = filename

        with open(filename) as file:
            data = pickle.load(file, "b")
            feature = data["hits"]
            truth = data["truth"]
            offsets = data["offsets"]
        
        truth = truth.astype(
            dtype='float32',
            order='C',
            casting="same_kind")
        
        feature = feature.astype(
            dtype='float32',
            order='C',
            casting="same_kind")
        
        return ([SimpleArray(feature, offsets, name="Features")], [truth], [])

    def writeOutPrediction(
            self,
            predicted,
            features,
            truth,
            weights,
            outfilename,
            inputfile) -> None:
        """ Defines the way the predicted data is stored to disk. Currently
        done using pickle.

        Predicted will be a list of numpy arrays
        """

        data = {
            "Hits": features,
            "Truth": truth[0],
            "Predicted": predicted[0][:, 0:12]}
        if len(predicted[0]) == 24:
            data["Uncertainties"] = predicted[0][:, 12:24]

        with open(outfilename, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        return None
