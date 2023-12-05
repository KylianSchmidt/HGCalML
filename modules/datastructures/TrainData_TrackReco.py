from DeepJetCore.TrainData import TrainData
from DeepJetCore import SimpleArray
import uproot
import pickle


class TrainData_TrackReco(TrainData):
    def __init__(self):
        TrainData.__init__(self)
        self.description = "This Class converts Root files from a Geant4 " + \
            "simulation to Awkward and Numpy arrays"

    def convertFromSourceFile(self, filename, weightobjects, istraining):
        """ Construct the feature, truth from a root file

        Relies on the structure provided by the ExtractFromRootFile class

        The data from the geant4 root files are jagged arrays of shape

            Properties x hits x eventNum

        Where properties are listed here as keys. The prefix guides uproot
        to the correct branch->event

        Truth has the shape:
            0   1   2   3   4   5   6   7   8   9   10  11
           [px1 py1 pz1 vx1 vy1 vz1 px2 py2 pz2 vx2 vy2 vz2]
        """

        with uproot.open(filename) as file:
            hits_normalized = file["Hits"]["hits_normalized"].arrays(library="np")[
                "hits_normalized"]
            offsets = file["Hits_row_splits"]["rowsplits"].arrays(library="np")[
                "rowsplits"]
            truth_normalized = file["Truth"]["truth_normalized"].arrays(library="np")[
                "truth_normalized"]

        truth = truth_normalized.astype(
            dtype='float32',
            order='C',
            casting="same_kind")

        feature = hits_normalized.astype(
            dtype='float32',
            order='C',
            casting="same_kind")

        assert offsets[0] == 0 and offsets[-1] == len(feature)
        assert len(offsets)-1 == len(truth)

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
            "Predicted": predicted[0]
        }

        with open(outfilename, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        return None
