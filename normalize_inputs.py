import numpy as np
import awkward as ak
import uproot
import sys
import pickle


class NormalizeInputs:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.prefix = "Events/Output/fParticles/fParticles."
        self.keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]

        print("Normalizing inputs for file: ", filename)
        self._find_truth(self.filename)

    def _find_truth(self, filename: str) -> np.ndarray:
        truth, mean, std = {}, [], []

        # NOTE The ordering here is different than in the truth array of the
        # network. Take this into consideration when undoing the normalization
        with uproot.open(filename) as data:
            for key in self.keys:
                print(key)
                d = data[self.prefix+key].array(library="ak")
                truth[key] = d[..., np.newaxis]
                mean = np.append(mean, np.mean(ak.to_list(truth[key]), axis=0))
                std = np.append(std, np.std(ak.to_list(truth[key]), axis=0))

        with open(filename.rstrip(".root")+"_normalization.pkl", "wb") as file:
            data = {"mean": mean, "std": std}
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            print("Saved normalization data to: ",
                  filename.rstrip(".root")+"_normalization.pkl")

        return truth


if __name__ == "__main__":
    filename = sys.argv[1]
    NormalizeInputs(filename)
