import numpy as np
import awkward as ak
import pandas as pd
import uproot
import sys
import pickle
        

class ExtractFromRootFile:
    def __init__(self, filename):
        self.filename = filename.rstrip('.root')

        print("Find hits")
        hits = self._find_hits()
        print("Found hits")
        truth = self._find_truth()
        print("Found truth")
        offsets = self._find_offsets(hits)

        all_features = {"hits": hits, "truth": truth, "offsets": offsets}

        with open(f"{self.filename}_preprocessed.pkl", "wb") as file:
            pickle.dump(all_features, file)
        print("Successfully written to file")

    def _extract_to_array(self, key) -> ak.Array:
        """ This method extracts the events and hits for a given key in the root
        tree
        """
        with uproot.open(self.filename+".root") as data:
            return data[key].array(library="ak")[:100]

    def _find_hits(self) -> pd.DataFrame:
        keys = ["layerType", "cellID", "x", "y", "z", "E"]
        prefix = "Events/Output/fHits/fHits."
        d = {}

        print(f"[1/3] Extract from root file")
        for keys in keys:
            d[keys] = self._extract_to_array(prefix+keys)
            d[keys] = d[keys].mask[d["layerType"] > 0]

        print("[2/3] Convert to DataFrame and sum over same cellIDs (will take approx. " +
            f"{7/100*len(d['layerType']):.0f} seconds)")
        df_original = ak.to_dataframe(d)
        df_calo = df_original[df_original["layerType"] == 2]
        
        df_calo_grouped = df_calo.groupby(["entry", "cellID"])[["layerType", "x", "y", "z"]].first()
        df_calo_grouped["E"] = df_calo.groupby(["entry", "cellID"])["E"].sum()

        print("[3/3] Concatenate DataFrames")
        df = pd.concat([df_calo[df_calo["layerType"] == 1], df_calo_grouped], join="outer")
        return df

    def _find_truth(self) -> np.ndarray:
        prefix = "Events/Output/fParticles/fParticles."
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        truth, mean, std = {}, [], []

        for key in keys:
            truth[key] = self._extract_to_array(prefix+key)[..., np.newaxis]
            mean = np.append(mean, np.mean(ak.to_list(truth[key]), axis=0))
            std = np.append(std, np.std(ak.to_list(truth[key]), axis=0))

        with open(f"{self.filename}_normalization.pkl", "wb") as file:
            data = {"mean": mean, "std": std}
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

            print(f"Saved normalization data to: {self.filename}_normalization.pkl")

    def _find_offsets(self, df) -> np.ndarray:
        offsets = df.reset_index(level=1).index.value_counts().sort_index().to_numpy()
        return  np.cumsum(offsets).astype(int)

if __name__ == "__main__":
    filename = sys.argv[1]
    ExtractFromRootFile(filename)
