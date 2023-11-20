import numpy as np
import awkward as ak
import pandas as pd
import uproot
import sys
import itertools
        

class ExtractFromRootFile:
    def __init__(self, filename):
        self.filename = filename.rstrip('.root')

        print("Find hits")
        hits = self._find_hits()
        print("Found hits")
        truth, mean, std = self._find_truth()
        print("Found truth")
        offsets = self._find_offsets(hits)

        all_features = {"hits": hits, "truth": truth, "offsets": offsets, "mean": mean, "std": std}

        with uproot.recreate(f"{self.filename}_preprocessed.root") as file:
            file["Hits"] = hits
            file["Truth"] = truth
            file["Offsets"] = offsets
            file["TruthMean"] = mean
            file["TruthStdDeviation"] = std

    def _extract_to_array(self, key) -> ak.Array:
        """ This method extracts the events and hits for a given key in the root
        tree
        """
        with uproot.open(self.filename+".root") as data:
            return data[key].array(library="ak")[:100]

    def _find_hits(self) -> pd.DataFrame:
        keys = ["layerType", "cellID", "x", "y", "z", "E"]
        prefix = "Events/Output/fHits/fHits."
        raw_data_dict = {}

        print(f"[1/3] Extract from root file")
        for keys in keys:
            raw_data_dict[keys] = self._extract_to_array(prefix+keys)
        
        raw_arr = []
        for key in raw_data_dict.keys():
            raw_arr.append(raw_data_dict[key][..., np.newaxis])
        raw_arr = ak.concatenate(raw_arr, axis=-1)

        print("[2/3] Convert to DataFrame and sum over same cellIDs (will take approx. " +
            f"{0.7/100*len(raw_data_dict['layerType']):.0f} seconds)")
        print("DEBUG 0", len(raw_arr[0]))
        offsets = ak.count(raw_arr, axis=1)[:, 0]
        print("DEBUG 1")
        i_0 = list(np.repeat(np.arange(len(offsets)), offsets))
        print("DEBUG 2")
        i_1 = list(itertools.chain.from_iterable([(np.arange(i)) for i in ak.to_list(offsets)]))
        print("DEBUG 3")
        nindex = pd.MultiIndex.from_arrays(arrays=[i_0, i_1], names=["event", "hit"])
        df_original = pd.DataFrame(
            ak.flatten(raw_arr),
            columns=["layerType", "cellID", "x", "y", "z", "E"],
            index=nindex)

        df_calo = df_original[df_original["layerType"] == 2]
        
        df_calo_grouped = df_calo.groupby(["entry", "cellID"])[["layerType", "x", "y", "z"]].first()
        df_calo_grouped["E"] = df_calo.groupby(["entry", "cellID"])["E"].sum()

        print("[3/3] Concatenate DataFrames")
        df = pd.concat([df_original[df_original["layerType"] == 1], df_calo_grouped])
        print(df["layerType"])
        return df

    def _find_truth(self) -> np.ndarray:
        prefix = "Events/Output/fParticles/fParticles."
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        truth, mean, std = {}, [], []

        for key in keys:
            truth[key] = self._extract_to_array(prefix+key)[..., np.newaxis]
            mean = np.append(mean, np.mean(ak.to_list(truth[key]), axis=0))
            std = np.append(std, np.std(ak.to_list(truth[key]), axis=0))

        return truth, mean, std

    def _find_offsets(self, df) -> np.ndarray:
        offsets = df.reset_index(level=1).index.value_counts().sort_index().to_numpy()
        return  np.cumsum(offsets).astype(int)

if __name__ == "__main__":
    filename = sys.argv[1]
    ExtractFromRootFile(filename)
