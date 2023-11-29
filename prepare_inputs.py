import numpy as np
import awkward as ak
import pandas as pd
import uproot
import sys
import itertools
import matplotlib.pyplot as plt


class PrepareInputs:
    def __init__(self, filename):
        self.filename = filename.rstrip('.root')
        
        # Truth
        truth_array = self._find_truth()

        def line(truth, particle_num, z=0):
            if particle_num == 1:
                p = truth[:, 0:3]
                v = truth[:, 3:6]
            elif particle_num == 2:
                p = truth[:, 6:9]
                v = truth[:, 9:12]
            x = p[:, 0]/p[:, 2]*(z - v[:, 2])
            y = p[:, 1]/p[:, 2]*(z - v[:, 2])
            z = np.repeat(z, len(p))
            return np.stack([x, y, z], axis=-1)

        A1 = line(truth_array, 1, 0)
        B1 = line(truth_array, 1, 300)
        A2 = line(truth_array, 2, 0)
        B2 = line(truth_array, 2, 300)
        V1 = truth_array[:, 3:6]
        V2 = truth_array[:, 9:12]

        truth_points = np.concatenate([A1, B1, A2, B2, V1, V2], axis=1)
        print("Truth points:    ", truth_points.shape)
        print("Truth points[0]: ", truth_points[0])

        # Hits
        raw_array = self._find_hits()

        print("[2/3] Convert to DataFrame and sum over same cellIDs (will take approx. " +
            f"{0.7/100*len(raw_array[0]):.0f} seconds)")

        offsets = ak.count(raw_array, axis=1)[:, 0]
        hits_flattened = np.array(ak.to_list(ak.flatten(raw_array)))
        print("Hits flattened: ", len(hits_flattened))

        # Create dataFrame (it is faster to create the pd.MultiIndex by hand than using
        # ak.to_dafaframe)
        truth_points_repeated = list(np.repeat(truth_points, offsets, axis=0))
        i_0 = list(np.repeat(np.arange(len(offsets)), offsets))
        i_1 = list(itertools.chain.from_iterable(
            [(np.arange(i)) for i in ak.to_list(offsets)]))
        nindex = pd.MultiIndex.from_arrays(
            arrays=[i_0, i_1], names=["event", "hit"])
        values = np.concatenate([hits_flattened, truth_points_repeated], axis=1)
        print("Values: ", values.shape)
        print("Values[0]: ", values[0])
        truth_keys = ["A1x", "A1y", "A1z", "B1x", "B1y", "B1z",
                "A2x", "A2y", "A2z", "B2x", "B2y", "B2z",
                "V1x", "V1y", "V1z", "V2x", "V2y", "V2z"]
        df_original = pd.DataFrame(
            values,
            columns=[
                "layerType", "cellID", "x", "y", "z", "E",
                *truth_keys],
            index=nindex)
        print("DataFrame: ", df_original)

        # Sum over same cellIDs in the calorimeter
        df_calo = df_original[df_original["layerType"] == 2]
        df_calo_grouped = df_calo.groupby(["event", "cellID"])[
            ["layerType", "x", "y", "z", *truth_keys]].first()
        df_calo_grouped["E"] = df_calo.groupby(["event", "cellID"])["E"].sum()

        self.offsets_for_truth_empty_events = ak.to_numpy(offsets)

        print("[3/3] Concatenate DataFrames")
        df = pd.concat(
            [df_original[df_original["layerType"] == 1], df_calo_grouped])
        self.offsets = self._find_offsets(df)

        # Remove cellID as an entry and normalize hits
        hits = np.delete(df.to_numpy(), 1, axis=1)
        mean = np.mean(hits, axis=0)+1E-10
        std = np.std(hits, axis=0)+1E-10
        hits_normalized = (hits-mean)/std
        print(df)
        print(hits.shape)

        # Write to file
        offsets_cumsum = np.append(0, np.cumsum(offsets))

        # with uproot.recreate(f"{self.filename}_preprocessed.root") as file:
        #     file["Hits"] = {"hits": hits, "hits_normalized": hits_normalized}
        #     file["Hits_offsets_cumsum"] = {"offsets_cumsum": offsets_cumsum}
        #     file["Hits_offsets"] = {"offsets": offsets}
        #     file["Hits_parameters"] = {"hits_mean": hits_mean, "hits_std": hits_std}
        #     file["Truth"] = {"truth_normalized": truth_normalized}
        #     file["Truth_parameters"] = {"truth_mean": truth_mean, "truth_std": truth_std}

    def _find_hits(self):
        keys = ["layerType", "cellID", "x", "y", "z", "E"]
        prefix = "Events/Output/fHits/fHits."
        raw_array = []

        print("[1/3] Extract from root file")
        for keys in keys:
            raw_array.append(self._extract_to_array(prefix+keys)[..., np.newaxis])
        raw_array = ak.concatenate(raw_array, axis=-1)
        return raw_array

    def _find_truth(self):
        """ Shape:
        (events x 12)
        """
        prefix = "Events/Output/fParticles/fParticles."
        keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]
        truth_array = []

        for key in keys:
            truth_array.append(self._extract_to_array(prefix+key)[..., np.newaxis])
        truth_array = ak.concatenate(truth_array, axis=-1)
        truth_array = ak.flatten(truth_array, axis=-1)
        truth_array = np.array(truth_array.tolist())
        return truth_array

    def _extract_to_array(self, key) -> ak.Array:
        """ This method extracts the events and hits for a given key in the root
        tree

        Returns shapes:
        (hits x [layerType, x, y, z, E])
        """
        with uproot.open(self.filename+".root") as data:
            return data[key].array(library="ak")[1000:2000]

    def _find_offsets(self, df: pd.DataFrame) -> np.ndarray:
        return df.reset_index(level=1).index.value_counts().sort_index().to_numpy()
    
    def plot(self, truth_array, truth_points, raw_array):
        # PLOTS
        # -----------------------
        plt.figure(figsize=(10, 10))
        v1 = truth_array[0, 3:6]
        p1 = truth_array[0, 0:3]
        plt.scatter(A1[0, 2], A1[0, 0], s=200, label="A1", marker="+")
        plt.scatter(B1[0, 2], B1[0, 0], s=200, label="B1", marker="+")
        plt.scatter(A2[0, 2], A2[0, 0], s=200, label="A2", marker="+")
        plt.scatter(B2[0, 2], B2[0, 0], s=200, label="B2", marker="+")
        plt.scatter(V1[0, 2], V1[0, 0], s=200, label="V1", marker="+")
        plt.scatter(V2[0, 2], V2[0, 0], s=200, label="V2", marker="+")

        tracker = raw_array[0][raw_array[0][:, 0] == 1]
        calo = raw_array[0][raw_array[0][:, 0] == 2]
        plt.scatter(tracker[:, 4], tracker[:, 2], s=5, alpha=0.5, label="Tracker")
        plt.scatter(calo[:, 4], calo[:, 2], s=5, alpha=0.5, label="Calorimeter")
        plt.plot(
            [v1[2], p1[2]],
            p1[0]/p1[2]*([v1[2], p1[2]] - v1[2]),
            color="k",
            linestyle="dashed",
            alpha=0.5,
            label="Truth 1")
        v2 = truth_array[0][9:12]
        p2 = truth_array[0][6:9]
        plt.plot(
            [v2[2], p2[2]],
            p2[0]/p2[2]*([v2[2], p2[2]] - v2[2]),
            color="k",
            linestyle="dashed",
            alpha=0.5,
            label="Truth 2")
        plt.legend()
        plt.xlabel("z")
        plt.xlim(-1000, 500)
        plt.ylabel("x")
        plt.savefig("Plot.png")
        # -----------------------


if __name__ == "__main__":
    filename = sys.argv[1]
    PrepareInputs(filename)
