import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import uproot
import sys
import os
from datetime import datetime
from typing import Type, TypeVar, Literal
import logging
logger = logging.getLogger(__name__)

today = datetime.today().strftime("%Y-%m-%d")


class Extract:
    """ Class which automatically extracts the data from the neural network
    output (.djctd file) and converts it to awkward arrays

    Notes
    -----
    Opens the file <model_dir><model_name>/Predicted/pred_Testing.djctd"

    Parameters
    ----------
    model_dir : str
        Directory where the models are stored
    model_name : str
        Specific name of the model output
    root_files_dir : str
        Directory of the root files used by the network

    Returns
    -------
    Data container of Type[TObject] with objects:
        self.predicted\n
        self.truth\n
        self.uncertainties (if provided, empty otherwise)\n
    And additional information to be passed to plotting tools
        self.model_dir\n
        self.model_name\n
    """

    def __init__(self,
                 model_dir="./nntr_models",
                 model_name="",
                 read_hits=False,
                 testing_root_files="./nntr_data/Raw/Testing.root"):

        self.model_dir = model_dir+"/"
        self.model_name = model_name
        self.nn_raw_data = {}
        self.testing_root_files = testing_root_files
        self.predicted = self.nn_data(key="Predicted")
        self.truth = self.nn_data(key="Truth")
        self.hits = []
        self.ATCCellNumCols = np.NaN
        self.ATCCellNumRows = np.NaN
        self.ATLayerNum = np.NaN
        self.CalorimeterThickness = np.NaN
        self.ATCellSideLength = np.NaN

        if "Uncertainties" in self.nn_raw_data.keys():
            self.uncertainties = self._nn_data(key="Uncertainties")
        else:
            self.uncertainties = []

        if read_hits:
            self._read_hits()

    def _undo_normalization(self, array: np.array):
        """ Undo the normalization from the preprocessing step.
        The ordering here is the same as in 'normalize_inputs.py' which
        is different than in the truth array of the network
        """
        # Only temporary as this will be handled by conversion
        # in DJC
        #
        # Update: no it wont, blame it on DJC
        logger.info("Undo normalization, mean was", np.mean(array, axis=0))

        with open(self.testing_root_files.rstrip(".root")+"_normalization.pkl",
                  "rb") as file:
            data = pickle.load(file)
            std, mean = np.array([]), np.array([])
            keys = ["mcPx", "mcPy", "mcPz", "decayX", "decayY", "decayZ"]

            for i in [0, 1]:
                for key in keys:
                    mean = np.append(mean, data["mean"][key][i, 0])
                    std = np.append(std, data["std"][key][i, 0])

        original_array = array*(std + 1E-10) + mean

        logger.info("Mean is now", np.mean(original_array))
        return original_array

    def _nn_data(self, key="Truth"):
        if self.model_name not in os.listdir(self.model_dir):
            logger.error("No such model could be found at:\n" +
                         f"{self.model_dir}/{self.model_name}\n" +
                         "Available models are:", os.listdir(self.model_dir))
        else:
            with open(f"{self.model_dir}/{self.model_name}/"
                      + "Predicted/pred_Testing.djctd", "rb") as file:
                self.nn_raw_data = pickle.load(file)

        return self._nn_find_physical_variables(self.nn_raw_data[key])

    def _nn_find_physical_variables(self, array: np.array):
        array = self.undo_normalization(array)

        if np.shape(array)[1] == 12:
            return {
                "p1": array[:, 0:3],
                "v1": array[:, 3:6],
                "p2": array[:, 6:9],
                "v2": array[:, 9:12]}
        else:
            logger.error(f"Wrong shape of array, is {np.shape(array)}")

    def read_hits(self):
        filename = self.testing_root_files

        with uproot.open(filename) as event:
            keys = ["layerType", "layerID", "cellID", "E", "x", "y", "z"]
            self.hits = {}
            for name in keys:
                self.hits[name] = event["Events"]["Output"]["fHits"][f"fHits.{name}"].array(library="ak")

            self.ATCCellNumCols = event["Events"]["ParametersNumericals"]["fATCCellNumCols"].array(library="np")[0]
            self.ATCCellNumRows = event["Events"]["ParametersNumericals"]["fATCCellNumRows"].array(library="np")[0]
            self.ATLayerNum = event["Events"]["ParametersNumericals"]["fNumberATLayers"].array(library="np")[0]
            self.CalorimeterThickness = event["Events"]["ParametersNumericals"]["fCalorimeterThickness"].array(library="np")[0]
            self.ATCellSideLength = event["Events"]["ParametersNumericals"]["fAbsorberSideLength"].array(library="np")[0]

        return None


TOutput = TypeVar("TOutput", bound=Extract)


class Plot:

    def savefig(savefigdir,
                data=Type[TOutput],
                plot_type="",
                obs=""):
        if not savefigdir:
            savefigdir = f"{data.model_dir}/{data.model_name}/Plots/"
        os.makedirs(savefigdir, exist_ok=True)
        plt.savefig(f"{savefigdir}/{plot_type}_{obs}.png", dpi=600)
    
    def gaussian_uncertainties(ax: plt.axis,
                               data: Type[TOutput],
                               feature,
                               errors):
        mean = np.mean(feature, axis=0)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        h = np.histogram(feature, bins=nbins)
        height = h[0][h[0].argmax()]
        ax.scatter(x=mean*scalar_factor,
                    y=height,
                    color=colors[0])
        ax.errorbar(x=mean*scalar_factor,
                    y=height,
                    color=colors[0],
                    xerr=errors*scalar_factor,
                    label="Predicted Error",
                    ls="")

    def scatter(data=Type[TOutput],
                obs="v1",
                axis=("x", "y"),
                units="mm",
                xlim=(),
                ylim=(),
                scalar_factor=1,
                savefigdir="") -> None:

        """ Produce a scatter plot in the plane defined by 'axis' to
        compare the predicted and true distributions of 'obs'.

        Parameters
        ----------
        data : Type[TOutput]
            Data produced by the Extract class
        obs : str
            Name of the observable ('v1' or 'v2) to be plotted
        units : str
        scalar_factor : float
            Factor for all quantities to account for unit conversions
            (e.g. MeV -> Gev)
        """

        p = data.predicted[obs]
        t = data.truth[obs]
        ax = {"x": 0, "y": 1, "z": 2}

        plt.figure()
        plt.scatter(x=p[..., ax[axis[0]]] * scalar_factor,
                    y=p[..., ax[axis[1]]] * scalar_factor,
                    alpha=0.1, marker=".",
                    label="Predicted")
        plt.scatter(x=t[..., ax[axis[0]]] * scalar_factor,
                    y=t[..., ax[axis[1]]] * scalar_factor,
                    alpha=0.3, marker=".",
                    label="True")
        plt.legend()
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel(f"{axis[0]} [{units}]")
        plt.ylabel(f"{axis[1]} [{units}]")
        plt.title(f"Model {data.model_name}")
        Plot.savefig(savefigdir, data, f"scatter_{axis[0]}_{axis[1]}", obs)
        return None

    def df(data=Type[TOutput],
           mean=False,
           eventID=0):

        predicted = data.predicted
        truth = data.truth
        uncertainties = data.uncertainties
        iterables = [["Mean", "Uncertainties"],
                     ["Predicted", "Truth"]]
        obs = predicted.keys()

        if mean:
            header = pd.MultiIndex.from_product(iterables=iterables,
                                                names=["DataType", "Property"])
            df = pd.DataFrame(columns=header)
            
            for i, ob in enumerate(obs):
                for a, axis in enumerate({"x": 0, "y": 1, "z": 2}):
                    df.loc[ob+axis, ("Mean", "Predicted")] = np.mean(predicted[ob], axis=0)[a]
                    df.loc[ob+axis, ("Uncertainties", "Predicted")] = np.mean(uncertainties[ob], axis=0)[a]
                    df.loc[ob+axis, ("Mean", "Truth")] = np.mean(truth[ob], axis=0)[a]
                    df.loc[ob+axis, ("Uncertainties", "Truth")] = np.std(truth[ob], axis=0)[a]
        else:
            header = pd.MultiIndex.from_product(iterables=[[eventID],
                                                           ["Predicted",
                                                            "Truth"]],
                                                names=["EventID", "Property"])
            df = pd.DataFrame(index=obs, columns=header)

            for i, ob in enumerate(obs):
                df.loc[ob, (eventID, "Predicted")] = predicted[eventID, i]
                df.loc[ob, (eventID, "Truth")] = truth[eventID, i]

        return df

    def tracks(data=Type[TOutput],
               eventID=0,
               axis="x",
               savefigdir=""):
        """ Method which calls the FullTrackReco class and plots the hits as
        well as the true and predicted trajectory of the photons based on the
        output from the network
        """
        if not data.hits:
            logger.error("Error: Root files have not been extracted " +
                         "yet. Use Extract.read_hits() or set " +
                         "'read_hits' to True when using Extract().")

        ftr = FullTrackReco(data, eventID)
        plot = ftr.plot_mpl(axis=axis)
        return plot


class FullTrackReco:
    def __init__(self,
                 data=Type[TOutput],
                 eventID=0,
                 particle="photon",
                 zlim=-10):
        self.nn_raw_data = []
        self.particle = particle
        self.zlim = zlim
        self.eventID = eventID
        self.data = data
        self.ATLayerNum = data.ATLayerNum
        self.ATCCellNumCols = data.ATCCellNumCols
        self.ATCCellNumRows = data.ATCCellNumRows
        self.ATCellSideLength = data.ATCellSideLength
        self.CalorimeterThickness = data.CalorimeterThickness

        if self.ATLayerNum <= 5:
            self.colors = ["purple", "blue", "darkred", "red", "darkorange"]
        else:
            self.colors = ["darkblue"]*int(self.ATLayerNum)

        if self.data.hits:
            hits_event = {}
            for name in self.data.hits.keys():
                hits_event[name] = self.data.hits[name][eventID]
            self.df = pd.DataFrame(hits_event)

    def tracker_hits(self, layerID):
        """ Find the tracker hits for each layer
        """
        tracker = self.df[self.df["layerType"] == 1]
        tracker_layer = tracker[tracker["layerID"] == layerID]
        return tracker_layer

    def plot_calo(self):
        """ Heatmap of the Calorimeter energy deposition with tracker hits"""
        self.fig = plt.figure(figsize=(6, 5))
        self.ax = self.fig.subplots()

        # Calo hits
        calo = self.df[self.df["layerType"] == 2]
        cg = np.zeros(self.ATCCellNumRows*self.ATCCellNumCols)
        cg[calo["cellID"]] = 1e3*calo["E"]  # Are in GeV in the root file
        calo_colormesh = np.reshape(cg, (self.ATCCellNumRows,
                                         self.ATCCellNumCols))
        width = self.ATCellSideLength*np.arange(
                                        -np.floor(self.ATCCellNumCols/2),
                                        np.ceil(self.ATCCellNumCols/2))
        height = self.ATCellSideLength*np.arange(
                                        -np.floor(self.ATCCellNumRows/2),
                                        np.ceil(self.ATCCellNumRows/2))
        im = self.ax.pcolormesh(width,
                                height,
                                calo_colormesh,
                                cmap="GnBu",
                                alpha=0.7,
                                edgecolors="lightblue",
                                linewidth=0.5)
        plt.colorbar(mappable=im,
                     pad=0.01,
                     label="Calorimeter Energy Deposition [MeV]")

        # Tracker hits
        for layerID in range(0, self.ATLayerNum):
            trh = self.tracker_hits(layerID)

            if self.ATLayerNum <= 5:
                label = f"Tracking Layer {layerID}"
            else:
                label = None

            self.ax.scatter(x=trh["x"].values,
                            y=trh["y"].values,
                            marker=".",
                            c=self.colors[layerID],
                            label=label)

        self.energy = 1e3*np.sum(calo["E"])
        t = self.ax.text(x=0.50,
                         y=0.88,
                         s=f"{self.particle}, {self.ATLayerNum} AT layers",
                         transform=self.ax.transAxes)
        t.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor="k"))

        self.ax.set_xlim(width[0]-20, width[-1]+20)
        self.ax.set_ylim(height[0]-20, height[-1]+20)
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")
        plt.legend(loc="upper left")
        plt.title("Detector Hits")
        plt.tick_params(direction="in")

    def plot_mpl(self,
                 axis: Literal["x", "y"] = "x"):

        fig, plot = plt.subplots(figsize=(6, 6))
        ax = {"x": 0, "y": 1}[axis]

        # Trajectories
        p = self.data.predicted
        t = self.data.truth
        scalar_factor = 1E3

        # Predicted
        p1 = p["p1"][self.eventID]*scalar_factor
        v1 = p["v1"][self.eventID]*scalar_factor
        plot.plot([v1[2], p1[2]],
                  p1[ax]/p1[2]*([v1[2], p1[2]] - v1[2]),
                  color="green",
                  label="Reconstr. photon 1")

        p2 = p["p2"][self.eventID]*scalar_factor
        v2 = p["v2"][self.eventID]*scalar_factor
        plot.plot([v2[2], p2[2]],
                  p2[ax]/p2[2]*([v2[2], p2[2]] - v2[2]),
                  color="lightgreen",
                  label="Reconstr. photon 2")

        # Truth
        p1 = t["p1"][self.eventID]*scalar_factor
        v1 = t["v1"][self.eventID]*scalar_factor
        plot.plot([v1[2], p1[2]], 
                  p1[ax]/p1[2]*([v1[2], p1[2]] - v1[2]),
                  color="k",
                  linestyle="dashed",
                  alpha=0.5)

        p2 = t["p2"][self.eventID]*scalar_factor
        v2 = t["v2"][self.eventID]*scalar_factor
        plot.plot([v2[2], p2[2]], 
                  p2[ax]/p2[2]*([v2[2], p2[2]] - v2[2]),
                  color="k",
                  label="True trajectory",
                  linestyle="dashed",
                  alpha=0.5)

        # Tracker
        for layerID in range(0, self.ATLayerNum):
            trh = self.tracker_hits(layerID)

            if self.ATLayerNum <= 5:
                label = f"Tracking Layer {layerID}"
            else:
                label = None

            if not trh["z"].empty:
                plot.scatter(x=trh["z"].values,
                             y=trh[axis].values,
                             marker=".",
                             c=self.colors[layerID],
                             label=label)
                plot.vlines(x=trh["z"].values,
                            ymin=-self.ATCellSideLength*self.ATCCellNumRows/2,
                            ymax=self.ATCellSideLength*self.ATCCellNumCols/2,
                            color=self.colors[layerID],
                            alpha=0.1)
            else:
                pass

        # Calorimeter
        self.calo = self.df[self.df["layerType"] == 2]
        if not self.calo["z"].empty:
            print(self.calo)
            plot.vlines(x=self.calo["z"],
                        ymin=self.calo[axis]-self.ATCellSideLength,
                        ymax=self.calo[axis]+self.ATCellSideLength,
                        alpha=0.7,
                        colors="k")
            self.calo_center = self.calo["z"].iloc[0]
        else:
            self.calo_center = 450
            pass

        detector = patches.Rectangle(
                          xy=(0, -self.ATCellSideLength*self.ATCCellNumRows/2),
                          width=+self.CalorimeterThickness/2,
                          height=self.ATCellSideLength*self.ATCCellNumRows,
                          color="grey",
                          alpha=0.1)
        plot.add_patch(detector)
        return fig, plot


if __name__ == "__main__":
    model_name = sys.argv[1]

    read_hits = False
    data = Extract(model_dir="./nntr_models/idealized_detector",
                   model_name=model_name,
                   read_hits=read_hits,
                   testing_root_files="./nntr_data/idealized_detector/Raw/Testing.root")

    Plot.histogram(data=data,
                   obs="p1",
                   xlabel="Normalized Momentum [MeV]",
                   scalar_factor=1E3,
                   nbins=np.linspace(-2.5E3, 2.5E3, 100))

    Plot.histogram(data=data,
                   obs="v1",
                   xlabel="Vertex Position [mm]",
                   scalar_factor=1E3,
                   nbins=np.linspace(-1200, 100, 100))

    Plot.scatter(data=data,
                 obs="v1",
                 xlim=(-1, 1),
                 ylim=(-1, 1),
                 scalar_factor=1,
                 units="m")

    Plot.scatter(data,
                 "v1",
                 xlim=(-2, 0),
                 ylim=(-2, 2),
                 scalar_factor=1,
                 axis=("z", "x"),
                 units="m")
