import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from typing import Type, TypeVar

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
                 model_dir=".", 
                 model_name="") :
        
        self.model_dir = model_dir+"/"
        self.model_name = model_name
        self.raw_data = []
        self.predicted = self.data(key="Predicted")
        self.truth = self.data(key="Truth")
        self.uncertainties = self.find_uncertainties()
        
    def read_data(self) :
        if not self.model_name in os.listdir(self.model_dir) :
            print("No such model could be found at:")
            print(f"{self.model_dir}{self.model_name}")
        else :
            with open(f"{self.model_dir}/{self.model_name}/"+\
                      "Predicted/pred_Testing.djctd", "rb") as file:
                self.raw_data = pickle.load(file)
            return self.raw_data
        
    def find_physical_variables(self, array: np.array):
        return {
            "p1" : array[:,0:3],
            "n1" : array[:,3],
            "v1" : array[:,4:7],
            "p2" : array[:,7:10],
            "n2" : array[:,10],
            "v2" : array[:,11:14]}
    
    def find_uncertainties(self):
        try :
            if not self.raw_data :
                array = self.read_data()["Predicted"][0]
            else :
                array = self.raw_data["Predicted"][0]
            return {
                "Dp1" : array[:,14:17],
                "Dn1" : array[:,17],
                "Dv1" : array[:,18:21],
                "Dp2" : array[:,21:24],
                "Dn2" : array[:,24],
                "Dv2" : array[:,25:28]}
        except :
            print("No uncertainties could be found")
            return None
    
    def data(self, key="Truth") :
        if not self.raw_data :
            self.physical_data = self.find_physical_variables(
                                                    self.read_data()[key][0])
        else :
            self.physical_data = self.find_physical_variables(
                                                    self.raw_data[key][0])
        return self.physical_data
    
TOutput = TypeVar("TOutput", bound=Extract)



class Plot:

    def savefig(savefigdir,
                data=Type[TOutput],
                plot_type="",
                obs="") :
        if not savefigdir :
            savefigdir = f"{data.model_dir}/{data.model_name}/Plots/"
        os.makedirs(savefigdir, exist_ok=True)
        plt.savefig(f"{savefigdir}/{plot_type}_{obs}.png", dpi=600)

    def histogram(data=Type[TOutput],
                obs="v1",
                xlabel="Distance from detector [mm]",
                scalar_factor=1.0,
                nbins=np.array([]),
                show_uncertainties=True,
                savefigdir="") -> None :
        """ Plot a pair of histograms comparing the predicted and
        true distributions of a given quantity from the network 
        output.

        Notes
        -----
        If uncertainties are included in 'data', they will be added
        automatically to the plots as horizontal errorbars.

        Parameters
        ----------

        data : Type[TOutput]
            Data extracted from the network using the Extract class
        obs : str
            Key in 'data' of the observable to be plotted, can be a
            single array (e.g. Energy) or a 3-vector (e.g. Momentum)
        xlabel : str
            Shared xlabel passed to matplotlib, should include units
        scalar_factor : float
            Factor for all quantities to account for unit conversions
            (e.g. MeV -> Gev)
        """
        
        predicted = data.predicted[obs]
        truth = data.truth[obs]
        uncertainties = data.uncertainties[f"D{obs}"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.suptitle(f"Model {data.model_name}")

        if not nbins.any() :
            xlim = (np.min([predicted, truth])*scalar_factor, 
                    np.max([predicted, truth])*scalar_factor)
            nbins = np.linspace(xlim[0], xlim[1], 50)
        else :
            xlim = nbins[0], nbins[-1]
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

        label = obs
        if len(predicted[0]) == 3 :
            label = [f"{obs}_{i}" for i in ["x", "y", "z"]]

        ax1.hist(predicted*scalar_factor,
                 bins=nbins,
                 histtype="step",
                 alpha=0.9,
                 label=label)

        def gaussian_uncertainties(ax):
            mean = np.mean(predicted, axis=0)
            errors = np.mean(uncertainties, axis=0)
            height = []

            if np.atleast_2d(predicted).shape[0] != 1:
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']

                for i in range(predicted.shape[1]):
                    h = np.histogram(predicted[...,i], bins=nbins)
                    height = h[0][h[0].argmax()]
                    ax.scatter(x=mean[i]*scalar_factor,
                               y=height,
                               color=colors[i])
                    ax.errorbar(x=mean[i]*scalar_factor,
                                y=height,
                                xerr=errors[i]*scalar_factor,
                                label=f"$\\Delta {label[i]}$",
                                ls="",
                                color=colors[i])
                    
            else:
                h = np.histogram(predicted, bins=nbins)
                height = h[0][h[0].argmax()]
                ax.scatter(x=mean*scalar_factor,
                           y=height)
                ax.errorbar(x=mean*scalar_factor,
                            y=height,
                            xerr=errors*scalar_factor,
                            label="Predicted Error",
                            ls="")

        if uncertainties.any() and show_uncertainties:
            gaussian_uncertainties(ax1)

        ax1.set_title('Predicted')

        ax2.hist(truth*scalar_factor,
                 bins=nbins,
                 histtype="step", 
                 alpha=0.9,
                 label=label)
        ax2.set_title('Truth')
                
        for ax in (ax1, ax2) :
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Number of occurences')
            ax.legend(loc="upper right")

        plt.tight_layout()
        Plot.savefig(savefigdir, data, "hist", obs)
        return None

    def scatter(data=Type[TOutput],
                obs="v1", 
                axis=("x", "y"),
                units="mm",
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
        ax = {"x" : 0, "y" : 1, "z" : 2}

        plt.figure()
        plt.scatter(x=p[...,ax[axis[0]]]*scalar_factor,
                    y=p[...,ax[axis[1]]]*scalar_factor,
                    alpha=0.1, marker=".",
                    label="Predicted")
        plt.scatter(x=t[...,ax[axis[0]]]*scalar_factor,
                    y=t[...,ax[axis[1]]]*scalar_factor,
                    alpha=0.3, marker=".",
                    label="True")
        plt.legend()
        plt.xlabel(f"{axis[0]} {units}")
        plt.ylabel(f"{axis[1]} {units}")
        plt.title(f"Model {data.model_name}")
        Plot.savefig(savefigdir, data, f"scatter_{axis[0]}_{axis[1]}", obs)
        return None



if __name__ == "__main__":
    model_name = sys.argv[1]

    data = Extract(model_dir="./nntr_models",
                   model_name=model_name)
    
    Plot.histogram(data=data,
                   obs="p1",
                   xlabel="Normalized Momentum",
                   scalar_factor=1)

    Plot.histogram(data=data,
                   obs="v1",
                   xlabel="Vertex Position [mm]",
                   scalar_factor=1E3,
                   nbins=np.linspace(-2000, 100, 100))

    Plot.scatter(data=data,
                 obs="v1",
                 scalar_factor=1E3)

    Plot.scatter(data,
                 "v1",
                 scalar_factor=1E3,
                 axis=("z", "x"))
