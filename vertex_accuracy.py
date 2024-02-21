from zfitwrapper import utilities, Models, Fitter
import pandas as pd
import numpy as np
import pickle
import os, sys
import matplotlib.pyplot as plt
import uncertainties as unc
plt.style.use("belle2")


class FitDSCB():
    def dofit(array, params, limits=(-800, 800)):
        obs = utilities.set_obs("VertexAccuracy", (limits[0], limits[1]))

        utilities.clear_existing_parameters()
        model = Models.Model(obs=obs, modelstring="Model", models=params)
        fit = Fitter.Fitter(array, model, retry=20)
        fit.fit()
        return fit.model.dict, fit.model.model.pdf, fit.gof


def plot_add_info(ax: plt.Axes, text="Demonstration testing set"):
    ax.text(
        0.02, 0.94,
        "Beamdump",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold")
    ax.text(
        0.02, 0.8,
        "Photon Reconstruction 2024\n"
        + f"{text}\n",
        transform=ax.transAxes,
        fontsize=12)


params_dscb = {
    "30": {
        "Model": {
        "pdf": "DoubleCB", "parameter": {
            "mu": {"name": "mu", "value": -10, "lower": -20, "upper": 0, "floating": True},
            "sigma": {"name": "sigma", "value": 30, "lower": 0, "upper": 120, "floating": True},
            "alphal": {"name": "alphal", "value": 0.6, "lower": 0.01, "upper": 5, "floating": True},
            "alphar": {"name": "alphar", "value": 0.5, "lower": 0.01, "upper": 5, "floating": True},
            "nl": {"name": "nl", "value": 4, "lower": 1, "upper": 4, "floating": True},
            "nr": {"name": "nr", "value": 6.18, "lower": 1, "upper": 7, "floating": True}
            }
        }
    },
    "50": {
        "Model": {
        "pdf": "DoubleCB", "parameter": {
            "mu": {"name": "mu", "value": 0, "lower": -10, "upper": 20, "floating": True},
            "sigma": {"name": "sigma", "value": 100, "lower": 90, "upper": 140, "floating": True},
            "alphal": {"name": "alphal", "value": 1.13, "lower": 0.1, "upper": 2, "floating": True},
            "alphar": {"name": "alphar", "value": 0.5, "lower": 0.1, "upper": 2, "floating": True},
            "nl": {"name": "nl", "value": 13, "lower": 10, "upper": 20, "floating": True},
            "nr": {"name": "nr", "value": 6.18, "lower": 6, "upper": 20, "floating": True}
            }
        }
    },
    "80": {
        "Model": {
        "pdf": "DoubleCB", "parameter": {
            "mu": {"name": "mu", "value": 20.17, "lower": 15, "upper": 25, "floating": True},
            "sigma": {"name": "sigma", "value": 110, "lower": 90, "upper": 150, "floating": True},
            "alphal": {"name": "alphal", "value": 1.13, "lower": 0.1, "upper": 2, "floating": True},
            "alphar": {"name": "alphar", "value": 0.5, "lower": 0.1, "upper": 2, "floating": True},
            "nl": {"name": "nl", "value": 13, "lower": 10, "upper": 20, "floating": True},
            "nr": {"name": "nr", "value": 6.18, "lower": 6, "upper": 20, "floating": True}
            }
        }
    },
    "100": {
        "Model": {
        "pdf": "DoubleCB", "parameter": {
            "mu": {"name": "mu", "value": 20.17, "lower": 15, "upper": 25, "floating": True},
            "sigma": {"name": "sigma", "value": 110, "lower": 90, "upper": 160, "floating": True},
            "alphal": {"name": "alphal", "value": 1.13, "lower": 0.1, "upper": 2, "floating": True},
            "alphar": {"name": "alphar", "value": 0.5, "lower": 0.1, "upper": 2, "floating": True},
            "nl": {"name": "nl", "value": 13, "lower": 10, "upper": 20, "floating": True},
            "nr": {"name": "nr", "value": 6.18, "lower": 6, "upper": 20, "floating": True}
            }
        }
    },
    "150": {
        "Model": {
        "pdf": "DoubleCB", "parameter": {
            "mu": {"name": "mu", "value": 20.17, "lower": 15, "upper": 25, "floating": True},
            "sigma": {"name": "sigma", "value": 110, "lower": 90, "upper": 180, "floating": True},
            "alphal": {"name": "alphal", "value": 1.13, "lower": 0.1, "upper": 2, "floating": True},
            "alphar": {"name": "alphar", "value": 0.5, "lower": 0.1, "upper": 2, "floating": True},
            "nl": {"name": "nl", "value": 13, "lower": 10, "upper": 20, "floating": True},
            "nr": {"name": "nr", "value": 6.18, "lower": 6, "upper": 20, "floating": True}
            }
        }
    },

}


if __name__ == "__main__":
    ta_distance = sys.argv[1]
    region = ""

    data = pd.read_csv(
        f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracy{region}.csv").to_numpy()[:, 1]

    fitdict, pdf, gof = FitDSCB.dofit(data, params=params_dscb[ta_distance])
    with open(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/VertexAccuracyFitDict", "wb") as file:
        pickle.dump(fitdict, file)

    print(fitdict)

    # Plotting

    limits = (-800, 800)
    bins = np.linspace(limits[0], limits[1], 64+1)
    x = np.linspace(limits[0], limits[1], 2000)
    y = pdf(x)
    params = fitdict["modelparameter"]["Model"]["parameter"]

    parameter = {}
    fit_parameter_names_dscb = ["mu", "sigma", "alphal", "alphar", "nl", "nr"]
    fit_parameter_names_cauchy = ["m", "gamma"]

    for key in fit_parameter_names_dscb:
        parameter[key] = unc.ufloat(
            params[key]["value"],
            abs(params[key]["value"] -
                max(abs(params[key]["lower"]), params[key]["upper"]))
        )

    def FWHM(x: np.array, y: np.array):
        def find_nearest(array, value):
            array = np.asarray(array)
            index = (np.abs(array - value)).argmin()
            return index

        max = find_nearest(y, np.max(y))
        fwhmheight = np.max(y) / 2
        x1 = x[find_nearest(y[:max], fwhmheight)]
        x2 = x[max + find_nearest(y[max:], fwhmheight)]
        fwhm = x2 - x1
        return fwhm, fwhmheight, x1, x2

    fig, ax = plt.subplots()
    plot_add_info(ax, f"TA distance : {ta_distance} mm")
    # Histogram
    hist_height, hist_bins = np.histogram(data, bins)
    bin_centers = 0.5*(hist_bins[1:] + hist_bins[:-1])
    plt.plot(
        bin_centers,
        hist_height,
        marker=None,
        drawstyle="steps-mid",
        label="MC Data",
        color="black",
    )
    ax.bar(
        bin_centers,
        height=2*hist_height**0.5,
        width=hist_bins[1]-hist_bins[0],
        bottom=hist_height - hist_height**0.5,
        color="black",
        hatch="///////",
        fill=False,
        lw=0,
        label="Stat. unc.",
            )
    # Density
    counts, bins_fit = np.histogram(data, x)
    density = np.sum(counts)*np.diff(bins_fit)[0]/len(bins)*len(x)
    # FWHM
    fwhm, fwhm_height, x1, x2 = FWHM(x, y*density)
    plt.hlines(fwhm_height, x1, x2, color="red", label=f"FWHM={fwhm:.2f} mm")
    # Fit
    label = "DSCB Fit Parameters:\n"
    for key in parameter.keys():
        label = label+f" {key}={parameter[key]:.2f}\n"
    label = label.rstrip("\n")
    plt.plot(
        x,
        y*density,
        label=label
    )
    print("Goodness of fit:", gof)
    plt.xlim(bins[0], bins[-1])
    plt.ylim(0, )
    plt.xlabel(r"$V_{1,z}^{true} - V_{1,z}^{pred}$"+"[mm]")
    plt.ylabel(f"Counts / ({(bins[1]-bins[0]):.1f} [mm])")

    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Plots/vertex_accuracy")
    plt.show()
