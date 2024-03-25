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
        + f"{text}",
        transform=ax.transAxes,
        fontsize=12)


params_dscb = {
    "30": {
        "Model": {
        "pdf": "DoubleCB", "parameter": {
            "mu": {"name": "mu", "value": -2, "lower": -5, "upper": 0, "floating": True},
            "sigma": {"name": "sigma", "value": 5, "lower": 0.2, "upper": 10, "floating": True},
            "alphal": {"name": "alphal", "value": 0.6, "lower": 0.1, "upper": 2, "floating": True},
            "alphar": {"name": "alphar", "value": 0.5, "lower": 0.1, "upper": 2, "floating": True},
            "nl": {"name": "nl", "value": 4, "lower": 1, "upper": 20, "floating": True},
            "nr": {"name": "nr", "value": 6.18, "lower": 1, "upper": 20, "floating": True}
            }
        }
    }
}


if __name__ == "__main__":
    ta_distance = sys.argv[1]

    data = pd.read_csv(
        f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/RadiusAccuracy.csv").to_numpy()[:, 1]

    fitdict, pdf, gof = FitDSCB.dofit(data, params=params_dscb[ta_distance])
    with open(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Predicted/RadiusAccuracyFitDict", "wb") as file:
        pickle.dump(fitdict, file)

    print(fitdict)

    # Plotting

    limits = (-20, 20)
    bins = np.linspace(limits[0], limits[1], 80+1)
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
    plot_add_info(ax, f"Demonstration testing set\n{len(data)} MC Events")
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
        label=label,
        )
    print("Goodness of fit:", gof)
    plt.xlim(bins[0], bins[-1])
    plt.xlabel(r"$r_{true} - r_{pred}$"+"[mm]")
    plt.ylabel(f"Counts / ({(bins[1]-bins[0]):.1f} [mm])")

    plt.legend(loc="upper right")
    plt.savefig(f"/work/kschmidt/DeepJetCore/TrackReco_DeepJetCore/HGCalML/nntr_models/normal_detector/ta_distance_sweep/tad_{ta_distance}/Plots/radial_accuracy")
    plt.show()
