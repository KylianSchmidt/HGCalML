import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import uncertainties as unc
from uncertainties import unumpy
from plotting import PlottingWrapper as pw
plt.style.use("belle2")


ta_distance = [30.0, 50.0, 80.0, 100.0, 150.0]
num_testing_events = np.array([23465, 24498, 22770, 21553, 18810])
num_training_events = np.array([130633, 124211, 114129, 102029, 89326])
ta_std = [186.8, 159.77, 163.53, 153.34, 173.2]
tad_res = unumpy.uarray(
    [171.7, 142.6, 150.0, 139.7, 157.1],
    [1.1, 0.9, 1.0, 1.0, 1.1]
)

#yerr = np.array([ta_std[i] for i, j in enumerate(ta_std)])
correction_factor = 1 #/np.sqrt(num_training_events)
plt.errorbar(
    x=ta_distance,
    y=unumpy.nominal_values(tad_res)*correction_factor,
    yerr=unumpy.std_devs(tad_res)*correction_factor,
    linestyle="",
    marker="+",
    )
plt.ylabel("Resolution [mm]")
plt.xlabel("TA distance [mm]")
plt.legend()
plt.savefig("./ta_distance")

plt.plot(
    ta_distance,
    num_training_events
)
plt.xlabel("TA distance [mm]")
plt.ylabel("Number of training events")
plt.savefig("./num_training_events")
