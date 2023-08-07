import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

data = []
with open(filename, "rb") as file:
    data = pickle.load(file)

features = data["Features"][0]

def find_physical_variables(array):
    phys_dict = {
        "p1" : array[:,0:3],
        "n1" : array[:,3],
        "v1" : array[:,4:7],
        "p2" : array[:,7:10],
        "n2" : array[:10],
        "v2" : array[:,11:14],
    }
    return phys_dict

t = find_physical_variables(data["array"][0])
p = find_physical_variables(data["Predicted"][0])
