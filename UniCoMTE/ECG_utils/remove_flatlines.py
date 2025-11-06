#!/usr/bin/env python
# coding: utf-8

'''
    Purpose: 
     - remove flatlined signals from a KDTree
     - this ensures that CoMTE distractors that are not flatlined
'''

# imports
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,"/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/ECG_Visualization/ecg_plot_counterfactual")
import ecg_plot_counterfactual 
import pickle


# load in KDTree + inspect
kd_tree_path = "/projectnb/peaclab-mon/JLi/projectx/AutomaticECGDiagnosis/KDTree_all_training/combined_kdtrees.pkl"

with open(kd_tree_path, 'rb') as f:
    print('loading kdtree from pkl')
    combined_kdtrees = pickle.load(f)

for key, kdtree in combined_kdtrees.items():
    print(f"Key: {key}")
    print("Data points in the KDTree:")
    print(kdtree.data.shape)  # Print the data points stored in the KDTree
    print("-" * 50)  # Print a separator line for clarity




# filter with vectorized method
from scipy.spatial import KDTree

samples = combined_kdtrees[0].data
print(type(samples))

batch_size = 280394

batch = np.array(samples[:batch_size])
print(batch.shape)

std_devs = np.std(batch, axis=1)
print(std_devs.shape)

threshold = 0.1  # Adjust as needed
mask = std_devs >= threshold
# print(mask)
print(mask.shape)

# Filter out low-std-dev (flatline) samples
filtered_samples = batch[mask]
print(filtered_samples.shape)

# log indices
removed_indices = np.where(~mask)[0]
kept_indices = np.where(mask)[0]

# print(removed_indices)
# print(kept_indices)

# turn into kdTree
# new_tree = KDTree(filtered_samples)



# save off
import pickle
# save off samples that were not filtered
file_path = "/projectnb/peaclab-mon/JLi/projectx/AutomaticECGDiagnosis/KDTree_all_training/filtered_data.npy"
np.save(file_path, filtered_samples)
# with open(file_path, 'wb') as file:
#     pickle.dump(new_tree, file)
# save off samples that were filtered
file_path_array = "/projectnb/peaclab-mon/JLi/projectx/AutomaticECGDiagnosis/KDTree_all_training/removed_indices.npy"
np.save(file_path_array, removed_indices)


