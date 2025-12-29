#!/usr/bin/env python
# coding: utf-8


'''
    Purpose: 
    - earlier, we divided the creation of KDTrees into batches. We have separate dictionaries of KDTrees.
    -  The purpose of this script is to generate a combined dictionary that contains a combined KDTree under
    each key.
    - assuming the pkl files are gzipped
'''

# load in prediciton df 

#imports
#print('importing modules')
import numpy as np
import pandas as pd
import logging
from sklearn.neighbors import KDTree
from collections import Counter
import pickle
import os
import time
print('finished importing modules')


import pickle
from sklearn.neighbors import KDTree
import numpy as np
import time
import gzip

base_path_to_kdtree = "/projectnb/peaclab-mon/JLi/projectx/AutomaticECGDiagnosis/KDTree_all_training/"

n_batch = 68
all_data_points = []
combined_kdtrees = {}

for i in range(1,n_batch+1):
    # make path
    batch_path = f"{base_path_to_kdtree}full_trainingset_kdtree_alldat_chunk{i}.pkl.gz"
    print(f"batch {i}")
    with gzip.open(batch_path, 'rb') as f:
        dict_kdtree = pickle.load(f)
        # change default dict to reg dict
        regular_dict = dict(dict_kdtree)

        # iterate through dictionary and append datapoints to a combined dictionary
        for key, kdtree in regular_dict.items():
            print(key)
            print(kdtree.data.shape)
            # iterate through all parts of dicationary
            if key not in combined_kdtrees:
                # If this key doesn't exist in combined_kdtrees, initialize an empty list for data points
                combined_kdtrees[key] = []
            # Append the data from the current KDTree to the list of data points for this key
            combined_kdtrees[key].append(kdtree.data)

# turn the data in each key into a KDTree
print('combining data in new dictionary')
for key, data_list in combined_kdtrees.items():
    print(f'key: {key}')
    print(f"data_list length: {len(data_list)}")
    print('vertically stacking data')
    # combined_data = np.vstack(data_list)  # Stack the data points vertically
    combined_data = np.concatenate(data_list, axis=0)
    print('Converting into KDTree')
    combined_kdtrees[key] = KDTree(combined_data)

# Validation: Iterate over each key-value pair in combined_kdtrees
for key, kdtree in combined_kdtrees.items():
    print(f"Key: {key}")
    print("Data points in the KDTree:")
    print(kdtree.data.shape)  # Print the data points stored in the KDTree
    print("-" * 50)  # Print a separator line for clarity


# save off
# Save the combined dictionary to a file
output_path = "/projectnb/peaclab-mon/JLi/projectx/AutomaticECGDiagnosis/KDTree_all_training/combined_kdtrees.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(combined_kdtrees, f)

print(f"Combined KDTree dictionary saved to {output_path}")