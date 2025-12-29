#!/usr/bin/env python
# coding: utf-8


'''
    Purpose: 
    - Script for creating KD tree for all samples in training dataset
    - Thus, when we want to run CoMTE with all the ECG Samples, we can just load the KDTree, which saves time

    Original variables that are loaded in (no preprocessing)
    - y_pred: probabilities from inference (np array)
    - y_true: one-hot encoded labels (pd array) 
    - x_train: Nx4096x12 array of tracings (np array)
    
    Structure: 
    - make appropariate converions for all variables 
        - y_pred and y_true conversion to Nx1 list of labels
        - x_train converted to pd multiindex array

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

# os.environ['TF_NUM_INTRAOP_THREADS'] = '1' #set to 1
# os.environ['TF_NUM_INTEROP_THREADS'] = '3' #set to 1 less than # of requested cores
# print(f"TF_NUM_INTRAOP_THREADS is {os.getenv('TF_NUM_INTRAOP_THREADS')}")
# print(f"TF_NUM_INTEROP_THREADS is {os.getenv('TF_NUM_INTEROP_THREADS')}")
############################################

# modified functions for getting true positives and making kd trees
class Constructing_Trees:

    # y_pred is a list
    # y_true is a pd dataframe
    # x_train is a pd multiindex array
    def __init__(self, y_pred, y_true,x_train,silent=False):
        from collections import defaultdict

        self.y_pred = y_pred
        self.y_true = y_true
        self.x_train = x_train
        self.silent = silent
        self.classes = [0,1,2,3,4,5,6]
        #self.per_class_trees = None   
        # self.per_class_node_indices = None
        self.per_class_node_indices = defaultdict(list)
        self.per_class_trees = defaultdict(list)

    def construct_per_class_trees(self):
        """Used to choose distractors"""

        # if self.per_class_trees is not None:
        #     for c, tree in self.per_class_trees.items():
        #         num_indices = len(tree.data)  # The number of points in the KDTree
        #         print(f"Class {c} has {num_indices} indices.")
        #     return

        from collections import Counter
        #checking preds ...
        print('Validate Predictions')
        counter = Counter(self.y_pred)
        # Print unique items and their frequencies
        for item, freq in counter.items():
            print(f"{item}: {freq}")

        true_positive_node_ids = {c: [] for c in self.classes}
        
        print('printing types for y_pred and y_true')
        print(type(y_pred))
        print(type(y_true))

        for pred, (idx, row) in zip(self.y_pred, self.y_true.iterrows()):
            # print to check
            # print(idx)
            # print(type(row))
            # print(f'pred is {pred}')
            # print(f'row is {row}')
            
            if isinstance(row['label'], tuple):  # skip tuples for now - not sure how to handle them in MLRose Optimization
                continue
            if row['label'] == pred:
                if isinstance(idx, int): # wrap single datapoints in array
                    idx = [idx]
                true_positive_node_ids[pred].append(idx[0])
        
        # validation
        print("\nTrue Positive Dictionary Stats")
        for key, value in true_positive_node_ids.items():
            print(f"Key: {key}, Length of list: {len(value)}")
        
        
        print('making per class trees')

        print(f"shape of self.timeseries is {self.x_train.shape}")
        print(f"type of self.timeseries is {type(self.x_train)}")

        start_time = time.time()
        for c in self.classes:
            dataset = []
            for node_id in true_positive_node_ids[c]:
                # The below syntax of timeseries.loc[[node_id], :, :] is extremely fragile. The first two ranges index into the multi-index
                # while the third range indexes the columns. But anything other than ":" for the third range causes the code to crash, apparently
                # due to ambiguity. See the Warning here: https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#using-slicers
                try:
                    sliced_node = self.x_train.loc[[node_id], :, :]
                except pd.errors.IndexingError: # try slicing with fallback
                    sliced_node = self.x_train.loc[[node_id], :]
                
                print(f"shape of sliced node is {sliced_node.shape}")
                print(f"type of sliced node is {type(sliced_node)}")
                
                dataset.append(sliced_node.values.T.flatten())
                self.per_class_node_indices[c].append(node_id)
            if dataset:
                # print(dataset.shape)
                self.per_class_trees[c] = KDTree(np.stack(dataset))
        end_time = time.time()
        print(f"elapsed time = {end_time - start_time}")
        if not self.silent:
            logging.info("Finished constructing per class kdtree")

        return self.per_class_trees

# define functions
def ECG_one_d_labels(model_predictions, onehot_labels = True):
    '''
    
    Purpose: turn one-hot encoding (N,6) array into (Nx1) vector of classes

    Input: 
    model_predictions: 2D array of probabilities or one-hot encodings (827x6)
    onehot_labels: boolean variable 

    Output: 
    (Nx1) vector of classes

    Comments: 
    The sample class is the class that exceeds the threshold
    If there are >1 classes that exceed the threshold, a tuple will be used to store the multiple classes 
    '''
    
    if not onehot_labels:
        # establish threshold
        threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
        # generate class 0 probability
        exceedances = 1 - (np.maximum((model_predictions - threshold) , 0) / (1 - threshold))
        normal_prob = np.mean(exceedances, axis = 1, keepdims = True) # normal prob should be (N,1)
        # add normal prob
        probability_n = np.column_stack((normal_prob, model_predictions))
        # new threshold
        new_threshold = np.array([1, 0.124, 0.07, 0.05, 0.278, 0.390, 0.174])

        # make mask
        mask = probability_n >= new_threshold
    else:
        print(model_predictions.shape)

        mask = model_predictions == 1

        # Ensure each row has at least one '1'
        # no_positive_class is a column vector
        # Find rows with all False (no '1') # rows with all false becomes true
        no_positive_class = ~mask.any(axis=1) 
        
        # Expand mask by adding a new first column of zeros
        mask = np.column_stack((no_positive_class, mask))
    
    sample_classes = []
    for row in mask:
        passing_indices = np.where(row)[0]
        if len(passing_indices) > 1:  # If more than one indices pass
            if not onehot_labels: 
                # calc exceedances    
                exceedances = row - new_threshold
                # Get class with the highest exceedance
                max_class = np.argmax(exceedances)
                sample_classes.append(max_class)
            else:
                sample_classes.append(tuple(sorted(passing_indices)))  # Ensure passing indices are sorted in ascending order
        elif len(passing_indices) == 0:  # no passes
            sample_classes.append(0) 
        else:
            sample_classes.append(passing_indices[0])  
    return sample_classes

############################################

# define paths
path_to_y_predicted_proba = "/projectnb/peaclab-mon/JLi/projectx/AutoECGDiagnosisData/full_trainingset_outputs.npy"
path_to_y_true = "/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/all_parts_GT.csv"
path_to_x_train = "/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/all_parts_hdf_tracings.npy"
path_to_kd_tree_save = "/projectnb/peaclab-mon/JLi/projectx/AutomaticECGDiagnosis/KDTree_all_training/full_trainingset_kdtree_alldat.pkl"
print('Starting to load in datasets')

# load in data
x_train = np.load(path_to_x_train, allow_pickle=True)  # allow_pickle=True if objects are stored"
y_pred_probas = np.load(path_to_y_predicted_proba)
y_true_onehot = pd.read_csv(path_to_y_true)
y_true_onehot = y_true_onehot[['1dAVb','RBBB', 'LBBB', 'SB', 'AF', 'ST']]
print(x_train.shape)
print('done loading in datasets')

# create y_true and y_pred...
# y_pred = ECG_one_d_labels(y_pred_probas[:10, :], onehot_labels=False)
# y_true = ECG_one_d_labels(y_true_onehot.iloc[:10,:], onehot_labels=True)
y_pred = ECG_one_d_labels(y_pred_probas[:,:], onehot_labels=False)
y_true = ECG_one_d_labels(y_true_onehot.iloc[:,:], onehot_labels=True)
# convert y_true into a pd dataframe with col um title 'labels'
y_true = pd.DataFrame(y_true, columns=['label'])
print(len(y_pred))
print(len(y_true))
print(type(y_pred))
print(type(y_true))

# wrap the training dataset...
import sys
sys.path.append('/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/CoMTE_V2/comlex_core/src')  # Path to the comlex_core directory
# import project (wrapper) modules
from explainable_data_ECG import ClfData as ClfData


# if we do not have enough space, process in batches 
# Assume x_train is a large 3D numpy array: (samples, height, width)
num_chunks = 68 # 5000 samples per batch
chunk_size = len(x_train) // num_chunks
dfs = []
print('wrapping training points in chunks')
for i in range(num_chunks):
    print(f"Chunk {i}")
    start = i * chunk_size
    # Ensure the last chunk includes the     remainder
    end = (i + 1) * chunk_size if i < num_chunks - 1 else len(x_train)
    
    # define chunk
    x_chunk = x_train[start:end]
    y_pred_chunk = y_pred[start:end]
    y_true_chunk = y_true[start:end]

    # wrap df
    print(f'Processing chunk {i+1}/{num_chunks}: indices {start}:{end}')
    df_chunk = ClfData.wrap_df_x(x_chunk, 12, start)
    # dfs.append(df_chunk)

    # make KDTree
    print('making KDTree with batch')
    KDTree_creator = Constructing_Trees(y_pred_chunk, y_true_chunk, df_chunk, silent=False)
    KD_tree_chunk = KDTree_creator.construct_per_class_trees()
    print('KD Tree Creation Finished')

    # print structre for validation
    print('KD Tree Structure:')
    for label, tree in KD_tree_chunk.items():
        print(f"\nKD Tree for class: {label}")
        print(f"Tree data shape: {tree.data.shape}")  # Data shape

    # save off
    chunk_save_path = f"{path_to_kd_tree_save.rstrip('.pkl')}_chunk{i+1}.pkl"
    directory = os.path.dirname(chunk_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(chunk_save_path, 'wb') as file:
        pickle.dump(KD_tree_chunk, file)
    print(f"KDTree for chunk {i+1} saved to {chunk_save_path}")
    







# Concatenate all chunks into a single DataFrame
# df_train_points = pd.concat(dfs)
# print('finished wrapping training points')

# # initialize class for creating kdtree
# KDTree_creator = Constructing_Trees(y_pred,y_true,df_train_points,silent=False)
# # get KD Trees with modified class
# print('constructing KD trees')
# KD_tree = KDTree_creator.construct_per_class_trees()
# print('KD Tree Creation Finished')

# # print for validation
# print('KD Tree Structure:')
# for label, tree in KD_tree.items():
#     print(f"\nKD Tree for class: {label}")
#     print(f"Tree data shape: {tree.data.shape}")  # Data shape

# # Save the KDTree for this chunk
    
# # save off KD_tree
# directory = os.path.dirname(path_to_kd_tree_save)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# with open(path_to_kd_tree_save, 'wb') as file:
#     pickle.dump(KD_tree, file)

# print("KDTree saved successfully with pickle!")