#!/usr/bin/env python
# coding: utf-8

'''
    Question: How many instances does an explanation cover? Is one explanation applicable to many test cases?

    Data:
    Use ECG Time series data
    Use existing pipeline and existing explainability framework


    Process: 
     1. Moidify full training KDTree to exclude the first 3000 samples
     2. Run generalizability algorithm
           For each explanation, perform the feature substitution to every other sample that matches (true class, mispredicted class)
           Record how many other samples of the same class flip their prediction
           Report the coverage
'''

import copy
import gc
import itertools
import logging
from multiprocessing import Pool
import functools
import sys 
import random
from pathlib import Path
import time
import mlrose_ky as mlrose
#import mlrose
import h5py

from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

import ecg_analysis.data 
import ecg_analysis.classifier
import explainers_benchmarking as explainers

import datasets

from explainable_model_ECG import ClfModel as ClfModel
from explainable_data_ECG import ClfData as ClfData

import pickle

def ECG_one_d_labels(model_predictions, onehot_labels = True):
    '''
    
    Purpose: turn one-hot encoding (N,d) array into (Nx1) vector of classes

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
    
def batch_inference(data, model, batch_size = 500):
    
    results = []
    
    for i in range(0,data.shape[0], batch_size):
        print(f'working on indicies {i} to {i+batch_size}')
        batch = data[i:i+batch_size]
        batch_results = model.predict(batch, verbose = 1)
        results.append(batch_results)
    
    print(len(results))
    
    return np.concatenate(results)

def test_batch_vs_full_inference(data, model, batch_size=500):
    full_results = model.predict(data)
    batched_results = batch_inference(data, model, batch_size=batch_size)
    
    np.testing.assert_allclose(full_results, batched_results, rtol=1e-5)
    print("Test passed: full and batched inference match!")

########################### Part 1: Create Combined Datasets (y_pred, y_true) ##############################

# load x_test
path_to_hdf5_test = "/projectnb/peaclab-mon/JLi/projectx/AutoECGDiagnosisData/CODE/ecg_tracings.hdf5"
dataset_name_test = "tracings" 
with h5py.File(path_to_hdf5_test, "r") as f:
    x_test = np.array(f['tracings'])
    print(x_test.shape)
    
# load x_train
num_train_samples_for_test = 3000
code15_all_tracings_path = "/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/all_parts_hdf_tracings.npy"
tracings = np.load(code15_all_tracings_path, allow_pickle=True)  # allow_pickle=True if objects are stored
x_train_subset = tracings[:num_train_samples_for_test,:,:]
print(x_train_subset.shape)
print('printing shapess')
print(x_train_subset.shape)
print(x_test.shape) 
# create full testing set
x_test_combined = np.concatenate((x_train_subset, x_test), axis=0)
print(x_test_combined.shape)


# load pretrained model (still need to compile later) 
model_path = "/projectnb/peaclab-mon/JLi/projectx/AutoECGDiagnosisData/PretrainedModels/model/model.hdf5"
pre_model = load_model(model_path)  
# compile and apply model to testing dataset
pre_model.compile(loss='binary_crossentropy', optimizer=Adam())
model_predictions = pre_model.predict(x_test_combined,verbose=1)   # y_score is a numpy array with dimensions 827x6. It holds the predictions generated by the model

# Generate dataframe
np.save("/projectnb/peaclab-mon/JLi/projectx/AutoECGDiagnosisData/dnn_output_generalizability.npy", model_predictions)
print("Output predictions saved")

# make y_pred
y_pred = ECG_one_d_labels(model_predictions, onehot_labels = False)
print(len(y_pred))

# make y_true
# load data
y_true_training_2D = pd.read_csv('/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/all_parts_GT.csv')
y_true_testing_2D = pd.read_csv('/projectnb/peaclab-mon/JLi/projectx/AutoECGDiagnosisData/CODE/annotations/gold_standard.csv').values
# convert 2D into 1D
y_true_training = ECG_one_d_labels(y_true_training_2D.iloc[:num_train_samples_for_test, 2:], onehot_labels=True)
y_true_testing = ECG_one_d_labels(y_true_testing_2D, onehot_labels=True)

# concatinate (training before testing)
print(type(y_true_training))
print(type(y_true_testing))
y_true = y_true_training + y_true_testing


for idx,i in enumerate(y_pred):
    print(f"y_pred is {y_pred[idx]}, y_true is {y_true[idx]}")


# evaluaton via confusion matrix
# make confusion matrix for test (ignore tuples)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Filter out indices where either y_true or y_pred is a tuple
filtered_y_true = []
filtered_y_pred = []

for true, pred in zip(y_true, y_pred):
    if not isinstance(true, tuple) and not isinstance(pred, tuple):
        filtered_y_true.append(true)
        filtered_y_pred.append(pred)


############################## Part 2: KDTree Creation and Modification #########################

# load in KDTree + inspect
import pickle

kd_tree_path = "/projectnb/peaclab-mon/JLi/projectx/AutomaticECGDiagnosis/KDTree_all_training/combined_kdtrees.pkl"

with open(kd_tree_path, 'rb') as f:
    print('loading kdtree from pkl')
    combined_kdtrees = pickle.load(f)

for key, kdtree in combined_kdtrees.items():
    print(f"Key: {key}")
    print("Data points in the KDTree:")
    print(kdtree.data.shape)  # Print the data points stored in the KDTree
    print("-" * 50)  # Print a separator line for clarity


# run inference to understand where the first N true positives went
from collections import defaultdict

model_pred_train_samples_for_test = pre_model.predict(x_train_subset,verbose=1)   # y_score is a numpy array with dimensions 827x6. It holds the predictions generated by the model
y_pred_train = ECG_one_d_labels(model_pred_train_samples_for_test, onehot_labels=False)
y_true_train = ECG_one_d_labels(y_true_training_2D.iloc[:num_train_samples_for_test, 2:], onehot_labels=True)
y_true_train = pd.DataFrame(y_true_train, columns=['label'])

true_positive_node_ids = defaultdict(list)

# make dictionary of true positives
for pred, (idx, row) in zip(y_pred_train, y_true_train.iterrows()):
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

# for each KDTree, remove the first few samples 
#number of samples to remove equates to the number of samples that are included in the first N true positives as part of the class

import time
from scipy.spatial import KDTree

for idx, item in combined_kdtrees.items(): 
    print(f'for class{idx}')
    num_samples = len(true_positive_node_ids[idx])
    print(f'deleting first {num_samples} samples')
    
    start_time = time.time()
    
    print('removing datapoints')
    # remove X datapoints
    data = item.data
    new_data = data[num_samples:]
    end_time_1 = time.time()
    print(f"time: {end_time_1 - start_time}")
    
    print('building KDTree')
    # rebuild KDTree
    newKD = KDTree(new_data)
    end_time_2 = time.time()
    print(f"time: {end_time_2 - end_time_1}")
    
    # replace entry in the dictionary
    combined_kdtrees[idx] = newKD

# print for validation
for key, kdtree in combined_kdtrees.items():
    print(f"Key: {key}")
    print("Data points in the KDTree:")
    print(kdtree.data.shape)  # Print the data points stored in the KDTree
    print("-" * 50)  # Print a separator line for clarity




####################### Part 3: Init CoMTE_V2 wrappers ########################

import sys
sys.path.append('/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/CoMTE_V2/comlex_core/src')  # Path to the comlex_core directory
# import project (wrapper) modules
import explainers_input_kd as explainers_V2

# load in train and test sets, define key variables
class BasicData:
    # define basic variables
    classes_available = [0,1,2,3,4,5,6]
    num_columns = 4096
    num_features = 12

    # define new train and test data (should be temporary)
    num_train_samples_for_comte = 20000

    # need to make new training dataset
    x_train_comte = tracings[num_train_samples_for_test:num_train_samples_for_test+num_train_samples_for_comte,:,:]
    y_train_comte = ECG_one_d_labels(y_true_training_2D.iloc[num_train_samples_for_test:num_train_samples_for_test+num_train_samples_for_comte, 2:], onehot_labels=True)

    # iterable of corresponding labels for the samples for the data wrapper (returns 20000x6 np array) <--- take out first column that represents ExamID
    # labels = pd.read_csv(y_train_csv_path)

"""
Part 1: A Classifier that works with COMLEX

The classifier must have 2 capabilities:
1. Predict a class ie: class 0 in classes {0, 1}
2. Predict the probability for each class
-ie: [0.1, 0.9]

and

Be able to execute capability 1 and 2 on a PANDAS dataframe,
returning an array of corresponding predictions.



input:
    samples to be classified (pandas multiindex dataframe)

output: 
    for contrived_classification: length N list of classes

    for contrived_classification_proba: 
            length N list of 1x7 np arrays
"""

class BasicClassifier:
    classifier = pre_model  # tensorflow CNN
    import os
    
    @staticmethod
    def contrived_classification(pandas_dfs):
        import os
        classifier = pre_model  # tensorflow CNN

        # convert 2D pandas df to 3D dataframe (N,4096,12)
        array_3d = pandas_dfs.to_numpy().reshape(int(pandas_dfs.shape[0]/4096), 4096, 12)

        # create instance of ECGSequence to store the (N,4096,12) dataset
        temp_path = "/projectnb/peaclab-mon/JLi/projectx/AutoECGDiagnosisData/temporary.hdf5"
        temp_dataset_name = "tracings"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # create hdf with appropriate data
        hdf_file = h5py.File(temp_path, 'w')
        hdf_file.create_dataset(temp_dataset_name,data = array_3d)
        # init instnace of ECG Sequence holding modified with hdf path
        modified_instance = datasets.ECGSequence(temp_path, temp_dataset_name)

        # get classification and probability
        probability = classifier.predict(modified_instance, verbose = 0)    
        
    
        # close hdf5's
        modified_instance._closehdf()
        hdf_file.close()
        os.remove(temp_path)

        # analyze model output with thresholding
        # define given thresholds
        threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
        
        # generate class 0 probability
        exceedances = 1 - (np.maximum((probability - threshold) , 0) / (1 - threshold))
        normal_prob = np.mean(exceedances, axis = 1, keepdims = True) # normal prob should be (N,1)
        
        # Add normal_prob as a new column
        probability_n = np.column_stack((normal_prob, probability))     

        # new threshold
        new_threshold = np.array([1, 0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
        
        mask = probability_n >= new_threshold
        sample_classes = []  # init list for appends later
        
        for row, mask in zip(probability_n, mask):
            passing_indices = np.where(mask)[0]
            if len(passing_indices) > 1:  # If more than one indices pass
                # find margin between threshold and probability
                diff_array = row - new_threshold
                passing_index = np.argmax(diff_array)
                # append the index that has the highest margin
                sample_classes.append(passing_index)
            
            elif len(passing_indices) == 0:  # no passes
                sample_classes.append(0) 
            else:
                sample_classes.append(passing_indices[0])  # Select the first (or adjust logic)
                
        return sample_classes


    @staticmethod
    def contrived_classification_proba(pandas_dfs):
        import os
        classifier = pre_model  # tensorflow CNN
        
        # convert 2D pandas df to 3D dataframe (N,4096,12)
        array_3d = pandas_dfs.to_numpy().reshape(int(pandas_dfs.shape[0]/4096), 4096, 12)

        # create instance of ECGSequence to store the (N,4096,12) dataset
        temp_path = "/projectnb/peaclab-mon/JLi/projectx/AutoECGDiagnosisData/temporary.hdf5"
        temp_dataset_name = "tracings"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # create hdf with appropriate data
        hdf_file = h5py.File(temp_path, 'w')
        hdf_file.create_dataset(temp_dataset_name,data = array_3d)
        # init instnace of ECG Sequence holding modified with hdf path
        modified_instance = datasets.ECGSequence(temp_path, temp_dataset_name)

        # get classification and probability
        probability = classifier.predict(modified_instance, verbose = 0)  
        
        # close hdf5's
        modified_instance._closehdf()
        hdf_file.close()
        os.remove(temp_path)

        # analyze model output with thresholding
         # define given thresholds
        threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
        
        # generate class 0 probability
        exceedances = 1 - (np.maximum((probability - threshold) , 0) / (1 - threshold))
        normal_prob = np.mean(exceedances)

        # modify result 
        probability = np.insert(probability,0,normal_prob)   

        # probability should be in a 2D array format
        if probability.ndim == 1:  # Check if it's 1D
            probability = probability.reshape(1, -1)
        
        return probability

"""
Part 3: Wrapping it up.

The training data, training labels, and trained classifier need to be wrapped up
into a form that can pass through COMLEX.

While wrapping up the training data and labels is relatively straightforward,
wrapping up the classifier is more difficult

Pipeline:
input: Raw ECG data (pandas multiindex array). 
output: classification from random forest model (class number)

Note: Ensure inputs/outputs are identical with the input/output for CoMTE classification algorithm, so we can just use this pipeline as a direct replacement

"""

class BasicComlexInput:

    # 1. wrap training points
    df_train_points = ClfData.wrap_df_x(BasicData.x_train_comte, BasicData.num_features)
    
    # 2. wrap training labels
    dummy_train = pd.Series([(i, i) for i in range(len(BasicData.y_train_comte))], name="dummy")
    df_train_labels = pd.DataFrame(BasicData.y_train_comte, index=dummy_train, columns=["label"])
    

    # 3. wrap up the classifier
    # note: column_attr, or the corresponding name of the columns in the sample,
    #  is unique to dataframes, and auto-generated by wrap_df_x
    wrapped_classifier = ClfModel(BasicClassifier.classifier,
                                predict_attr=BasicClassifier.contrived_classification,
                                predict_proba_attr=BasicClassifier.contrived_classification_proba,
                                column_attr=df_train_points.columns.values.tolist(),
                                classes_attr=BasicData.classes_available,
                                window_size_attr=BasicData.num_columns)    

comlex = explainers_V2.OptimizedSearch(BasicComlexInput.wrapped_classifier,
                                    BasicComlexInput.df_train_points,
                                    BasicComlexInput.df_train_labels,
                                    combined_kdtrees,
                                    silent=True, threads=4, num_distractors=2)

def dict_to_pickle(my_dict,filename):

    dir = "/projectnb/peaclab-mon/JLi/projectx/CoMTE_V2_JLi/LIME_SHAP_Comparison_Experiments/"
    with open(dir+filename+'.pkl', 'wb') as f:
        print(dir+filename+'.pkl')
        pickle.dump(my_dict,f)
 
def run_generality(pipeine, x_test, y_test, y_pred, true_label, pred_label,picklename):

    '''
    input requirements: 
        pipeline = pipeline that returns probability when .predict_probas and class when .predict
        x_test = testing dataset (3D NP array)
        y_test = testing labels (pandas df, column titled 'label')
        y_pred = model predictions on testing dataset (list)
        true_label = true class 
        pred_label = class predicted instead of true class (misclassification)
        picklename = name of .pkl file to store
    '''
    
    #l = list(enumerate(zip(y_test['label'], y_pred)))

    l = [
    (i, (true, pred)) 
    for i, (true, pred) in enumerate(zip(y_test['label'], y_pred))
    if not isinstance(true, tuple) and not isinstance(pred, tuple)
    ]

    print("l is : ")
    print(l)
    
    exp_set = {}

    # create set of misclassified samaples satisfying true_label and pred_label
    print('creating set of misclassified samples')
    print(f'picklename is {picklename}')
    indices = []
    for idx, (true, pred) in l:  
        print(f"true: {true}")  
        print(f"true: {pred}")  
        print(f"true: {true_label}")  
        print(f"true: {pred_label}")  
        if true != pred and true == true_label and pred == pred_label:
            x_sample = np.expand_dims(x_test[idx, :, :], axis=0)
            exp_set[idx] = ClfData.wrap_df_test_point(x_sample) 
            indices.append(idx)
    
    print(f"appended indices: {indices}")
    print(f"number of indices: {len(indices)}")
          
    
    logging.info("Explanation set length %d",len(exp_set))
    
    # init 
    columns = exp_set[indices[0]].columns
    scores = []


    for idx, x_test in exp_set.items():
        test_sample_index = idx
        # get distractor
        explanation = comlex.explain(x_test,to_maximize=true_label,
                             return_dist=True,single=True,
                             savefig=True,train_iter=100,
                             timeseries=False,filename="sample_result.png")

        # log explanation
        baseline_exp = explanation[0]
        best_dist = explanation[1]

        
        counter = 0
        prob_list = []
        # for every other misclassified sample, use distractor
        for idx, temp_test in exp_set.items():
            print(f"index of sample: {idx}")
            
            # temp_test and x_test should be pd multiindex arrays
            # print(temp_test.shape)
            # print(x_test.shape)
            # print(type(temp_test))
            # print(type(x_test))
            
            if not temp_test.equals(x_test): 
                print('making replacements to misclassified sample')
                # initial prob
                modified = temp_test.copy()
                prob_first = pipeline.predict_proba(modified)[0][true_label]     
                print(f"previous probability: {prob_first}")
                
                # change sample distractor
                for c in columns:
                    if c in baseline_exp:
                        print(f"replacing {c}")
                        modified[c] = best_dist[c].values
                    
                # final prob
                prob_last = pipeline.predict_proba(modified)[0][true_label]
                print(f"new probability: {prob_last}")
                prob_list.append(prob_last - prob_first)

                new_class = pipeline.predict(modified)[0]
                print(f"new class: {new_class}")
                
                if new_class == true_label:
                    print("adding to counter")
                    counter += 1    
                    
            else:  # doing nothing with the same sample
                pass

        # calc scores for each                            
        coverage = counter / (len(exp_set)-1)
        scores.append({
                'explanation': baseline_exp,
                'count': counter,
                'coverage': coverage,
                #'prob_xtest': prob_xtest,
                'prob_change': prob_list,
                'x_test_sample_index': test_sample_index,                                
                #'dist_node_id':best_dist.index.get_level_values('node_id').unique().values[0]                                
        })
        
        # logging.info("Explanation: %s",baseline_exp)
        # logging.info("Coverage: %f" ,coverage)
        # logging.info("Prob list: %s",prob_list)
        # logging.info("node id %s",scores[-1]['dist_node_id'])
    print('saving')
    dict_to_pickle(scores,picklename)
    print('done saving')
    return scores

####################### Part 4: Generalizability Run ########################

# define conditions for running
configs = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]
configs = [(0,5)]
pipeline = BasicComlexInput.wrapped_classifier
x_test = x_test_combined
dummy_train = pd.Series([(i, i) for i in range(len(y_true))], name="dummy")
y_test = pd.DataFrame(y_true, index=dummy_train, columns=["label"])
y_pred = y_pred

# run loop
for element in configs: 
    true_label = element[0]
    pred_label = element[1]
    picklename = f"ECG_Generalizability_true_{true_label}_pred_{pred_label}"

    # run generalizability
    scores = run_generality(pipeline, x_test, y_test, y_pred, true_label, pred_label, picklename)
    print(scores)
    print(len(scores))
    for i in scores:
        print("." * 50)  # prints 50 dots in a row
        print(i)

