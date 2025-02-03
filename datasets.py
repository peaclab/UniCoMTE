
import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np

# define functions for making datasets

class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        
        # define key variables
        n_samples = len(pd.read_csv(path_to_csv)) # defines number of samples in 
        n_train = math.ceil(n_samples*(1-val_split)) 
        
        # print n_samples and n_train to understand dataset
        print("The number of samples in the dataset is {}".format(n_samples))
        print("The index in which the validation set starts and train set ends is {}".format(n_train))
        
        # create 2 class instances of ECG sequence for training and validation by specifying the appropriate start and end indices
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        
        print(train_seq)
        print(valid_seq)
        
        # return these two instances of the ECG sequence class
        return train_seq, valid_seq

    
    # Constructor function that sets up an instance of the ECGSequence Class
    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        
        # if a csv is provided, it loads the labels into a numpy aray
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
            
        # open the hdf5 file and accesses the specified dataset
        self.f = h5py.File(path_to_hdf5, "r")  # open the hdf5 dataset
        self.x = self.f[hdf5_dset]             # access the name specified
        #print(f"the size of the input dataset is {self.x.shape}")
        #print(type(self.f))
        
        
        
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        #self.f.close() # close hdf5 dataset
        

    
    @property
    # returns the number of classes in the dataset by checking the shape of the label array
    def n_classes(self):
        return self.y.shape[1]

    # retrieves a certain batch of data
    def __getitem__(self, idx):
        
        start = self.start_idx + idx * self.batch_size 
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    # returns the number of batches in seq
    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    # ensures the HDF file is properly closed when the instnace of ECGSequence is deleted
    def __del__(self):
        self.f.close()
    
    # retrieves data corresponding to a certain sample 
    def _getsample_(self, samplenum):
        temp = np.array(self.x[samplenum, :, :])
        return temp[None, :, :]
    
    # retrieves all the true labels from CSV
    def _gettruelabel_(self):
        return self.y
    
    def _get_one_by_N_labels_(self):  # this should return a (Nx1) array with numbers 
        # make trainset labels (20000x1)
        # Use threshold to determine class from probabilities
        threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
        # apply threshold to convert array of probabilities to array of selections (1x6)
        mask = self.y > threshold # record instances in which y_score_best > threshold
        orig_preds = np.zeros_like(self.y)     # fill array with same size as self.y with zeros
        orig_preds[mask] = 1                            # set certain values (defined by mask) to 1

        # convert (Nx6) --> (Nx1) to get the class number
        y_pred_number_label = []
        for i in range(self.y.shape[0]):
            one_present = 0
            for j in range(self.y.shape[1]):   # for each row, iterate through columns
                if self.y[i, j] == 1:
                    y_pred_number_label.append(j + 1)
                    one_present = 1
                    break
            if one_present == 0:   # after each row, check if a 1 has been assigned
                y_pred_number_label.append(0)
            
        return y_pred_number_label
    
    # returns all the timeseires data (N,4096,12) 
    def _gettimeseries_(self):
        # convert to np array
        timeseries = np.array(self.x)
        return timeseries
    
    # returns the list of metrics (features)
    def _listofmetrics_(self):
        # add names
        columns = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
        return list(columns)
    

    def _getmultiindexarray_(self): 
        # returns multiindex array

        # turn timeseries from 3D dataframe to 2D pandas multi level dataframe for CoMTE algorithms (may need to put this into ECG Sequence function later)
        # make 3D numpy array
        timeseries = np.array(self.x)
        
        # find the number of samples
        #print(timeseries.shape[0])

        #reshape data for easier conversion to DF. flatten last dimension into columns (attributes)
        reshaped_data = timeseries.reshape((-1, timeseries.shape[-1]))
        
       
        # Example index creation for the DataFrame
        # Creating a MultiIndex with two levels: 'node_id' and 'timestamp'
        index_num = np.repeat(np.arange(timeseries.shape[0]), 4096)  # Repeat node_id for each timestamp
        timestamps = np.tile(np.arange(4096), timeseries.shape[0])  # Tile timestamp for each node_id

        # create index tuples
        index = pd.MultiIndex.from_arrays([index_num, timestamps], names=['index', 'timestamp'])

        # make multi layerd pd dataframe
        timeseries = pd.DataFrame(reshaped_data, index=index)
        # add names
        timeseries.columns = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
        
        
        return timeseries
    
    def _getwindowsize_(self):
        return len(self.x[1])
    
    def _get_min_max_std_(self):
        
        # turn timeseries from 3D dataframe to 2D pandas multi level dataframe for CoMTE algorithms (may need to put this into ECG Sequence function later)
        # make 3D numpy array
        timeseries = np.array(self.x)

        #reshape data for easier conversion to DF. flatten last dimension into columns (attributes)
        reshaped_data = timeseries.reshape((-1, timeseries.shape[-1]))

        # Example index creation for the DataFrame
        # Creating a MultiIndex with two levels: 'node_id' and 'timestamp'
        index_num = np.repeat(np.arange(timeseries.shape[0]), timeseries.shape[1])  # Repeat node_id for each timestamp
        timestamps = np.tile(np.arange(timeseries.shape[1]), timeseries.shape[0])  # Tile timestamp for each node_id

        # create index tuples
        index = pd.MultiIndex.from_arrays([index_num, timestamps], names=['index', 'timestamp'])

        # make multi layerd pd dataframe
        timeseries = pd.DataFrame(reshaped_data, index=index)
        # add names
        timeseries.columns = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

        ts_min = np.repeat(timeseries.min().values, timeseries.shape[1])
        ts_max = np.repeat(timeseries.max().values, timeseries.shape[1])
        ts_std = np.repeat(timeseries.std().values, timeseries.shape[1])

        return ts_min, ts_max, ts_std
    
    def _closehdf(self):
        #print('hdf closed')
        self.f.close() # close hdf5 dataset
    
