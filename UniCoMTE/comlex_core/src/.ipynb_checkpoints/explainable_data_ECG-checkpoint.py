import numpy as np
import pandas as pd


class ClfData:
    """
    Supports the ability to run data through the pandas-specific comte explainers pipeline,
    by transforming and wrapping the input data into the expected specific pandas format.
    """
    
    @staticmethod
    def wrap_df_data(raw_x_train, raw_y_train, feature_size, create_array=False):
        """
        Create the specific pandas dataframe data needed to pass through explainer

        Args:
            raw_x_train (iterable of array-likes): array-like samples
            raw_y_train (iterable): iterable of corresponding labels for the samples
            feature_size (int): number of features for a samples
            create_array (bool, optional): whether the iterables are arrays (or numpy arrays)

        Returns:
            pandas.dataframe, pandas.dataframe: dataframes of the sample and labels
        """
        x_train = raw_x_train
        y_train = raw_y_train
        if create_array:
            x_train = [x_point for x_point in raw_x_train]
            y_train = [y_point for y_point in raw_y_train]
        return ClfData.wrap_df_x(x_train, feature_size), ClfData.wrap_df_y(y_train)

    @staticmethod
    def wrap_df_x(x_train, feature_size):
        """
        Create the specific pandas dataframe sample data needed to pass through explainer.
        
        Args:
            x_train (iterable of array-likes): array-like samples. Specifically, the input is a 3D np dataframe (N,4096, 12).
            feature_size (int): number of features for a samples

        Returns:
            pandas.dataframe: dataframe of the sample data
        """

        # make 3D numpy array
        timeseries = np.array(x_train)
        
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
        x_train_df = pd.DataFrame(reshaped_data, index=index)
        # add names
        x_train_df.columns = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
        
        return x_train_df

    @staticmethod
    def wrap_df_y(y_train):
        """
        Create the specific pandas dataframe sample label needed to pass through explainer.

        Args:
            y_train (iterable): iterable of corresponding labels for the samples

        Returns:
            pandas.dataframe: dataframe of the sample label
        """

        # transform y_trained from (N,6) to (N,1)
        onehot_labels = True
        model_predictions = y_train
        
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
            mask = model_predictions == 1
            print(mask.shape)
            print(model_predictions.shape)
    
            # Ensure each row has at least one '1'
            # no_positive_class is a column vector
            # Find rows with all False (no '1') # rows with all false becomes true
            no_positive_class = ~mask.any(axis=1) 
            print(no_positive_class.shape)
            
            # Expand mask by adding a new first column of zeros
            mask = np.column_stack((no_positive_class, mask))
            print(mask.shape)
        
    
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


        numerical_labels = sample_classes
        dummy_train = pd.Series([(i, i) for i in range(len(numerical_labels))], name="dummy")
        y_train_df = pd.DataFrame(numerical_labels, index=dummy_train, columns=["label"])
        
        return y_train_df
    
    @staticmethod
    def wrap_df_test_point(test_point):
        """
        Wraps up a single test point into an array and then wrap that into a dataframe
        so that it can pass through explainer.

        Args:
            test_point (array-like): an array-like test point similar to the samples. Specifically, this is a (1,4096,12) df

        Returns:
            pandas.dataframe: dataframe of a single test point
        """
        # convert a 3D np dataframe to a 2D pd dataframe
        np_array_2d = np.squeeze(test_point, axis=0)  # Shape becomes (4096, 12)

        
        test_df = pd.DataFrame(np_array_2d)
        test_df.columns = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']

        
        return test_df
