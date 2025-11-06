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
            x_train (iterable of array-likes): array-like samples
            feature_size (int): number of features for a samples

        Returns:
            pandas.dataframe: dataframe of the sample data
        """
        i_1 = pd.Series([i for i in range(len(x_train))], name="node_id")
        i_2 = pd.Series([i for i in range(len(x_train))], name="dummy")
        x_train_df = pd.DataFrame(x_train, index=[i_1, i_2], columns=[i for i in range(feature_size)])
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
        dummy_train = pd.Series([(i, i) for i in range(len(y_train))], name="dummy")
        y_train_df = pd.DataFrame(y_train, index=dummy_train, columns=["label"])
        return y_train_df
    
    @staticmethod
    def wrap_df_test_point(test_point):
        """
        Wraps up a single test point into an array and then wrap that into a dataframe
        so that it can pass through explainer.

        Args:
            test_point (array-like): an array-like test point similar to the samples

        Returns:
            pandas.dataframe: dataframe of a single test point
        """
        test_np = np.array([test_point])
        test_df = pd.DataFrame(test_np)
        return test_df
