import numpy as np
from sklearn.neighbors import KDTree

class ClfModel:
    """
    steps: steps=[('reduce_dim', PCA()), ('clf', SVC())]
    uses function.column_names, ie: PCA().column_names
    -should become obsolete
    
    predict (predict_attr): function that takes in a list of values and returns list of labels
    array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
    
    predict_proba (predict_proba_attr): function that takes in a list of values and returns list of list of probabilities
    array([[0. , 1. , 0. ],
       [0. , 0.4, 0.6],
       [0. , 1. , 0. ],
       [1. , 0. , 0. ],
       [1. , 0. , 0. ],
       [1. , 0. , 0. ],
       [0. , 0. , 1. ],
       [0. , 1. , 0. ],
       [0. , 0. , 1. ],
       [1. , 0. , 0. ]])
    
    columns (column_attr): column_names of the initial model
    ie: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]    

    classes_ (classes_attr): classes_ndarray of shape (n_classes,) The classes labels.
    ie: [0, 2, 1] for 4 classes with INDEX corresponding to prediction index:
    ie: [0.4, 0.5, 0.1] = [0: 0.4, 2: 0.5, 1: 0.1]
    ie: [0, 1] for WASP or ["folded", "allclear", "etc"] for natops
    aka: np.ndarray of list(set(test_labels['label'].values))
    """
    # predictor
    def __init__(self, clf, predict_attr=None, predict_proba_attr=None, column_attr=None, classes_attr=None, window_size_attr=None):
        self.orig_clf_obj = clf
        self.predict = predict_attr
        self.predict_proba = predict_proba_attr
        self.metrics = column_attr
        self.classes_ = classes_attr
        self.window_size = window_size_attr
        if hasattr(clf, "steps"):
            self.steps = clf.steps
        if column_attr is None:
            if not hasattr(clf, "steps"):
                print("columns property not defined!")
                return
            self.metrics = clf.steps[0][1].column_names
        if predict_attr is None:
            if not hasattr(clf, "predict"):
                print("predict property not defined!")
                return
            self.predict = clf.predict
        if predict_proba_attr is None:
            if not hasattr(clf, "predict_proba"):
                print("predict_proba property not defined!")
                return
            self.predict_proba = clf.predict_proba
        if classes_attr is None:
            if not hasattr(clf, "classes_"):
                print("classes_ property not defined!")
                return
            self.classes_ = clf.classes_
