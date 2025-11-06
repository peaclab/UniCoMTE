#!/usr/bin/env python3
"""
    modified code that allows for KDTree input
    this allows us to bypass waiting for training and creation of KDTrees
"""

import logging
import numbers
import multiprocessing
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import mlrose_ky as mlrose
import time


class BaseExplanation:
    def __init__(self, clf, timeseries, labels, kd_tree, silent=True,
                 num_distractors=2, dont_stop=False, 
                 threads=multiprocessing.cpu_count()):
        self.clf = clf
        self.timeseries = timeseries
        self.labels = labels
        self.silent = silent
        self.num_distractors = num_distractors
        if hasattr(clf, "metrics") and clf.metrics is not None:
            self.metrics = clf.metrics
        else:
            self.metrics = self.clf.steps[0][1].column_names
        self.dont_stop = dont_stop
        if hasattr(clf, "window_size") and clf.window_size is not None:
            self.window_size = clf.window_size
        else:
            self.window_size = len(timeseries.loc[
                timeseries.index.get_level_values('node_id')[0]])
        self.ts_min = np.repeat(timeseries.min().values, self.window_size)
        self.ts_max = np.repeat(timeseries.max().values, self.window_size)
        self.ts_std = np.repeat(timeseries.std().values, self.window_size)
        self.tree = None
        self.per_class_trees = kd_tree   ## user must provide an input for KDTree
        self.threads = threads

    def explain(self, x_test, **kwargs):
        raise NotImplementedError("Please don't use the base class directly")

    def _get_feature_names(self, clf, timeseries):
        if hasattr(self.clf.steps[1][1], 'transform'):
            return self.clf.steps[2][1].column_names
        else:
            window_size = len(timeseries.loc[
                [timeseries.index.get_level_values('node_id')[0]], :, :])
            names = []
            for c in timeseries.columns:
                for i in range(window_size):
                    names.append(c + '_' + str(i) + 's')
            return names

    def _transform_data(self, data, sample=None):
        if hasattr(self.clf.steps[1][1], 'transform'):
            transformed = self.clf.steps[1][1].transform(data)
            if sample:
                transformed = transformed.sample(sample)
            return self.clf.steps[3][1].transform(transformed)
        else:
            # autoencoder
            train_set = []
            for node_id in data.index.get_level_values('node_id').unique():
                train_set.append(data.loc[[node_id], :, :].values.T.flatten())
            result = np.stack(train_set)
            if sample:
                idx = np.random.randint(len(result), size=sample)
                result = result[idx, :]
            return result

    def _plot_explanation(self, original, distractor, explanations, savefig=False, timeseries=True, filename=None):
        fig = plt.figure(figsize=(6,3))
        ax = fig.gca()
        original_val = original.values[0] if hasattr(original, "values") else original
        distractor_val = distractor.values[0] if hasattr(distractor, "values") else [distractor[0]]
        if timeseries:
            plt.plot(range(distractor.shape[1]),
                    original_val, label='Original',
                    figure=fig,
                    )
            plt.plot(range(distractor.shape[1]),
                    distractor_val, label='Distractor',
                    figure=fig)
            ax.set_xlabel('Time (s)')
        else:
            plt.plot(range(distractor.shape[1]),
                    original_val, label='Original',
                    figure=fig, color='blue',
                    marker='o',
                    )
            plt.plot(range(distractor.shape[1]),
                    distractor_val, label='Distractor',
                    figure=fig, color='red',
                    marker='x')
            ax.set_xlabel('X')
        ax.set_ylabel('Y')
        for explanation in explanations:
            plt.axvline(x = explanation, color = 'r', label = str(explanation))
        ax.legend()
        if savefig:
            if not filename:
                filename = "{}.pdf".format(uuid.uuid4())
            fig.savefig(filename, bbox_inches='tight')
            logging.info("Saved the figure to %s", filename)
            plt.close(fig)
        else:
            fig.show()
        
    def _plot_changed(self, metric, original, distractor, savefig=False, filename=None):
        fig = plt.figure(figsize=(6,3))
        ax = fig.gca()
        orig_metric = original[metric].values if hasattr(original[metric], "values") else [original[metric]]
        distractor_metric = distractor[metric].values if hasattr(distractor[metric], "values") else [distractor[metric]]
        plt.plot(range(distractor.shape[0]),
                 orig_metric, label='x$_{test}$',
                 figure=fig,
                 )
        plt.plot(range(distractor.shape[0]),
                 distractor_metric, label='Distractor',
                 figure=fig)
        ax.set_ylabel(metric)
        ax.set_xlabel('Time (s)')
        ax.legend()
        if savefig:
            if not filename:
                filename = "{}.pdf".format(uuid.uuid4())
            fig.savefig(filename, bbox_inches='tight')
            logging.info("Saved the figure to %s", filename)
            plt.close(fig)
        else:
            fig.show()

    def construct_per_class_trees(self):
        """Used to choose distractors"""
        # check if KDTree has been created
        if self.per_class_trees is not None:
            for c, tree in self.per_class_trees.items():
                num_indices = len(tree.data)  # The number of points in the KDTree
                print(f"Class {c} has {num_indices} indices.")
            return
        
        # initialize
        self.per_class_trees = {}
        self.per_class_node_indices = {c: [] for c in self.clf.classes_}
        print('making predictions for per class trees')
        preds = self.clf.predict(self.timeseries)

        from collections import Counter
        #checking preds ...
        print('Here are the predictions made')
        counter = Counter(preds)
        # Print unique items and their frequencies
        for item, freq in counter.items():
            print(f"{item}: {freq}")

        # check which 
        true_positive_node_ids = {c: [] for c in self.clf.classes_}
        
        for pred, (idx, row) in zip(preds, self.labels.iterrows()):
            # print to check
            # print(f'pred is {pred}')
            # print(f'row is {row}')
            # print(f"row label is {row['label']}")
            

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

        print(f"shape of self.timeseries is {self.timeseries.shape}")
        print(f"type of self.timeseries is {type(self.timeseries)}")
        start_time = time.time()
        for c in self.clf.classes_:
            dataset = []
            for node_id in true_positive_node_ids[c]:
                # The below syntax of timeseries.loc[[node_id], :, :] is extremely fragile. The first two ranges index into the multi-index
                # while the third range indexes the columns. But anything other than ":" for the third range causes the code to crash, apparently
                # due to ambiguity. See the Warning here: https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#using-slicers
                try:
                    sliced_node = self.timeseries.loc[[node_id], :, :]
                except pd.errors.IndexingError: # try slicing with fallback
                    sliced_node = self.timeseries.loc[[node_id], :]

                
                
                dataset.append(sliced_node.values.T.flatten())
                self.per_class_node_indices[c].append(node_id)
            if dataset:
                self.per_class_trees[c] = KDTree(np.stack(dataset))

        end_time = time.time()
        print(f"elapsed time = {end_time - start_time}")

        if not self.silent:
            logging.info("Finished constructing per class kdtree")
        
        print('KD Tree Structure printout:')
        for label, tree in self.per_class_trees.items():
            print(f"\nKD Tree for class: {label}")
            print(f"Tree data shape: {tree.data.shape}")  # Data shape
        for c, tree in self.per_class_trees.items():
                num_indices = len(tree.data)  # The number of points in the KDTree
                print(f"Class {c} has {num_indices} indices.")
        

    def construct_tree(self):
        if self.tree is not None:
            return
        train_set = []
        self.node_indices = []
        for node_id in self.timeseries.index.get_level_values(
                'node_id').unique():
            train_set.append(self.timeseries.loc[
                [node_id], :, :].values.T.flatten())
            self.node_indices.append(node_id)
        self.tree = KDTree(np.stack(train_set))
        if not self.silent:
            logging.info("Finished constructing the kdtree")

    def _get_distractors(self, x_test, to_maximize, n_distractors=2):
        import sys
        self.construct_per_class_trees()
        # to_maximize can be int, string or np.int64
        if isinstance(to_maximize, numbers.Integral):
            to_maximize = self.clf.classes_[to_maximize]
        
        distractors = []
        distractors_idx = []

        if not isinstance(x_test, pd.DataFrame): # instantiate array-like as DataFrame
            x_test = pd.DataFrame(x_test)

        if to_maximize not in self.per_class_trees.keys():
            print('there are no distractors to explain the class of interest. Please select another class of interest')
            sys.exit()

        # EXTRA  
        # make and modify list of indices
        queried_indices = self.per_class_trees[to_maximize].query(
                x_test.values.T.flatten().reshape(1, -1),
                k=n_distractors+33)[1].flatten()
        print(f"queried indices are {queried_indices}")

        # remove index if it is present in the remove list (for class 0)
        remove_samples = {191664,
                        173587,
                        241987,
                        16276,
                        2503,
                        128186,
                        200550,
                        105241,
                        199739,
                        167677,
                        101082,
                        211539,
                        105506,
                        152205,
                        61685,
                        221222,
                        92533,
                        199794,
                        142860,
                        77362,
                        147504,
                        267315,
                        156202,
                        222377,
                        65574,
                        49150,
                        122168,
                        77297,
                        98568,
                        173823,
                        118504,
                        255993,
                        152912,
                        162368}
        
        keep_mask = np.array([x not in remove_samples for x in queried_indices])
        queried_indices_edited = queried_indices[keep_mask]
        print(f"edited queried indices: {queried_indices_edited}")
        
        # reduce size to length n_distractors
        if queried_indices_edited.shape[0] >= n_distractors:
            final_queried_indices = queried_indices_edited[:n_distractors]
        else:
            print('check!!')
            final_queried_indices = queried_indices[:n_distractors]



        for idx in final_queried_indices:
            # when there are no self.per_class_node_indices (since KDTree was given)
            # if not self.per_class_node_indices:
            print('need to reconstruct data from KDTree')
            flat = np.array(self.per_class_trees[to_maximize].data[idx])
            sliced_distractor = flat.reshape(12, 4096).T
            print(sliced_distractor.shape)
            # set distractor columns
            column_names = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            # Create the DataFrame with the given column names
            sliced_distractor = pd.DataFrame(sliced_distractor, columns=column_names)
            print(sliced_distractor.shape)
            print(type(sliced_distractor))
            distractors.append(sliced_distractor)
            distractors_idx.append(idx)
 
            # else: 
            #     # when kdtree was not given and there are node indices
            #     try:
            #         sliced_distractor = self.timeseries.loc[[self.per_class_node_indices[to_maximize][idx]], :, :]
            #     except pd.errors.IndexingError: # try slicing with fallback
            #         sliced_distractor = self.timeseries.loc[[self.per_class_node_indices[to_maximize][idx]], :]
            #         sliced_distractor['node_id'] = [idx]
            #         sliced_distractor.set_index('node_id', inplace=True) # aka sample_id

        if not self.silent:
            logging.info("Returning distractors %s", [
                x.index.get_level_values('node_id').unique().values[0]
                for x in distractors])
            
        print(f"STARSTAR indices of distractors on KDTree for class {to_maximize} are {distractors_idx}")
        
        return distractors

CLASSIFIER = None
X_TEST = None
DISTRACTOR = None

def _eval_one(tup):
    column, label_idx = tup
    global CLASSIFIER
    global X_TEST
    global DISTRACTOR
    x_test = X_TEST.copy()
    x_test[column] = DISTRACTOR[column].values
    pred = CLASSIFIER.predict_proba(x_test)[0][label_idx]
    return pred


class BruteForceSearch(BaseExplanation):
    def _find_best(self, x_test, distractor, label_idx):
        global CLASSIFIER
        global X_TEST
        global DISTRACTOR
        CLASSIFIER = self.clf
        X_TEST = x_test
        DISTRACTOR = distractor
        best_case = self.clf.predict_proba(x_test)[0][label_idx]
        best_column = None
        tuples = []
        for c in distractor.columns:
            dist_c = distractor[c].values if hasattr(distractor[c], "values") else [distractor[c]]
            test_c = x_test[c].values if hasattr(x_test[c], "values") else [x_test[c]]
            if np.any(dist_c != test_c):
                tuples.append((c, label_idx))


        # remove if statement for max pooling temporarily
        #if self.threads == 1:
        results = []
        for t in tuples:
            results.append(_eval_one(t))
        #else:
        #    print('pooling')
        #    pool = multiprocessing.Pool(self.threads)
        #    results = pool.map(_eval_one, tuples)
        #    pool.close()
        #    pool.join()

        for (c, _), pred in zip(tuples, results):
            if pred > best_case:
                best_column = c
                best_case = pred
        if not self.silent:
            logging.info("Best column: %s, best case: %s",
                         best_column, best_case)
        return best_column, best_case

    def explain(self, x_test, to_maximize=None, num_features=3, return_dist=False,
                savefig=False, single=False, train_iter=100, timeseries=True, filename=None):
        
        print('\n-------- Using Greedy Search --------')
        
        orig_preds_probas = self.clf.predict_proba(x_test)
        orig_label = self.clf.predict(x_test)
        
        # prints to check functionality
        print(f"original sample probabilities: {orig_preds_probas}")
        print(f"original sample classification: {orig_label}")
        print(f"class of interest: {to_maximize}")  

        if to_maximize is None:
            to_maximize = np.argmin(orig_preds_probas)
        if orig_label == to_maximize:
            return []
        if not self.silent:
            logging.info("Working on turning label from %s to %s",
                         orig_label, to_maximize)
        
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors)
        
        # init vars
        best_explanation = set()
        best_explanation_score = 0

        for count, dist in enumerate(distractors):
            if not self.silent:
                logging.info("Trying distractor %d / %d",
                             count + 1, self.num_distractors)
            dist_idx = 900
            
            # initializations
            explanation = []
            modified = x_test.copy()
            prev_best = 0
            # best_dist = dist
            print(f"trying distractor {count + 1} of {self.num_distractors}")


            # need code to find the best dist  
            while True:
                probas = self.clf.predict_proba(modified)[0]
                label = self.clf.predict(modified)[0]
                print(f'probability is {probas}')
                print(f'label is {label}')

                if not self.silent:
                    logging.info("Current probas: %s", probas)

                if label == to_maximize:
                    print("label equates to to_max!")
                    current_best = probas[label]
                    if current_best > best_explanation_score:
                        best_explanation = explanation
                        best_dist = dist # update best distractor if this distractor's score is better
                        best_explanation_score = current_best
                        dist_idx = count
                        print(f"best explanation is now {explanation}")
                        print(f"best explanation score is now {best_explanation_score}")
                        print(f"dist_idx is {dist_idx}")
                        
                    if current_best <= prev_best:    
                        break
                    prev_best = current_best
                    if not self.dont_stop:   
                        break

                # not sure what this does
                # if self.dont_stop is False and there is a best explanation already
                # will the loop already break from the code above?
                if (not self.dont_stop and
                        len(best_explanation) != 0 and
                        len(explanation) >= len(best_explanation)):
                    break

                #print("finding best_col")
                best_column, _ = self._find_best(modified, dist, to_maximize)
                print(f'best col is {best_column}')

                if best_column is None:
                    break
                
                if not self.silent and not single:
                    self._plot_changed(best_column, modified, dist, savefig=savefig, filename=filename)
                
                # update modified
                modified[best_column] = dist[best_column].values
                
                explanation.append(best_column)
                print(f"explanation is now{explanation}")
            
            if not self.silent and single and len(best_explanation) != 0:
                self._plot_explanation(x_test, best_dist, best_explanation, savefig=savefig, timeseries=timeseries, filename=filename)

        # prints for clarity
        print(f"Final greedy search explanation: {best_explanation}")
        print(f"STARSTAR final distractor idx used {dist_idx}")
        print(f"Final sample probabilities: {self.clf.predict_proba(modified)}")

        if not return_dist:
            return best_explanation
        else:
            return best_explanation, best_dist


class LossDiscreteState:

    def __init__(self, label_idx, clf, x_test, distractor, cols_swap, reg,
                 max_features=3, maximize=True):
        self.target = label_idx
        self.clf = clf
        self.x_test = x_test
        self.reg = reg
        self.distractor = distractor
        self.cols_swap = cols_swap  # Column names that we can swap
        self.prob_type = 'discrete'
        self.max_features = 3 if max_features is None else max_features
        self.maximize = maximize

    def __call__(self, feature_matrix):
        return self.evaluate(feature_matrix)

    def evaluate(self, feature_matrix):
        #print('evaluating')
        #print(feature_matrix)

        new_case = self.x_test.copy()
        assert len(self.cols_swap) == len(feature_matrix)
        #print(self.clf.predict_proba(new_case))

        # If the value is one, replace from distractor
        for col_replace, a in zip(self.cols_swap, feature_matrix):
            if a == 1:
                #print('replacing')
                if hasattr(new_case[col_replace], '__iter__'):
                    new_case[col_replace] = self.distractor[col_replace].values
                else: # new_case is a single value instead of array-like
                    new_case[col_replace] = self.distractor[col_replace].values[0]



        replaced_feature_count = np.sum(feature_matrix)

        # if replaced_feature_count > self.max_features:
        #     feature_loss = 1
        #     loss_pred = 1
        # else:
        # Will return the prob of the other class
        #print(self.clf.predict_proba(new_case))
        result = self.clf.predict_proba(new_case)[0][self.target]
        #print(f'modified probability is {result}')
        feature_loss = self.reg * np.maximum(0, replaced_feature_count - self.max_features)
        #print(f'feature loss is {feature_loss}')
        loss_pred = np.square(np.maximum(0, 0.95 - result))
        #print(np.maximum(0, 0.95 - result))
        #print(f"loss_pred is {loss_pred}")
        loss_pred = loss_pred + feature_loss

        
        #print(-loss_pred)

        return -loss_pred if self.maximize else loss_pred

    def get_prob_type(self):
        """ Return the problem type."""

        return self.prob_type


class OptimizedSearch(BaseExplanation):

    def __init__(self, clf, timeseries, labels, kd_tree, **kwargs):
        super().__init__(clf, timeseries, labels, kd_tree, **kwargs)
        self.discrete_state = False
        self.backup = BruteForceSearch(clf, timeseries, labels, kd_tree, **kwargs)
    
    

    def opt_Discrete(self, to_maximize, x_test, dist, columns, init,
                     max_attempts, maxiter, num_features=None):

        print("\n\n-------Running Discrete Optimization Algorithm for Feature Selection------")

        # define fitness func
        fitness_fn = LossDiscreteState(
            to_maximize,
            self.clf, 
            x_test, 
            dist,
            columns, 
            reg=0.8, 
            max_features=num_features,
            maximize=False)
        
        #print('define fitness function')
        #print(to_maximize)
        #print(columns)
        #print(num_features)
        
        problem = mlrose.DiscreteOpt(
            length=len(columns), fitness_fn=fitness_fn,
            maximize=False, max_val=2)
        
        #print('define problem')
        #print(len(columns))
        
        best_state, best_fitness, _ = mlrose.random_hill_climb(  
            problem,
            max_attempts=max_attempts,
            max_iters=maxiter,
            init_state=init,
            restarts = 0,
            random_state = 42)
        
        #print("random hill climbing")
        #print(max_attempts)
        #print(maxiter)

        self.discrete_state = True
        return best_state

    def _prune_explanation(self, explanation, x_test, dist,
                           to_maximize, max_features=None):
        if max_features is None:
            max_features = len(explanation)
        short_explanation = set()
        while len(short_explanation) < max_features:
            modified = x_test.copy()
            for c in short_explanation:
                modified[c] = dist[c].values
            prev_proba = self.clf.predict_proba(modified)[0][to_maximize]
            best_col = None
            best_diff = 0
            for c in explanation:
                tmp = modified.copy()
                tmp[c] = dist[c].values
                cur_proba = self.clf.predict_proba(tmp)[0][to_maximize]
                if cur_proba - prev_proba > best_diff:
                    best_col = c
                    best_diff = cur_proba - prev_proba
            if best_col is None:
                break
            else:
                short_explanation.add(best_col)
        return short_explanation

    def explain(self, x_test, num_features=None, to_maximize=None, return_dist = False,
                savefig=False, single=False, train_iter=100, timeseries=True, filename=None, custom=False):
        
        # num_feature is maximum number of features
        orig_label = self.clf.predict(x_test)
        orig_preds_probas = self.clf.predict_proba(x_test)

        print('-------Preliminary Statistics-------')
        print(f'Original Sample Class: {orig_label} \nSample Probabilities: {orig_preds_probas}\nClass of Interest: {to_maximize}\n\n')
        
        #binary classification
        if to_maximize is None:
            to_maximize = np.argmin(orig_preds_probas)
        
        if isinstance(orig_label, tuple) and to_maximize in orig_label:
            print('Sample is already classified as the class of interest')
            return []
        elif orig_label == to_maximize:
            print('Sample is already classified as the class of interest')
            return []
        
        if not self.silent:
            logging.info("Working on turning label from %s to %s",
                         self.clf.classes_[orig_label],
                         self.clf.classes_[to_maximize])
        explanation = self._get_explanation(
            x_test, to_maximize, num_features, return_dist, savefig=savefig,
            single=single, train_iter=train_iter, timeseries=timeseries, filename=filename)
        


        if not explanation:
            #if isinstance(x_test, pd.DataFrame) and not custom:
            #    logging.info("Used greedy search for %s",
            #                x_test.index.get_level_values('node_id')[0])
            #else:
            #    logging.info("Used greedy search for %s",
            #                x_test)
            print('using greedy search')
            explanation = self.backup.explain(x_test, num_features=num_features,
                                              to_maximize=to_maximize, return_dist=return_dist,
                                              savefig=savefig, single=single, train_iter=train_iter,
                                              filename=filename)
            
            print(f"GS explanation is {explanation}")
        return explanation

    def _get_explanation(self, x_test, to_maximize, num_features, return_dist=False,
                         savefig=False, single=False, train_iter=100, timeseries=True,
                         filename=None):
        
        print('generating distractors')
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors)
        print('validate distractors probabilities:')
        for count, dist in enumerate(distractors): 
            print(f"Distractor {count + 1} probability: ")
            print(self.clf.predict_proba(dist))

        # Avoid constructing KDtrees twice
        self.backup.per_class_trees = self.per_class_trees
        # self.backup.per_class_node_indices = self.per_class_node_indices

        best_explanation = set()
        best_explanation_score = 0

        for count, dist in enumerate(distractors):
            print(f"\nprocessing distractor {count + 1} of {self.num_distractors}")
            dist_idx = 900

            if not self.silent:
                logging.info("Trying distractor %d / %d",
                             count + 1, self.num_distractors)
            columns = []
            print(x_test.columns)
            for c in dist.columns:
                dist_c = dist[c].values if hasattr(dist[c], "values") else [dist[c]]
                test_c = x_test[c].values if hasattr(x_test[c], "values") else [x_test[c]]
                if np.any(dist_c != test_c):
                    columns.append(c)

            # Init options
            init = [0] * len(columns)

            columns = np.array(columns)
            init = np.array(init)

            dist_probas = self.clf.predict_proba(dist)
            print(f'Probabilities of distractor sample: {dist_probas}')
            print('mlrose starting')
            result = self.opt_Discrete(
                to_maximize, x_test, dist, columns, init=init,
                max_attempts=train_iter, maxiter=train_iter, num_features=num_features)
            print(f"mlrose result result is:{result}")

            if not self.discrete_state:
                explanation = {
                    x for idx, x in enumerate(columns)
                    if idx in np.nonzero(result.x)[0]
                }
            else:
                explanation = {
                    x for idx, x in enumerate(columns)
                    if idx in np.nonzero(result)[0]
                }
            print('pruning')
            explanation = self._prune_explanation(
                explanation, x_test, dist, to_maximize, max_features=num_features)
            print(f"pruning output is {explanation}")
            modified = x_test.copy()

            for c in columns:
                if c in explanation:
                    modified[c] = dist[c].values

            # check if the modified sample is classified as class of interest 
            # For multiple classes, check of one of the classes is the class of interest
            modified_label = self.clf.predict(modified)[0]
            probas = self.clf.predict_proba(modified)[0]
            print(f"pruned explanation is {explanation}")
            print(f"pruned sample probabilities are {probas}")
            print(f"pruned sample class is {modified_label}")
            print(f'to_max is {to_maximize}')
            print(modified_label==to_maximize)
            
            if not self.silent:
                logging.info("Current probas: %s", probas)
            
            successfully_modified = False
            if isinstance(modified_label, tuple):
                if to_maximize in modified_label:
                    successfully_modified = True
            else:
                if modified_label == to_maximize:
                    successfully_modified = True

            if successfully_modified:
                current_best = probas[to_maximize]
                if current_best > best_explanation_score:
                    
                    best_explanation = explanation
                    print(f'best explanation is {best_explanation}')
                    
                    best_explanation_score = current_best
                    best_modified = modified
                    best_dist = dist
                    dist_idx = count
                    print(f"dist_idx is {dist_idx}")

        if not self.silent and len(best_explanation) != 0:
            if single:
                self._plot_explanation(x_test, best_dist, best_explanation, savefig=savefig, timeseries=timeseries, filename=filename)
            else:
                for metric in best_explanation:
                    self._plot_changed(metric, x_test, best_dist, savefig=savefig, filename=filename)

        print('algorithm completed with optimized search')
        print(f"best_explanation is: {best_explanation}")
        print(f"STARSTAR final distractor index is: {dist_idx}")
        #print(f"best distractors is: {best_dist}")

        if return_dist == False or len(best_explanation) == 0:
            print('case 1')
            return best_explanation
        else:
            return best_explanation, best_dist
