import pickle
import pandas as pd
import numpy as np
import os


def read_results(click_model_path, trial_num=0, randomized_results=None):
    """

    :param click_model_path:
    :param trail_num:
    :return:
    """
    # open the new pickle file
    click_model = unpickle_results(click_model_path)
    # only initialize randomized_results.pickle for first trial_run
    if trial_num == 0:
        randomized_results = {}

    # otherwise, we should be passing previously generated randomized_results.pickle to this function
    for query in click_model.keys():
        docs_list = click_model[query]
        for docID, doc in docs_list.items():
            query_results = [docID, doc['rank'], doc['clicked']]
            # initialize query key and value
            if query not in randomized_results:  # trial_num == 0:
                randomized_results[query] = []
            randomized_results[query].append(query_results)
    return randomized_results


# ============== Pickling/Unpickling files ========================
def save_results(results_dict, filename):
    # use pickle to save dict
    with open(filename, 'wb') as my_dict:
        pickle.dump(results_dict, my_dict, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_results(results_dict):
    with open(results_dict, 'rb') as click_model:
        results_dict = pickle.load(click_model)
    return results_dict


