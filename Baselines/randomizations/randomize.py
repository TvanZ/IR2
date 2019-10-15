import pickle
import os
import numpy as np
# from Baselines.EM_regression.update_click import update_fold_click
from Baselines.randomizations.update_click import update_fold_click
from Baselines.randomizations.random_utils import unpickle_results


def get_topN_docs(click_model, queryID):
    """
    Returns the sorted topN docs for a single queryID.
    :param click_model: The unpickled click model.

    :return: ranked_docs list
    """
    # for every queryID, find the first top 10 relevant docs
    unordered_docs = []  # list of [ind, rank] for all relevant docs
    for ind in click_model[queryID]:
        document = click_model[queryID][ind]
        rank = document['rank']
        # moving the docID into the doc-dictionary, so I can shuffle documents and easily keep this info
        # a little hackey, but can be changed if needed later down the pipeline
        document['docID'] = ind
        if rank is not None:
            unordered_docs.append([ind, rank])
    # sorts unordered_docs by doc rankings
    ordered_docs = (sorted(unordered_docs, key=lambda docs: docs[1]))
    ordered_docIDs = list(zip(*ordered_docs))[0]

    ranked_docs = []
    for docID in ordered_docIDs:
        ranked_docs.append(click_model[queryID][docID])
    return ranked_docs


def shuffle(ranked_docs, randomType, topN=5):

    if randomType == "randTopN":  # shuffling top 5 documents randomly
        if len(ranked_docs) >= 5:
            shuffled_inds = np.random.permutation(topN)
        else:
            shuffled_inds = np.random.permutation(len(ranked_docs))

    elif randomType == "randPair":  # shuffling pairwise
        if len(ranked_docs) >= 5:
            shuffled_inds = np.arange(topN)
        else:
            shuffled_inds = np.arange(len(ranked_docs))

        # shuffling inds pair-wise
        for ind in range(len(shuffled_inds) - 1):
            shuffle_docs = np.random.choice([True, False])
            if shuffle_docs:
                shuffled_inds[ind], shuffled_inds[ind + 1] = shuffled_inds[ind + 1], shuffled_inds[ind]
    else:
        "ERROR! Unknown randomization type. Either input 'Top_RandN' or 'RandPairs' to randomize rankings."

    # adding the non-shuffled indicies back in
    nonshuffled_inds = np.arange(len(ranked_docs) - topN) + topN
    shuffled_inds = np.append(shuffled_inds, nonshuffled_inds)

    # use shuffled_inds to redefine ranked_docs
    shuffled_docs = [ranked_docs[ind] for ind in shuffled_inds]
    return shuffled_docs


def randomize(click_model_path, selected_randomType, click_simulation_method ='POSITION_BIASED_MODEL'):
    """
    Saves click decisions for shuffled rankings as pickle file.

    :param click_model_path: Pickled click model for un-shuffled rankings
    :param selected_randomType: Randomization method to use. Either Top_RandN or RandPairs.
    :param click_model: The click model called to simulate user clicks
    :return: None.
    """
    # loading the click model
    click_model = unpickle_results(click_model_path)

    # looping through all queryIDs
    query_IDs = click_model.keys()
    shuffled_results = {}
    for queryID in query_IDs:
        # First, access ranked docs --> get a list of dictionaries (each dict describes a doc)
        ranked_documents = get_topN_docs(click_model, queryID)
        shuffled_documents = shuffle(ranked_docs=ranked_documents, randomType=selected_randomType)

        for counter, doc in enumerate(shuffled_documents):
            doc['rank'] = counter + 1
            doc['clicked'] = False

        shuffled_results[queryID] = shuffled_documents

    with open(os.path.join('outputs', 'qd_shuffled.pickle'), 'wb') as handle:
        pickle.dump(shuffled_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calling cascade model to over shuffled docs
    pickled_filename = update_fold_click('shuffled', click_simulation_method)

    return pickled_filename


















