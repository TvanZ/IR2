from collections import defaultdict
from random import random

import numpy as np
import pickle as pk

from sklearn.ensemble import GradientBoostingRegressor

NO_FEATURES = 700
NO_ITERATIONS = 10

GBDT = GradientBoostingRegressor(learning_rate=0.2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_probability(examined, relevant, clicked, theta, gamma):
    """
    Calculates the probabilities needed for computed the marginals P(E=1|..) and P(R=1|..).


    :param examined: boolean
    :param relevant: boolean
    :param clicked: boolean
    :param theta: observance parameter
    :param gamma: relevance parameter
    :return:
    """
    if examined and relevant and clicked:
        return 1.
    elif examined and not relevant and clicked:
        return 0.
    elif not examined and relevant and clicked:
        return 0.
    elif not examined and not relevant and clicked:
        return 0.

    elif not examined and not relevant and not clicked:
        return (1 - theta) * (1 - gamma) / (1 - theta * gamma)
    elif examined and not relevant and not clicked:
        return theta * (1 - gamma) / (1 - theta * gamma)
    elif not examined and relevant and not clicked:
        return gamma * (1 - theta) / (1 - theta * gamma)
    elif examined and relevant and not clicked:
        return 1 - get_probability(True, False, False, theta, gamma) - get_probability(False, True, False, theta,
                                                                                       gamma) - get_probability(False,
                                                                                                                False,
                                                                                                                False,
                                                                                                                theta,
                                                                                                                gamma)
    return None


def save(obj, name):
    """
    Saves a given object to results

    :param obj: marginals, theta, predictions
    :param name: filename
    """
    with open("results/" + name + ".pickle", 'wb') as handle:
        pk.dump(obj, handle, protocol=pk.HIGHEST_PROTOCOL)


def em_regression(query_document, click_model, top_n=10):
    """
    EM regression algorithm using GBDT regression for computing position bias (theta), marginals and predictions.
    Note: the algorithm disregards unranked query-documents pairs

    :param query_document: structured clicked log
    :param top_n: number of how many first positions to handle
    :return:
    """

    # initilization of the theta and gamma parameters, Theta with random distribution and gamma based on relevance
    Theta = np.random.rand(top_n)
    Gamma = defaultdict(dict)
    for qid in query_document.keys():
        querry = query_document[qid]
        for idx in querry.keys():
            doc = querry[idx]
            rank = doc['rank']
            if rank is not None:
                relevance = doc['label']
                Gamma[qid][idx] = relevance / 4
    S_train = []
    S_target = []
    F = 0
    P_E = defaultdict(dict)
    P_R = defaultdict(dict)

    for it in range(NO_ITERATIONS):

        # Expectation computing stage, we compute the marginals P(E=1|..) and P(R=1|..)

        for qid in query_document.keys():
            querry = query_document[qid]
            for idx in querry.keys():
                doc = querry[idx]
                rank = doc['rank'] - 1 if doc['rank'] else None
                clicked = doc['clicked']
                if rank is not None:
                    P_E[qid][idx] = \
                        get_probability(True, True, clicked, Theta[rank], Gamma[qid][idx]) + get_probability(
                            True, False,
                            clicked,
                            Theta[rank],
                            Gamma[qid][idx])

                    P_R[qid][idx] = \
                        get_probability(True, True, clicked, Theta[rank], Gamma[qid][idx]) + get_probability(False,
                                                                                                             True,
                                                                                                             clicked,
                                                                                                             Theta[
                                                                                                                 rank],
                                                                                                             Gamma[
                                                                                                                 qid][
                                                                                                                 idx])
        # sampeling binary for each ranked document query pair from the P(R=1|..) distribution
        S_train = []
        S_target = []
        for qid in query_document.keys():
            querry = query_document[qid]
            for idx in querry.keys():
                doc = querry[idx]
                features = doc['features']
                rank = doc['rank']
                if rank is not None:
                    r = 0
                    if random() < P_R[qid][idx]:
                        r = 1
                    x = np.array(features)
                    y = np.array(r)
                    S_train.append(x)
                    S_target.append(y)

        S_train = np.stack(S_train)
        S_target = np.stack(S_target)

        # regression using query documents features to the binary relevance
        F = GBDT.fit(S_train, S_target)

        # Maximization stage, updating the parameters Theta and Gamma
        for k in range(0, top_n):
            theta_k = 0
            counter = 0
            for qid in query_document.keys():
                querry = query_document[qid]
                for idx in querry.keys():
                    doc = querry[idx]
                    rank = doc['rank'] - 1 if doc['rank'] else None
                    c = 1 if doc['clicked'] is True else 0
                    if rank is not None and rank == k:
                        theta_k += c + (1 - c) * P_E[qid][idx]
                        counter += 1
                        break

            theta_k /= counter + 0.0001
            Theta[k] = theta_k

        for qid in query_document.keys():
            querry = query_document[qid]
            for idx in querry.keys():
                doc = querry[idx]
                features = doc['features']
                rank = doc['rank']
                if rank is not None:
                    Gamma[qid][idx] = sigmoid(F.predict(features.reshape(1, 700)))
        save(Theta, "theta_" + click_model)
        save(P_R, "rel_prob_" + click_model)
        save(P_E, "exm_prob_" + click_model)
        print("iteration: " + str(it + 1) + " done")

    # Compute if each query document pair is relevant or not
    preds = F.predict(S_train)
    save(preds, "preds_" + click_model)
    return Theta, preds, P_E, P_R
