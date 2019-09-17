from collections import defaultdict
from random import random

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

NO_FEATURES = 700
NO_ITERATIONS = 100

GBDT = GradientBoostingRegressor(learning_rate=0.2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_probability(examined, relevant, clicked, theta, gamma):
    if examined and relevant and clicked:
        return 1.
    elif examined and not relevant and clicked:
        return 0.

    elif examined and relevant and not clicked:
        return 1 - theta * (1 - gamma) / (1 - theta * gamma)
    elif examined and not relevant and not clicked:
        return theta * (1 - gamma) / (1 - theta * gamma)

    if not examined and relevant and clicked:
        return 0.
    elif not examined and relevant and not clicked:
        return (1 - theta) * gamma / (1 - theta * gamma)

    return None


def em_regression(query_document, top_n=10):
    Theta = np.random.rand(top_n)
    Gamma = defaultdict(defaultdict(float))
    for qid in query_document.keys():
        querry = query_document[qid]
        for idx, doc in enumerate(querry):
            relevance = doc['label']
            Gamma[qid][idx] = relevance / 4

    F = 0
    for _ in range(NO_ITERATIONS):
        P_E = defaultdict(defaultdict(float))
        P_R = defaultdict(defaultdict(float))
        for qid in query_document.keys():
            querry = query_document[qid]
            for idx, doc in enumerate(querry):
                rank = doc['rank']
                clicked = doc['clicked']

                P_E[qid][idx] = \
                    get_probability(True, True, clicked, Theta[rank], Gamma[qid][idx]) + get_probability(
                        True, False,
                        clicked,
                        Theta[rank],
                        Gamma[qid][idx])

                P_R[qid][idx] = \
                    get_probability(False, True, clicked, Theta[rank], Gamma[qid][idx]) + get_probability(False,
                                                                                                          True,
                                                                                                          clicked,
                                                                                                          Theta[
                                                                                                              rank],
                                                                                                          Gamma[qid][
                                                                                                              idx])

        S_train = []
        S_target = []
        for qid in query_document.keys():
            querry = query_document[qid]
            for idx, doc in enumerate(querry):
                features = doc['features']
                r = 0
                if random() < P_R[qid][idx]:
                    r = 1
                x = np.array(features)
                y = np.array(r)
                S_train.append(x)
                S_target.append(y)

        S_train = np.stack(S_train)
        S_target = np.stack(S_target)
        F = GBDT.fit(S_train, S_target)

        for k in range(0, top_n):
            theta_k = 0
            counter = 0
            for qid in query_document.keys():
                querry = query_document[qid]
                for idx, doc in enumerate(querry):
                    rank = doc['rank']
                    c = 1 if doc['clicked'] is True else 0
                    if rank == k:
                        theta_k += c + (1 - c) * P_E[qid][idx]
                        counter += 1
                        break

            theta_k /= counter
            Theta[k] = theta_k

        for qid in query_document.keys():
            querry = query_document[qid]
            for idx, doc in querry:
                features = doc['features']
                Gamma[qid][idx] = sigmoid(F(features))

    return Theta, F
