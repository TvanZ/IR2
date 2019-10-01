import numpy as np
import pickle as pk
from collections import defaultdict

DATABASE_PATH = "./database/"
DATABASE_NAME = "set1"
NO_FEATURES = 700

THRESHOLD = 1e-8


def compare_docs(doc1, doc2):
    err = .0
    for i in range(len(doc1)):
        err += abs(doc1[i] - doc2[i])

    err /= NO_FEATURES
    return err < THRESHOLD


def read_fold(fold):
    """

    Reads a data set fold
    """
    file = open(DATABASE_PATH + DATABASE_NAME + fold)

    query_document = defaultdict(dict)

    for idx, line in enumerate(file):
        line = line.split(' ')
        label = int(line[0])
        qid = int(line[1].split(':')[1])

        # documents features
        features = np.zeros(700)
        for i in range(2, len(line)):
            feature = line[i].split(':')
            id = int(feature[0]) - 1
            value = float(feature[1])
            features[id] = value

        clicked = False
        query_document[qid][idx] = {'label': label, 'features': features, 'rank': None, 'clicked': clicked}

    return query_document


