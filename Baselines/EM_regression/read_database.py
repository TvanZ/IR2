import numpy as np
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
    file = open(DATABASE_PATH + DATABASE_NAME + fold)

    query_document = defaultdict(list)

    k = 0
    for line in file:
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

        rank = len(query_document[qid])
        clicked = False
        query_document[qid].append(
            {'label': label, 'features': features, 'rank': rank, 'clicked': clicked})

        if k == 100:
            break
        k += 1
    return query_document


documents = []

fold = ".train.txt"
query_document_train = read_fold(fold)

fold = ".valid.txt"
query_document_valid = read_fold(fold)

fold = ".test.txt"
query_document_test = read_fold(fold)

print(query_document_train)
