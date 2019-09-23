import numpy as np
import pickle as pk
from random import random
from collections import defaultdict

DATABASE = "./database/"
DEBUG = False


def generate_click(rank, relevance):
    if rank is None:
        return False

    prob_examination = 1. / rank

    if random() < prob_examination:
        if relevance > 2:
            return True
        if random() <= 0.1:
            return True
    return False


def update_fold_click(fold):
    with open('qd_' + fold + '.pickle', 'rb') as handle:
        qd = pk.load(handle)

    for qid in qd.keys():
        querry = qd[qid]
        for idx in querry.keys():
            doc = querry[idx]
            rank = doc['rank']
            relevance = doc['label']
            clicked = generate_click(rank, relevance)
            qd[qid][idx]['clicked'] = clicked

    if DEBUG:
        k = 0
        for qid in qd.keys():
            for idx in qd[qid].keys():
                clicked = qd[qid][idx]["clicked"]
                rank = qd[qid][idx]["rank"]
                if rank is not None and rank >0 and clicked:
                    print("qid:" + str(qid) + "idx: " + str(idx) + " clicked: " + str(clicked) + " rank: " + str(rank))


    if not DEBUG:
        with open('qd_' + fold + '.pickle', 'wb') as handle:
            pk.dump(qd, handle, protocol=pk.HIGHEST_PROTOCOL)

