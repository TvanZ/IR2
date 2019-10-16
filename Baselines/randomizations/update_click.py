import numpy as np
import pickle as pk
from random import random
from collections import defaultdict
import os

from GANoN.click_models import CascadeModel, SimpleModel, PositionBiasedModel

CASCADE_MODEL = "cascade_model"
SIMPLE_MODEL = "simple_model"
POSITION_BIASED_MODEL = "position_biased_model"

DATABASE = "./database/"
DEBUG = False
DEBUG = False

cascade_generator = CascadeModel()
simple_generator = SimpleModel()
position_bias_generator = PositionBiasedModel()


def generate_position_biased_click(rank, relevance_label):
    click, _, _ = position_bias_generator.sampleClick(rank, relevance_label)
    return click == 1


def generate_simple_click(rank, relevance_label):
    click, _, _, _ = simple_generator.sampleClick(rank, relevance_label)
    return click == 1


def generate_cascade_click(was_clicked, relevance_label):
    click, _, _, _ = cascade_generator.sampleClick(was_clicked, relevance_label)
    return click == 1


def reset(qd):
    for qid in qd.keys():
        query = qd[qid]
        for idx in query.keys():
            qd[qid][idx]['clicked'] = False


def generate_clicks(qd, qid, click_model):
    query = qd[qid]
    if click_model == CASCADE_MODEL:
        print("In 'if click_model == CASCADE_MODEL' statement.")
        current_rank = 0
        clicked = False

        ranking = [0 for _ in range(10)]
        maxi = 0
        for idx in query.keys():
            doc = query[idx]
            rank = doc['rank']
            if rank is not None:
                maxi += 1
                ranking[rank - 1] = idx

        while not clicked and current_rank < maxi:
            idx = ranking[current_rank]
            doc = query[idx]
            rel = doc['label']
            clicked = generate_cascade_click(clicked, rel)
            qd[qid][idx]['clicked'] = clicked
            current_rank += 1

    if click_model == SIMPLE_MODEL:
        print("In 'click_model == SIMPLE_MODEL' statement.")
        for idx in query.keys():
            doc = query[idx]
            rel = doc['label']
            rank = doc['rank']
            if rank is not None:
                clicked = generate_simple_click(rank, rel)
                qd[qid][idx]['clicked'] = clicked

    if click_model == POSITION_BIASED_MODEL:
        print("In 'click_model == POSITION_BIASED_MODEL' statement")
        for idx in query.keys():
            doc = query[idx]
            rel = doc['label']
            rank = doc['rank']
            if rank is not None:
                clicked = generate_position_biased_click(rank, rel)
                qd[qid][idx]['clicked'] = clicked


def update_fold_click(fold, click_model):
    """

    Generates clicks for a given fold of the data set
    """
    with open(os.path.join('outputs', 'qd_' + fold + '.pickle'), 'rb') as handle:
        qd = pk.load(handle)

    reset(qd)
    for qid in qd.keys():
        generate_clicks(qd, qid, click_model)

    if DEBUG:
        k = 0
        for qid in qd.keys():
            for idx in qd[qid].keys():
                clicked = qd[qid][idx]["clicked"]
                rank = qd[qid][idx]["rank"]
                if rank is not None and rank > 0:
                    print("qid:" + str(qid) + "idx: " + str(idx) + " clicked: " + str(clicked) + " rank: " + str(rank))

    if not DEBUG:
        with open(os.path.join('outputs', 'qd_' + fold + '_' + click_model + '.pickle'), 'wb') as handle:
            pk.dump(qd, handle, protocol=pk.HIGHEST_PROTOCOL)
    print(fold + " click done!")

    # for randomizations, we need to access the filepath of the pickle
    return 'qd_' + fold + '_' + click_model + '.pickle'
