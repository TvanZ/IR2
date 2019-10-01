import numpy as np
import pickle as pk
from random import random
from collections import defaultdict

from GANoN.click_models import CascadeModel, SimpleModel, PositionBiasedModel

CASCADE_MODEL = "cascade_model"
SIMPLE_MODEL = "simple_model"
POSITION_BIASED_MODEL = "position_biased_model"

DATABASE = "./database/"
DEBUG = False

cascade_generator = CascadeModel()
simple_generator = SimpleModel()
position_bias_generator = PositionBiasedModel()


def generate_position_biased_click(rank, relevance_label):
    click, _, _, _ = position_bias_generator.sampleClick(rank, relevance_label)
    return click == 1


def generate_simple_click(rank, relevance_label):
    click, _, _, _ = simple_generator.sampleClick(rank, relevance_label)
    return click == 1


def generate_cascade_click(was_clicked, relevance_label):
    click, _, _, _ = cascade_generator.sampleClick(was_clicked, relevance_label)
    return click == 1


def generate_click(doc, click_model, clicked):
    if click_model == CASCADE_MODEL:
        rel = doc['label']
        return generate_cascade_click(clicked, rel)
    if click_model == SIMPLE_MODEL:
        rel = doc['label']
        rank = doc['rank']
        return generate_simple_click(rank, rel)
    if click_model == POSITION_BIASED_MODEL:
        rel = doc['label']
        rank = doc['rank']
        return generate_position_biased_click(rank, rel)
    return False


def update_fold_click(fold, click_model):
    """

    Generates clicks for a given fold of the data set
    """
    with open('qd_' + fold + '.pickle', 'rb') as handle:
        qd = pk.load(handle)

    for qid in qd.keys():
        querry = qd[qid]
        clicked = False
        for idx in querry.keys():
            doc = querry[idx]
            clicked = generate_click(doc, click_model, clicked)
            qd[qid][idx]['clicked'] = clicked

    if DEBUG:
        k = 0
        for qid in qd.keys():
            for idx in qd[qid].keys():
                clicked = qd[qid][idx]["clicked"]
                rank = qd[qid][idx]["rank"]
                if rank is not None and rank > 0 and clicked:
                    print("qid:" + str(qid) + "idx: " + str(idx) + " clicked: " + str(clicked) + " rank: " + str(rank))

    if not DEBUG:
        with open('qd_' + fold + '_' + click_model + '.pickle', 'wb') as handle:
            pk.dump(qd, handle, protocol=pk.HIGHEST_PROTOCOL)
    print(fold + " click done!")
