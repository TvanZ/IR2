import os,sys
import random
import math
import json
import numpy as np
import timeit
from functools import partial

import data_utils
import click_models as cm

def generate_clicks(n, click_model, data, features):
    x = 0
    click_log = []
    relevance_log = []
    feature_log = []
    while x < n:
        choice = random.choice(list(zip(data, features)))
        ranking, feature = choice
        clicks,_,_,_ = click_model.sampleClicksForOneList(ranking)
        if sum(clicks) < 1:
            continue
        click_log.append(clicks)
        relevance_log.append(ranking[:])
        feature_log.append(feature[:][:])
        x += sum(clicks)

    # print(f'{x} clicks generated in {len(click_log)} sessions')
    return click_log, relevance_log, feature_log

def main():
    CLICK_MODEL_JSON = sys.argv[1]
    # the folder where the input data can be found
    INPUT_DATA_PATH = sys.argv[2]
    # the folder where output should be stored
    OUTPUT_PATH = sys.argv[3]
    # how many results to show in the results page of the ranker
    # this should be equal or smaller than the rank cut when creating the data
    RANK_CUT = int(sys.argv[4])

    with open(CLICK_MODEL_JSON) as fin:
        model_desc = json.load(fin)
        click_model = cm.loadModelFromJson(model_desc)

    # process dataset from file
    train_set = data_utils.read_data(INPUT_DATA_PATH, 'train', RANK_CUT)
    click_log, relevance_log, feature_log = generate_clicks(1000000, click_model, train_set.gold_weights, train_set.featuredids)
    timeit_results = timeit.Timer(partial(generate_clicks, 1000000, click_model, train_set.gold_weights, train_set.featuredids)).repeat(10, 1)
    # print(timeit_results)
    # print(sum(timeit_results)/10)

if __name__ == '__main__':
    main()

# To test this code: python generate_click_data.py sm_1_0_1_1.json ../Baselines/Unbiased-Learning-to-Rank-with-Unbiased-Propensity-Estimation-master/Input_Data/ /Test_Output/ 10
