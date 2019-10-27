from Baselines.randomizations.random_utils import *
import numpy as np

def calc_values(pickle_filepath):
    my_results = unpickle_results(pickle_filepath)

    # calculating position _bias
    theta = np.zeros(10)
    total = 0
    for qid in my_results.keys():
        query = my_results[qid]
        total += 1
        for doc in query:
            rank = doc[1]
            clicked = doc[2]
            if clicked:
                theta[rank - 1] += 1


    theta /= (total*100)
    print(theta)
    qd_test = unpickle_results('./click_models/qd_test_simple_model.pickle')
    counter = 0
    probs =  []
    for qid in qd_test.keys():
        query = qd_test[qid]
        counter+=1
        for idx in query.keys():
            rank = query[idx]['rank']
            if rank is not None:
                rel = query[idx]['rank']/4
                clicked = query[idx]['clicked']
                prob = rel * theta[rank-1]
                if not clicked:
                    prob = 1 - prob
                probs.append(np.log(prob))

    total = 0
    for prob in probs:
        total += prob
    total /= counter
    print(total)

calc_values('randomized_results.pickle')