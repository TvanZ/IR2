from Baselines.EM_regression.em_regression import em_regression
from Baselines.EM_regression.read_database import read_fold
import pickle as pk

from Baselines.EM_regression.update_click import update_fold_click
from Baselines.EM_regression.update_rank import update_fold_rank

import matplotlib.pyplot as plt
import numpy as np
from GANoN.click_models import SimpleModel, CascadeModel, PositionBiasedModel

CASCADE_MODEL = "cascade_model"
SIMPLE_MODEL = "simple_model"
POSITION_BIASED_MODEL = "position_biased_model"


def save_qd(click_model):
    fold = ".train.txt"
    query_document_train = read_fold(fold)
    with open('qd_train_' + click_model + '.pickle', 'wb') as handle:
        pk.dump(query_document_train, handle, protocol=pk.HIGHEST_PROTOCOL)
    print("Train done!")

    fold = ".valid.txt"
    query_document_valid = read_fold(fold)
    with open('qd_valid_' + click_model + '.pickle', 'wb') as handle:
        pk.dump(query_document_valid, handle, protocol=pk.HIGHEST_PROTOCOL)
    print("Validate done!")

    fold = ".test.txt"
    query_document_test = read_fold(fold)
    with open('qd_test_' + click_model + '.pickle', 'wb') as handle:
        pk.dump(query_document_test, handle, protocol=pk.HIGHEST_PROTOCOL)
    print("Test done!")


def update_clicks(click_model):
    fold = "train"
    update_fold_click(fold, click_model)
    fold = "valid"
    update_fold_click(fold, click_model)
    fold = "test"
    update_fold_click(fold, click_model)


def update_ranks(click_model):
    fold = "train"
    update_fold_rank(fold, click_model)
    fold = "valid"
    update_fold_rank(fold, click_model)
    fold = "test"
    update_fold_rank(fold, click_model)


def process_dataset(click_model):
    save_qd(click_model)
    update_ranks(click_model)
    update_clicks(click_model)


def run_em(click_model):
    with open('qd_' + 'train_' + click_model + '.pickle', 'rb') as handle:
        qd = pk.load(handle)
    Theta, preds, p_e, p_r = em_regression(qd, click_model)
    print_theta(click_model)


def get_actual_distribution(click_model, top_n=10):
    if click_model == SIMPLE_MODEL:
        simple_model = SimpleModel()
        return [simple_model.getExamProb(i) for i in range(0, top_n)]
    if click_model == POSITION_BIASED_MODEL:
        position_biased_model = PositionBiasedModel()
        return [position_biased_model.getExamProb(i) for i in range(0, top_n)]
    return []


def get_cascade_examination_prob():
    with open('qd_' + 'train_' + CASCADE_MODEL + '.pickle', 'rb') as handle:
        qd = pk.load(handle)

    position_bias = np.zeros(10)
    counter = 0
    for qid in qd.keys():
        counter += 1
        query = qd[qid]
        c_k = 0
        for idx in query.keys():
            clicked = query[idx]['clicked']
            rank = query[idx]['rank']
            if clicked:
                c_k += 1
                for i in range(rank):
                    position_bias[i] += 1

    position_bias = position_bias / counter
    return position_bias


def get_kl_divergence(click_model):
    with open('results/theta_' + click_model + '.pickle', 'rb') as handle:
        theta = pk.load(handle)
    if click_model == CASCADE_MODEL:
        y = get_cascade_examination_prob()
    else:
        y = np.array(get_actual_distribution(click_model))

    sum_theta = sum(theta)
    theta /= sum_theta

    sum_y = sum(y)
    y /= sum_y

    kl = 0

    for i, e in enumerate(theta):
        kl += theta[i] * np.log(theta[i] / y[i])

    print(click_model + ": " + str(kl))

def mse(click_model):
    with open('results/theta_' + click_model + '.pickle', 'rb') as handle:
        theta = pk.load(handle)
    if click_model == CASCADE_MODEL:
        y = get_cascade_examination_prob()
    else:
        y = np.array(get_actual_distribution(click_model))

    mse_t =  sum(abs(theta - y))/10

    print(click_model + ": " + str(mse_t))

def print_theta(click_model):
    with open('results/theta_' + click_model + '.pickle', 'rb') as handle:
        theta = pk.load(handle)

    print(theta)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if click_model == CASCADE_MODEL:
        y = get_cascade_examination_prob()
    else:
        y = get_actual_distribution(click_model)
    line, = plt.plot(x, theta)
    line.set_label("EM")
    line, = plt.plot(x, y)
    line.set_label(click_model + " distribution")
    plt.ylabel("Position bias")
    plt.xlabel("Ranking")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mse(CASCADE_MODEL)
