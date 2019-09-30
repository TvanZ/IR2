from Baselines.EM_regression.em_regression import em_regression
from Baselines.EM_regression.read_database import read_fold
import pickle as pk

from Baselines.EM_regression.update_click import update_fold_click
from Baselines.EM_regression.update_rank import update_fold_rank

import matplotlib.pyplot as plt

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


def get_actual_distribution(click_model, top_n):
    if click_model == SIMPLE_MODEL:
        simple_model = SimpleModel()
        return [simple_model.getExamProb(i) for i in range(1, top_n + 1)]
    if click_model == CASCADE_MODEL:
        cascade_model = CascadeModel()
        return [cascade_model.getExamProb(i, 3) for i in range(1, top_n + 1)]
    if click_model == POSITION_BIASED_MODEL:
        position_biased_model = PositionBiasedModel()
        return [position_biased_model.getExamProb(i) for i in range(1, top_n + 1)]
    return []


def print_theta(clcik_model):
    with open('results/theta_' + clcik_model + '.pickle', 'rb') as handle:
        theta = pk.load(handle)

    print(theta)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    y = get_actual_distribution(clcik_model, 10)

    line, = plt.plot(x, theta)
    line.set_label("EM")
    line, = plt.plot(x, y)
    line.set_label(clcik_model + " distribution")
    plt.ylabel("Position bias")
    plt.xlabel("Ranking")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_em(CASCADE_MODEL)
