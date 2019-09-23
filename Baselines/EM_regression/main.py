from Baselines.EM_regression.em_regression import em_regression
from Baselines.EM_regression.read_database import read_fold
import pickle as pk

from Baselines.EM_regression.update_click import update_fold_click
from Baselines.EM_regression.update_rank import update_fold_rank


def save_qd():
    fold = ".train.txt"
    query_document_train = read_fold(fold)
    with open('qd_train.pickle', 'wb') as handle:
        pk.dump(query_document_train, handle, protocol=pk.HIGHEST_PROTOCOL)

    fold = ".valid.txt"
    query_document_valid = read_fold(fold)

    with open('qd_valid.pickle', 'wb') as handle:
        pk.dump(query_document_valid, handle, protocol=pk.HIGHEST_PROTOCOL)

    fold = ".test.txt"
    query_document_test = read_fold(fold)

    with open('qd_test.pickle', 'wb') as handle:
        pk.dump(query_document_test, handle, protocol=pk.HIGHEST_PROTOCOL)


def update_clicks():
    fold = "train"
    update_fold_click(fold)
    fold = "valid"
    update_fold_click(fold)
    fold = "test"
    update_fold_click(fold)


def update_ranks():
    fold = "train"
    update_fold_rank(fold)
    fold = "valid"
    update_fold_rank(fold)
    fold = "test"
    update_fold_rank(fold)


def process_dataset():
    save_qd()
    update_ranks()
    update_clicks()


def run_em():
    with open('qd_' + 'train' + '.pickle', 'rb') as handle:
        qd = pk.load(handle)

    Theta, preds, p_e, p_r = em_regression(qd)
    print(Theta)

def print_theta():
    with open('results/theta.pickle', 'rb') as handle:
        theta = pk.load(handle)
    print(theta)

if __name__ == "__main__":
    run_em()
