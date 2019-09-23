import numpy as np
import pickle as pk
from collections import defaultdict

DATABASE = "./database/"
DEBUG = False


def update_fold_rank(fold):
    """
    Updates the ranking of the query document data structure accordingly to ranking obtain by the SVM
    :param fold: test, train or validation
    """
    with open('qd_' + fold + '.pickle', 'rb') as handle:
        qd = pk.load(handle)

    file = open(DATABASE + fold + "/" + fold + ".trec.init_list")
    for line in file:
        line = line.split(' ')
        qid = int(line[0])
        idx = int(line[2].split('_')[-1])
        rank = int(line[3])
        qd[qid][idx]['rank'] = rank

    if DEBUG:
        print(qd[31074][32278]['rank'])
        print(qd[31074][32278]['features'][16])

    if not DEBUG:
        with open('qd_' + fold + '.pickle', 'wb') as handle:
            pk.dump(qd, handle, protocol=pk.HIGHEST_PROTOCOL)
