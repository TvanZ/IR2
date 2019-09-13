import pretrainedmodels.dataset as dataset
import pretrainedmodels.pretrained_models as prtr
import numpy as np
from optparse import OptionParser

# TODO: what is this?....figure out

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--dataset_info_path', type=str, default="./pretrainedmodels/dataset_info.txt", help='The path to the dataset info')

args = parser.parse_args()
print(args)


#
# parser = OptionParser()
#     # parameters to tune)
# parser.add_option('-d', '--dataset_info_path', dest='dataset_info_path'
# , help='frequency to perform validation')
# # TODO: before completing parameter tuning change back to False
# # TODO: otherwise, problems
# parser.add_option('-x', '--save_predictions', dest='save_predictions', default=True,
#                   help='export network predictions')
# (options, args) = parser.parse_args()
#









# args = [ some argparser stuff ]

# reading the Yahoo data
data = dataset.get_dataset_from_json_info(
"Webscope_C14_Set1",
args.dataset_info_path,
shared_resource = False,
)

# there is only a single fold

data = data.get_data_folds()[0]

pretrain_model = prtr.read_model(args.model_file,  data, 1.0)

# get scores for every document in the dataset

scores = np.dot(data.feature_matrix, pretrain_model)

# sort these scores in descending order, per query, data.doclist_ranges indicates the range of documents per query i.e. the indices it starts and ends, for query with id q: data.doclist_ranges[q:q+2]




