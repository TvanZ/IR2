import pretrainedmodels.dataset as dataset
import pretrainedmodels.pretrained_models as prtr
import numpy as np

# TODO: what is this?....figure out
args = [ some argparser stuff ]

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