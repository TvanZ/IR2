import os,sys
import random
import math
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
import click_models as cm

def main():
	CLICK_MODEL_JSON = sys.argv[1]
	MODEL_NAME = 'GANoN'
	DATASET_NAME = 'set1'
	INITIAL_RANK_PATH = sys.argv[2]
	OUTPUT_PATH = sys.argv[3]
	RANK_CUT = int(sys.argv[4])
	SET_NAME = ['train','test','valid']

	with open(CLICK_MODEL_JSON) as fin:
		model_desc = json.load(fin)
		click_model = cm.loadModelFromJson(model_desc)

	for set_name in SET_NAME:
		if not os.path.exists(OUTPUT_PATH + set_name + '/'):
			os.makedirs(OUTPUT_PATH + set_name + '/')

	# process dataset from file
	train_set = data_utils.read_data(INITIAL_RANK_PATH, 'train', RANK_CUT)
	valid_set = data_utils.read_data(INITIAL_RANK_PATH, 'valid', RANK_CUT)

	full_click_list, full_exam_p_list, full_click_p_list = [],[],[]
	for ranking in train_set.initial_list[:10]:
		click_list, exam_p_list, click_p_list = click_model.sampleClicksForOneList(ranking)
		print(click_list)


if __name__ == "__main__":
	main()
