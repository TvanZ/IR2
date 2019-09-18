import os,sys
import random
import math
import json
import numpy as np
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
import click_models as cm

def main():
	# the click model in json format as exported when creating a model with the click_models.py
	CLICK_MODEL_JSON = sys.argv[1]
	MODEL_NAME = 'GANoN'
	DATASET_NAME = 'set1'
	# the folder where the input data can be found
	INPUT_DATA_PATH = sys.argv[2]
	# the folder where output should be stored
	OUTPUT_PATH = sys.argv[3]
	# how many results to show in the results page of the ranker
	# this should be equal or smaller than the rank cut when creating the data
	RANK_CUT = int(sys.argv[4])
	SET_NAME = ['train','test','valid']

	with open(CLICK_MODEL_JSON) as fin:
		model_desc = json.load(fin)
		click_model = cm.loadModelFromJson(model_desc)

	for set_name in SET_NAME:
		if not os.path.exists(OUTPUT_PATH + set_name + '/'):
			os.makedirs(OUTPUT_PATH + set_name + '/')

	# process dataset from file
	train_set = data_utils.read_data(INPUT_DATA_PATH, 'train', RANK_CUT)
	valid_set = data_utils.read_data(INPUT_DATA_PATH, 'valid', RANK_CUT)

	full_click_list, full_exam_p_list, full_click_p_list = [],[],[]
	for ranking in train_set.initial_list[:10]:
		click_list, exam_p_list, click_p_list = click_model.sampleClicksForOneList(ranking)
		print(click_list)

class GAN:
    def __init__(self, click_model, rank_list_size,
    g_settings, d_settings,
	batch_size, hparam_str, forward_only=False, feed_previous = False):
    	self.click_model = click_model
    	self.rank_list_size = rank_list_size
		self.batch_size = batch_size

		g = g_settings
		d = d_settings

		self.G = Generator(g.input_size, g.hidden_size, g.output_size, g.fn)
		self.D = Disriminator(d.input_size, d.hidden_size, d.output_size, d.fn)

class Generator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, fn):
		super(Generator, self).__init__()
		self.g = nn.Sequential(
   			nn.Linear(input_size, hidden_size),
   			fn,
   			nn.Linear(hidden_size, hidden_size),
   			fn,
   			nn.Linear(hidden_size, output_size),
   			# binairy aproximator here
   		)

	def forward(self, x):
		return self.g(x)

class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, fn):
		super(Discriminator, self).__init__()
		self.d = nn.Sequential(
   			nn.Linear(input_size, hidden_size),
   			fn,
   			nn.Linear(hidden_size, hidden_size),
   			fn,
   			nn.Linear(hidden_size, output_size),
   			fn
   		)

	def forward(self, x):
		return self.d(x)


if __name__ == "__main__":
	main()
