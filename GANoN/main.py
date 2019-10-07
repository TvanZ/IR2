import os,sys
import random
import math
import json
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
import click_models as cm
from generate_click_data import generate_clicks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
standard_dtype = torch.double
torch.set_default_dtype(standard_dtype)

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
	BATCH_SIZE = 512
	EMBED_SIZE = 700
	with open(CLICK_MODEL_JSON) as fin:
		model_desc = json.load(fin)
		click_model = cm.loadModelFromJson(model_desc)

	for set_name in SET_NAME:
		if not os.path.exists(OUTPUT_PATH + set_name + '/'):
			os.makedirs(OUTPUT_PATH + set_name + '/')

	# Determine if a gpu is available
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# process dataset from file
	train_set = data_utils.read_data(INPUT_DATA_PATH, 'train', RANK_CUT)
	valid_set = data_utils.read_data(INPUT_DATA_PATH, 'valid', RANK_CUT)

	GENERATE = False
	#Generate Clicks
	if GENERATE:
		click_logs, rankings, features = generate_clicks(1000000, click_model, train_set.gold_weights, train_set.featuredids)
		print("Clicks generated!")
		zipped_all = zip(click_logs,rankings,features)
		pickle.dump(zipped_all, open("clicks.p","wb"))
		print("Saved a pickle!")
	else:
		pickled_clicks = pickle.load(open("clicks.p", "rb"))
		click_logs, rankings, features = zip(*pickled_clicks)
		print("opened a pickle!")

	PBM_settings_Hardcoded = {
		"g":{
		'generator': Generator1,
		'input_size' : 2,
		'hidden_size' : 2,
		'output_size' : 10,
		'fn': nn.Sigmoid
		},
		"d":{
		'input_size': 20,
		'hidden_size': 50,
		'output_size': 1,
		'fn': nn.Sigmoid
		},
		"feature": False
	}

	PBM_settings_Learned_Relevance = {
		"g":{
		'generator': Generator2,
		'input_size' : 1,
		'hidden_size' : 20,
		'output_size' : 1,
		'fn': nn.Sigmoid,
		},
		"d":{
		'input_size': 20,
		'hidden_size': 50,
		'output_size': 1,
		'fn': nn.Sigmoid
		},
		"feature": False
	}
	PBM_settings_Learned_Features = {
		"g":{
		'generator': Generator2,
		'input_size' : 700,
		'hidden_size' : 300,
		'output_size' : 1,
		'fn': nn.Sigmoid
		},
		"d":{
		'input_size': 7010,
		'hidden_size': 300,
		'output_size': 1,
		'fn': nn.Sigmoid
		},
		"feature": True
	}



	def run(model_settings, BATCH_SIZE = BATCH_SIZE, EMBED_SIZE = EMBED_SIZE, RANK_CUT = RANK_CUT, click_logs = click_logs, rankings = rankings, features = features):

		g_optimizer = optim.Adam
		d_optimizer = optim.Adam

		gan = GAN(click_model, 10, BATCH_SIZE, model_settings, g_optimizer, d_optimizer)
		gan.to(device)

		num_epochs = 100
		real_errors, fake_errors, g_errors = [],[],[]
		for epoch in range(num_epochs):
			for mini_batch in get_minibatch(BATCH_SIZE, EMBED_SIZE, RANK_CUT, list(zip(click_logs, rankings, features))):
				click_logs_T, rankings_T, features_T = mini_batch
				if not model_settings['feature']:
					real_error, fake_error, g_error = gan.train_with_log(click_logs_T, rankings_T)
				else:
					real_error, fake_error, g_error = gan.train(click_logs_T, features_T)
			_, rankings_tensor, features_tensor = mini_batch
			with torch.no_grad():
				if model_settings['feature']:
					rankings_tensor = features_tensor
				observations, clicks = gan.G(rankings_tensor) 
			observations=torch.mean(observations,dim=0)

			print('observations:', observations)
			print(f"[{epoch + 1}/{num_epochs}] | Loss D: {(real_error + fake_error)/2} | Loss G: {g_error}")
			real_errors.append(real_error)
			fake_errors.append(fake_error)
			g_errors.append(g_error)

		print('real_errors', real_errors)
		print('fake_errors', fake_errors)
		print('g_errors', g_errors)

		return real_errors, fake_errors, g_errors


	real,fake, g = run(PBM_settings_Hardcoded)
	print(real, fake, g)


def get_minibatch(batch_size, embed_size, rank_cut, data):
	# data should be a zipped list of click_logs and rankings
	random.shuffle(data)
	while len(data) > 0:
		batch = data[:batch_size]
		data = data[batch_size:]
		click_logs, rankings, features = zip(*batch)
		click_logs = list(click_logs)
		rankings = list(rankings)
		features = list(features)

		# add padding
		for i in range(len(batch)):
			if len(click_logs[i]) < rank_cut:
				len_dif = rank_cut - len(click_logs[i])
				# print(f"Padding click log of length {len(click_logs[i])} with {len_dif} items")
				click_logs[i] += [0] * len_dif
				rankings[i] += [-1.] * len_dif
				for j in range(0,len_dif):
					features[i].append([0.0 for _ in range(embed_size)])



		click_logs_tensor = torch.Tensor(click_logs).to(device)
		rankings_tensor = torch.Tensor(rankings).to(device)[:,:,None]
		feature_tensor = torch.Tensor(features).to(device)
		yield click_logs_tensor, rankings_tensor, feature_tensor

class GAN:
	def __init__(self, click_model, rank_list_size, batch_size,
	model_settings, g_optimizer, d_optimizer, criterion = nn.BCEWithLogitsLoss(),
	forward_only=False, feed_previous = False):
		self.click_model = click_model
		self.rank_list_size = rank_list_size
		self.batch_size = batch_size

		g = model_settings["g"]
		d = model_settings["d"]
		generator = g["generator"]

		self.G = generator(g['input_size'], g['hidden_size'], g['output_size'], g['fn'], rank_list_size)
		self.D = Discriminator(d['input_size'], d['hidden_size'], d['output_size'], d['fn'])
		self.d_optimizer = d_optimizer(self.D.parameters(),lr=0.0001)
		self.g_optimizer = g_optimizer(self.G.parameters(),lr=0.0005)
		self.criterion = criterion
		self.errors = []

	def train(self, click_logs, rankings):
		# first train the discriminator
		self.D.zero_grad()
		# train on real data
		rankings_1, rankings_2 = torch.chunk(rankings, 2)
		click_logs_1, click_logs_2 = torch.chunk(click_logs,2)
		true_data = torch.cat((click_logs_1, rankings_1.view(rankings_1.size()[0], -1)), dim=1)
		real_decision = self.D(true_data)
		real_error = self.criterion(real_decision, torch.ones(real_decision.size(), device=device)  + torch.rand(real_decision.size(), device=device) *.3 - 0.2)
		real_error.backward()
		# train on fake data
		fake_observations, fake_data = self.G(rankings_2)
		fake_all = torch.cat((fake_data.detach(), rankings_2.view(rankings_2.size()[0], -1)), dim=1)
		fake_decision = self.D(fake_all) # detach the fake data so the generator does not get updated here
		fake_error = self.criterion(fake_decision, torch.zeros(fake_decision.size(), device=device) + torch.rand(fake_decision.size(), device=device) *.3)
		fake_error.backward()
		self.d_optimizer.step()

		# then train the generator
		self.G.zero_grad()
		fake2_all = torch.cat((fake_data, rankings_1.view(rankings_1.size()[0], -1)), dim=1)
		g_fake_decision = self.D(fake2_all)
		
		g_error = self.criterion(g_fake_decision, torch.ones(g_fake_decision.size(), device=device))
		g_error.backward()
		self.g_optimizer.step()

		return real_error.item(), fake_error.item(), g_error.item()

	def to(self, device):
		self.G.to(device)
		self.D.to(device)

	def train_with_log(self, click_logs, rankings, criterion=nn.BCELoss()):

		self.G.zero_grad()
		fake_observations, fake_data = self.G(rankings)
		
		g_error = criterion(fake_data, click_logs)
		g_error.backward()
		self.g_optimizer.step()

		return 0, 0, g_error.item()




### GENERATOR FUNCTIONS ======================================================

## Generator with Hardcoded module
class Generator1(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, fn, rank_cut):
		super(Generator1, self).__init__()
		self.binary_approximator = BinaryApproximator(rank_cut)
		self.sampler = ClickSampler()

	def forward(self, relevance_scores):
		batch_size = relevance_scores.size()[0]
		rank_size = relevance_scores.size()[1]
		random_noise = torch.rand((batch_size, rank_size), device=device)
		observation_scores = self.binary_approximator(random_noise)
		fake_click_logs = self.sampler(observation_scores, relevance_scores.view(batch_size, -1))
		return observation_scores, fake_click_logs

## Generator with Learned module
class Generator2(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, fn, rank_cut):
		super(Generator2, self).__init__()
		self.binary_approximator = BinaryApproximator(rank_cut)
		self.g = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, output_size)
		)
		self.binary_approximator_rel = BinaryApproximator(rank_cut, alpha = 1)

	def forward(self, relevance_scores):
		batch_size = relevance_scores.size()[0]
		rank_size = relevance_scores.size()[1]
		random_noise = torch.rand((batch_size, rank_size), device=device)
		random_noise2 = torch.rand((batch_size, rank_size), device=device)
		observation_scores = self.binary_approximator(random_noise)
		alpha = self.g(relevance_scores).squeeze(dim=2)
		relevance_understanding = self.binary_approximator_rel(random_noise2, alpha)
		fake_click_logs = observation_scores * relevance_understanding.view(batch_size, -1)
		return observation_scores, fake_click_logs

## ==== DISCRIMINATOR FUNCTION ==================================================

class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, fn):
		super(Discriminator, self).__init__()
		self.d = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, x):
		return self.d(x)



## HELPER FUNCTIONS ===========================================================

class BinaryApproximator(nn.Module):
	def __init__(self, input_size, gamma = -0.1, zeta = 1.1, alpha = None, beta = None):
		super(BinaryApproximator, self).__init__()
		# Variability of what should be trained
		if alpha:
			self.alpha = float(alpha)
		else:
			self.alpha = nn.Parameter(torch.randn((1,input_size)))
		if beta:    	    
			self.beta = float(beta)
		else:
			self.beta = nn.Parameter(torch.rand((1,input_size)))
		self.gamma = gamma
		self.zeta = zeta

		#if BinaryApproximator is used by a neural net put alpha on some value.
	def forward(self, u, alpha = None):
		if alpha is not None:
			s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(alpha)))/F.softplus(self.beta))
		else:
			s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(self.alpha))/F.softplus(self.beta)))
		mean_s = s * (self.zeta - self.gamma) + self.gamma
		binarysize = mean_s.size()
		z = torch.min(torch.ones(binarysize, device=device),(torch.max(torch.zeros(binarysize, device=device),mean_s)))
		return z



class ClickSampler(nn.Module):
	def __init__(self, relevance_threshold = 3):
		super(ClickSampler, self).__init__()
		self.relevance_threshold = relevance_threshold

	def forward(self, observation_scores, rankings):
		click_probabilities = torch.ones(rankings.size()).to(device)
		click_probabilities[rankings < self.relevance_threshold] = 0.1
		click_logs = torch.bernoulli(click_probabilities)
		return click_logs * observation_scores

if __name__ == "__main__":
	if device == 'cuda':
		torch.cuda.empty_cache()
	main()
