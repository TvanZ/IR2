import os,sys
import random
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
import click_models as cm
from generate_click_data import generate_clicks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

	click_logs, rankings = generate_clicks(1000000, click_model, train_set.gold_weights)

	# full_click_list, full_exam_p_list, full_click_p_list = [],[],[]
	# for ranking in train_set.gold_weights[:10]:
	# 	print("Rank")
	# 	print(ranking)
	# 	click_list, exam_p_list, click_p_list, exam_list = click_model.sampleClicksForOneList(ranking)
	# 	print("clicks")
	# 	print(click_list)
	# 	print("exam")
	# 	print(exam_list)

	g_settings = {
		'input_size': 2,
		'hidden_size': 2,
		'hidden_size_2': 10,
		'output_size': 10,
		'fn': nn.Sigmoid,
		'steps': 20
	}

	d_settings = {
		'input_size': 10,
		'hidden_size': 100,
		'output_size': 1,
		'fn': nn.Sigmoid,
		'steps': 20
	}

	g_optimizer = optim.Adam
	d_optimizer = optim.Adam

	gan = GAN(click_model, 10, BATCH_SIZE, g_settings, d_settings, g_optimizer, d_optimizer)
	gan.to(device)

	num_epochs = 500
	real_errors, fake_errors, g_errors = [],[],[]
	for epoch in range(num_epochs):
		# print('epoch', epoch)
		for mini_batch in get_minibatch(BATCH_SIZE, list(zip(click_logs, rankings))):
			real_error, fake_error, g_error = gan.train(*mini_batch)
		_, rankings_tensor = mini_batch
		observations, clicks = gan.G(rankings_tensor[-1:,:])
		print('input rankings:', rankings_tensor[-1:,:])
		print('observations:', observations)
		print('clicks:', clicks)
		print(f"[{epoch + 1}/{num_epochs}] | Loss D: {real_error + fake_error} | Loss G: {g_error}")
		# print('real_error', real_error)
		# print('fake_error', fake_error)
		# print('g_error', g_error)
		real_errors.append(real_error)
		fake_errors.append(fake_error)
		g_errors.append(g_error)

	print('real_errors', real_errors)
	print('fake_errors', fake_errors)
	print('g_errors', g_errors)

def get_minibatch(batch_size, data):
	# data should be a zipped list of click_logs and rankings
	random.shuffle(data)
	while len(data) > 0:
		batch = data[:batch_size]
		data = data[batch_size:]
		click_logs, rankings = zip(*batch)
		click_logs = list(click_logs)
		rankings = list(rankings)
		# print(click_logs)
		# print(rankings)
		# print(len(click_logs), len(rankings))
		# add padding
		for i in range(len(batch)):
			if len(click_logs[i]) < 10:
				len_dif = 10 - len(click_logs[i])
				# print(f"Padding click log of length {len(click_logs[i])} with {len_dif} items")
				click_logs[i] += [0] * len_dif
			if len(rankings[i]) < 10:
				rankings[i] += [-1.] * len_dif

		click_logs_tensor = torch.FloatTensor(click_logs).to(device)
		# print(click_logs_tensor)
		rankings_tensor = torch.FloatTensor(rankings).to(device)
		# print(rankings_tensor)
		# print('batch delivered')
		yield click_logs_tensor, rankings_tensor

class GAN:
	def __init__(self, click_model, rank_list_size, batch_size,
	g_settings, d_settings, g_optimizer, d_optimizer, criterion = nn.BCELoss(),
	forward_only=False, feed_previous = False):
		self.click_model = click_model
		self.rank_list_size = rank_list_size
		self.batch_size = batch_size

		g = g_settings
		d = d_settings

		self.G = Generator(g['input_size'], g['hidden_size'], g['hidden_size_2'], g['output_size'], g['fn'], rank_list_size)
		self.D = Discriminator(d['input_size'], d['hidden_size'], d['output_size'], d['fn'])
		self.d_optimizer = d_optimizer(self.D.parameters())
		self.g_optimizer = g_optimizer(self.G.parameters())
		self.criterion = criterion
		self.g_steps = g['steps']
		self.d_steps = d['steps']
		self.errors = []

	def train(self, click_logs, rankings):
		half = int(math.floor(len(click_logs)/2))
		# for d_index in range(self.d_steps):
		# first train the discriminator
		self.D.zero_grad()
		# train on real data
		real_decision = self.D(click_logs[:half,:])
		# print('real_decision:', real_decision)
		real_error = self.criterion(real_decision, torch.ones(real_decision.size(), device=device))
		real_error.backward()
		# train on fake data
		fake_observations, fake_data = self.G(rankings[half:,:])
		# print('fake_data:', fake_data)
		fake_decision = self.D(fake_data.detach())
		# print('fake_decision:', fake_decision)
		fake_error = self.criterion(fake_decision, torch.zeros(fake_decision.size(), device=device))
		fake_error.backward()
		self.d_optimizer.step()

		# for g_index in range(self.g_steps):
		# then train the generator
		self.G.zero_grad()
		# generate fake examples
		# g_fake_observations, g_fake_data = self.G(rankings[:half,:,:])
		g_fake_decision = self.D(fake_data)
		# print('g_fake_decision:', g_fake_decision)
		g_error = self.criterion(g_fake_decision, torch.ones(g_fake_decision.size(), device=device))
		g_error.backward()
		self.g_optimizer.step()
		return real_error.item(), fake_error.item(), g_error.item()

	def to(self, device):
		self.G.to(device)
		self.D.to(device)

class Generator(nn.Module):
	def __init__(self, input_size, hidden_size, hidden_size_2, output_size, fn, rank_cut):
		super(Generator, self).__init__()
		# self.observance_GRU = nn.GRU(1, hidden_size, batch_first=True)
		# self.observance = nn.Sequential(
		# 	fn(),
		# 	nn.Linear(hidden_size, hidden_size),
		# 	fn(),
		# 	nn.Linear(hidden_size, 1),
		# 	fn(),
		# 	BinaryApproximator()
		# )
		# self.relevance_GRU = nn.GRU(2, hidden_size_2, batch_first=True)
		# self.relevance = nn.Sequential(
		# 	fn(),
		# 	nn.Linear(hidden_size_2, hidden_size_2),
		# 	fn(),
		# 	nn.Linear(hidden_size_2, 1),
		# 	fn(),
		# 	BinaryApproximator()
		# )
		self.binary_approximator = BinaryApproximator(rank_cut)
		# self.input_size = input_size
		# self.hidden_size = hidden_size
		# self.hidden_size_2 = hidden_size_2
		self.sampler = ClickSampler()

	def forward(self, relevance_scores):
		batch_size = relevance_scores.size()[0]
		rank_size = relevance_scores.size()[1]
		random_noise = torch.rand((batch_size, rank_size), device=device)
		# h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
		# observation_GRU,_ = self.observance_GRU(random_noise, h_0)
		# observation_scores = self.observance(observation_GRU).squeeze(dim=2)
		observation_scores = self.binary_approximator(random_noise)
		fake_click_logs = self.sampler(observation_scores, relevance_scores)
		# print('observations_scores inside generator:', observation_scores)
		# modified_relevance = torch.cat((observation_scores,relevance_scores[:,:,None]), dim=2)
		# modified_relevance = (observation_scores * relevance_scores)[:,:,None]
		# print('modified relevance:', modified_relevance)
		# h_0 = torch.zeros(1, batch_size, self.hidden_size_2).to(device)
		# fake_click_logs, _ = self.relevance_GRU(modified_relevance, h_0)
		# fake_click_logs = self.relevance(fake_click_logs).squeeze(dim=2)
		# click_scores = self.relevance(relevance_scores)
		# fake_click_logs = observation_scores * click_scores
		return observation_scores, fake_click_logs

class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, fn):
		super(Discriminator, self).__init__()
		self.d = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, output_size),
			fn()
		)

	def forward(self, x):
		return self.d(x)

# class BinaryApproximator(nn.Module):
# 	def __init__(self, log_alpha = 0, beta = 0.5, gamma = -0.1, zeta = 1.1):
# 		super(BinaryApproximator, self).__init__()
# 		self.log_alpha = log_alpha
# 		self.beta = beta
# 		self.gamma = gamma
# 		self.zeta = zeta

# 	def forward(self, u):
# 		s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha)/self.beta)
# 		mean_s = s * (self.zeta - self.gamma) + self.gamma
# 		binarysize = mean_s.size()
# 		z = torch.min(torch.ones(binarysize, device=device),(torch.max(torch.zeros(binarysize, device=device),mean_s)))
# 		return z

class BinaryApproximator(nn.Module):
	def __init__(self, input_size, gamma = -0.1, zeta = 1.1):
		super(BinaryApproximator, self).__init__()
		self.log_alpha = nn.Parameter(torch.rand((1,input_size)))
		self.beta = nn.Parameter(torch.rand((1,input_size)))
		self.gamma = gamma
		self.zeta = zeta

	def forward(self, u):
		s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha)/torch.exp(self.beta))
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
	main()
