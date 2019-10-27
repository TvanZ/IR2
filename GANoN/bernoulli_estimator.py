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
import torch.multiprocessing as mp
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BinaryApproximator(nn.Module):
	def __init__(self, coins, gamma = -0.1, zeta = 1.1):
		super(BinaryApproximator, self).__init__()
		self.alpha = nn.Parameter(torch.zeros((1,coins))-0.5)
		self.beta = nn.Parameter(torch.zeros((1,coins))-0.5)
		self.gamma = gamma
		self.zeta = zeta
		# -0.4327521295671885 <- softplus-inverse(0.5)

		#if BinaryApproximator is used by a neural net put alpha on some value.
	def forward(self, u):
		s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(self.alpha))/F.softplus(self.beta)))
		mean_s = s * (self.zeta - self.gamma) + self.gamma
		binarysize = mean_s.size()
		z = torch.min(torch.ones(binarysize, device=device),(torch.max(torch.zeros(binarysize, device=device),mean_s)))
		return z

def CDFKuma(alpha, beta, threshold = 0.5):
	assert type(beta) == float or alpha.size() == beta.size()
	t = torch.zeros(alpha.size()).to(device) + threshold
	return 1-torch.pow(1-torch.pow(t, F.softplus(alpha)), F.softplus(beta))

class Discriminator(nn.Module):
	def __init__(self, coins, hidden_size, output_size, fn):
		super(Discriminator, self).__init__()
		self.d = nn.Sequential(
			nn.Linear(coins, hidden_size),
			fn(),
			nn.Linear(hidden_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, throws):
		return self.d(throws)

def LogLikelihood(observation_alpha, observation_beta, probabilities):

	p_throw = 1-CDFKuma(observation_alpha, observation_beta)
	likelihood = torch.pow(p_throw,probabilities)*torch.pow(1-p_throw,1-probabilities)

	return torch.sum(torch.log(likelihood))

def train(p):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)

	batch_size = 10000
	coins = 1
	throws = 10

	weights = torch.tensor([p]).to(device)
	data = (torch.ones((batch_size,coins*throws)).bernoulli_(p=weights)).to(device).detach()

	baf = BinaryApproximator(coins).to(device)
	d = Discriminator(coins*throws, 64, 1, nn.RReLU).to(device)
	optimizer = optim.Adam(baf.parameters(), lr=1e-2)
	optimizerd = optim.Adam(d.parameters(), lr=1e-4)

	num_epochs = 5000

	criterion = nn.BCEWithLogitsLoss()
	best_score = 100
	best_prob = 0
	epoch_nr = 0

	for epoch in range(num_epochs):
		noise = torch.rand((batch_size, coins*throws), device=device)
		fake_throws = baf(noise)
		# train discriminator
		prediction_real = d(data)
		prediction_fake = d(fake_throws.detach())

		d.zero_grad()
		real_loss = criterion(prediction_real, torch.ones((batch_size,1)).to(device))
		real_loss.backward()
		optimizerd.step()
		d.zero_grad()
		fake_loss = criterion(prediction_fake, torch.zeros((batch_size,1)).to(device))
		fake_loss.backward()
		optimizerd.step()

		# train generator
		prediction_fake = d(fake_throws)
		baf.zero_grad()
		generator_loss = criterion(prediction_fake, torch.ones((batch_size,1)).to(device))
		prob = 1-CDFKuma(baf.alpha.data, baf.beta.data)
		generator_loss.backward()
		optimizer.step()

		ll = -LogLikelihood(baf.alpha.data, baf.beta.data, weights)
		if ll.item() < best_score:
			best_score = ll.item()
			best_prob = prob.item()
			epoch_nr = epoch

	return p, epoch_nr, best_score, best_prob


if __name__ == "__main__":
	mp.set_start_method('spawn')
	p_values = [p/50 for p in range(51)]
	random.shuffle(p_values)

	with mp.Pool(processes=5) as pool:
		results = pool.map(train,p_values)

	results = sorted(results, key=lambda tup: tup[0])

	with open('epochs-bernoulli-values.txt', 'w') as fout:
		for result in results:
			fout.write(str(result) +'\n')

	with open('graph-values.txt', 'w') as fout:
		for result in results:
			fout.write(f"({result[1],result[0]})")








