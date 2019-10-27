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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
import click_models as cm
from generate_click_data import generate_clicks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
standard_dtype = torch.double
torch.set_default_dtype(standard_dtype)

def run(params):
	bt_s,lr_d,lr_g,hs_d,hs_g,emb_d = params
	BATCH_SIZE = bt_s

	click_logs, rankings, features, v_click_logs, v_rankings, v_features = open_data()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	standard_dtype = torch.double
	torch.set_default_dtype(standard_dtype)

	model_filename = f"./models/{bt_s}-{lr_d}-{lr_g}-{hs_d}-{hs_g}-{emb_d}-data-5000.p"
	model_scores_file =f"./scores/{bt_s}-{lr_d}-{lr_g}-{hs_d}-{hs_g}-{emb_d}-data-5000.txt"

	g_optimizer = optim.Adam
	d_optimizer = optim.Adam

	model_settings = {
		"g":{
		'generator': Generator,
		'input_size' : 700,
		'hidden_size' : hs_g,
		'output_size' : 2,
		'fn': nn.RReLU,
		'lr': lr_g
		},
		"d":{
		'hidden_size': hs_d,
		'output_size': 1,
		'fn': nn.Sigmoid,
		'feature_size': 700,
		'embed_size': emb_d,
		'lr': lr_d
		},
		"feature": True
	}

	num_epochs = 5000
	current_best = np.inf
	obs_ll = 0
	best_epoch = 0
	epoch = 0
	real_errors, fake_errors, g_errors, eval_errors, obs_lls = [],[],[],[],[]

	gan = GAN(10, BATCH_SIZE, model_settings, g_optimizer, d_optimizer)
	# if os.path.isfile(model_scores_file):
	# 	ckpt = torch.load(model_filename)
	# 	gan.G.load_state_dict(ckpt["g_state_dict"])
	# 	gan.D.load_state_dict(ckpt["d_state_dict"])
	# 	current_best = ckpt["best_eval"]
	# 	epoch = ckpt["best_epoch"]
	# 	best_epoch = ckpt["best_epoch"]
	#   real_errors, fake_errors, g_errors, eval_errors, obs_lls = ckpt["errors"]
	gan.to(device)


	while best_epoch >= epoch - 20 or epoch < num_epochs:
		# Train the model
		for mini_batch in get_minibatch(BATCH_SIZE, 700, 10, list(zip(click_logs, rankings, features))):
			click_logs_T, rankings_T, features_T = mini_batch
			real_error, fake_error, g_error = gan.train(click_logs_T, features_T)
			# observation_alphas = gan.G.binary_approximator.alpha.data
			# observation_betas = gan.G.binary_approximator.beta.data
			# observations = 1-CDFKuma(observation_alphas, observation_betas, threshold = 0.5)
			# print('alpha:', observation_alphas)
			# print('beta:', observation_betas)
			# print('observations:', observations)

		if epoch % 20 == 0:

			# Evaluate the model and store it if it performs better than previous models
			eval_error = 0
			obs_likelihood = 0
			nr_of_batches = 0
			for mini_batch in get_minibatch(BATCH_SIZE, 700, 10, list(zip(v_click_logs, v_rankings, v_features))):
				nr_of_batches += 1
				click_logs_T, rankings_T, features_T = mini_batch
				batch_eval_error, batch_obs_likelihood = gan.evaluate(click_logs_T, features_T)
				eval_error += batch_eval_error
				obs_likelihood += batch_obs_likelihood
			eval_error = - eval_error / nr_of_batches
			obs_likelihood = obs_likelihood / nr_of_batches

			real_errors.append(real_error)
			fake_errors.append(fake_error)
			g_errors.append(g_error)
			eval_errors.append(eval_error.item())
			obs_lls.append(obs_likelihood.item())

			# Save the model parameters.
			# Important! Saves parameters for G and D separately
			# When loading the parameters, also do this for G and D separately
			if eval_error < current_best:
				current_best = eval_error.item()
				obs_ll = obs_likelihood.item()
				best_epoch = epoch
				ckpt ={
					"g_state_dict": gan.G.state_dict(),
					"d_state_dict": gan.D.state_dict(),
					"best_eval": current_best,
					"best_epoch": epoch,
					"obs_likelihood": obs_ll,
					"errors": (real_errors, fake_errors, g_errors, eval_errors, obs_lls)
				}
				torch.save(ckpt, model_filename)


			with open(model_scores_file, 'w') as fout:
				fout.write(str(real_errors)+'\n')
				fout.write(str(fake_errors)+'\n')
				fout.write(str(g_errors)+'\n')
				fout.write(str(eval_errors)+'\n')
				fout.write(str(obs_lls)+'\n')

		epoch += 1

	return params, current_best, obs_ll, best_epoch

def open_data():
	pickled_clicks = pickle.load(open("train_clicks.p", "rb"))
	click_logs, rankings, features = zip(*pickled_clicks)
	pickled_clicks = pickle.load(open("valid_clicks.p", "rb"))
	v_click_logs, v_rankings, v_features = zip(*pickled_clicks)
	return click_logs, rankings, features, v_click_logs, v_rankings, v_features


#### Generative Adversarial Network for Observance iNference (GANON)

class GAN:
	def __init__(self, rank_list_size, batch_size,
	model_settings, g_optimizer, d_optimizer, criterion = nn.BCEWithLogitsLoss(),
	forward_only=False, feed_previous = False):
		self.rank_list_size = rank_list_size
		self.batch_size = batch_size

		g = model_settings["g"]
		d = model_settings["d"]
		generator = g["generator"]

		self.G = generator(g['input_size'], g['hidden_size'], g['output_size'], g['fn'], rank_list_size)
		self.D = Discriminator(d['hidden_size'], d['output_size'], d['fn'], d['feature_size'], d['embed_size'], rank_list_size)
		self.d_optimizer = d_optimizer(self.D.parameters(),lr=d['lr'])
		self.g_optimizer = g_optimizer(self.G.parameters(),lr=g['lr'])
		self.criterion = criterion
		self.errors = []

	def train(self, click_logs, rankings):
		self.G.train()
		self.D.train()
		# first train the discriminator
		self.D.zero_grad()
		# train on real data
		rankings_1, rankings_2 = torch.chunk(rankings, 2)
		click_logs_1, click_logs_2 = torch.chunk(click_logs,2)
		real_decision = self.D(click_logs_1, rankings_1)
		real_error = self.criterion(real_decision, torch.ones(real_decision.size(), device=device)  + torch.rand(real_decision.size(), device=device) *.3 - 0.2)
		real_error.backward()
		self.d_optimizer.step()

		self.D.zero_grad()
		# train on fake data
		fake_observations, fake_data = self.G(rankings_2)
		fake_decision = self.D(fake_data.detach(), rankings_2) # detach the fake data so the generator does not get updated here
		fake_error = self.criterion(fake_decision, torch.zeros(fake_decision.size(), device=device) + torch.rand(fake_decision.size(), device=device) *.3)
		fake_error.backward()
		self.d_optimizer.step()

		# then train the generator
		self.G.zero_grad()
		g_fake_decision = self.D(fake_data, rankings_2)

		g_error = self.criterion(g_fake_decision, torch.ones(g_fake_decision.size(), device=device))
		g_error.backward()
		self.g_optimizer.step()

		return real_error.item(), fake_error.item(), g_error.item()

	def evaluate(self, click_logs, rankings, criterion=nn.BCELoss()):
		self.G.eval()
		with torch.no_grad():
			self.G.zero_grad()
			observation_alphas = self.G.binary_approximator.alpha.data
			# print(observation_alphas)
			observation_betas = self.G.binary_approximator.beta.data
			# observation_betas = self.G.binary_approximator.beta
			# print(observation_betas)
			norm = self.G.normal(rankings.view(-1, rankings.size()[-1]))
			norm = norm.view(rankings.size())
			relevance_alpha_beta = self.G.g(norm)
			return LogLikelihood(observation_alphas, observation_betas,
			relevance_alpha_beta, click_logs)

			#
			# fake_observations, fake_data = self.G(rankings)
			# g_error = criterion(fake_data, click_logs)
			# return g_error.item()

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




### GENERATOR FUNCTION ======================================================

class Generator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, fn, rank_cut):
		super(Generator, self).__init__()
		self.binary_approximator = BinaryApproximator(rank_cut)
		self.normal = nn.BatchNorm1d(input_size)
		self.g = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, output_size)
		)
		self.binary_approximator_rel = BinaryApproximator(rank_cut, alpha = 1, beta = 1)

	def forward(self, relevance_scores):
		batch_size = relevance_scores.size()[0]
		rank_size = relevance_scores.size()[1]
		random_noise = torch.rand((batch_size, rank_size), device=device)
		random_noise2 = torch.rand((batch_size, rank_size), device=device)
		observation_scores = self.binary_approximator(random_noise)
		norm = self.normal(relevance_scores.view(-1, relevance_scores.size()[-1]))
		norm = norm.view(relevance_scores.size())
		alpha,beta = torch.chunk(self.g(norm), 2, dim=-1)
		# alpha = self.g(norm)
		relevance_understanding = self.binary_approximator_rel(random_noise2, alpha.squeeze(dim=2), beta.squeeze(dim=2))
		fake_click_logs = observation_scores * relevance_understanding
		return observation_scores, fake_click_logs

## ==== DISCRIMINATOR FUNCTION ==================================================

class Discriminator(nn.Module):
	def __init__(self, hidden_size, output_size, fn, feature_size=700, embed_size=8, rank_cut=10):
		super(Discriminator, self).__init__()
		self.normal = nn.BatchNorm1d(feature_size)
		self.embed = nn.Linear(feature_size, embed_size)
		self.input_size = rank_cut * embed_size + rank_cut
		self.d = nn.Sequential(
			nn.Linear(self.input_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, hidden_size),
			fn(),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, click_logs, rankings):
		norm = self.normal(rankings.view(-1, rankings.size()[-1]))
		norm = norm.view(rankings.size())
		embedding = self.embed(norm)
		data = torch.cat((click_logs, embedding.view(embedding.size()[0], -1)), dim=1)
		return self.d(data)



## HELPER FUNCTIONS ===========================================================

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
				click_logs[i] += [0] * len_dif
				rankings[i] += [-1.] * len_dif
				for j in range(0,len_dif):
					features[i].append([0.0 for _ in range(embed_size)])



		click_logs_tensor = torch.Tensor(click_logs).to(device)
		rankings_tensor = torch.Tensor(rankings).to(device)[:,:,None]
		feature_tensor = torch.Tensor(features).to(device)
		yield click_logs_tensor, rankings_tensor, feature_tensor

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
			self.beta = nn.Parameter(torch.randn((1,input_size)))
		self.gamma = gamma
		self.zeta = zeta

		#if BinaryApproximator is used by a neural net put alpha on some value.
	def forward(self, u, alpha = None, beta = None):
		if alpha is not None and beta is not None:
			s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(alpha)))/F.softplus(beta))
		elif alpha is not None:
			if type(self.beta) == float:
				s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(alpha)))/self.beta)
			else:
				s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(alpha)))/F.softplus(self.beta))
		else:
			if type(self.beta) == float:
				s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(self.alpha))/self.beta))
			else:
				s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(F.softplus(self.alpha))/F.softplus(self.beta)))
		mean_s = s * (self.zeta - self.gamma) + self.gamma
		binarysize = mean_s.size()
		z = torch.min(torch.ones(binarysize, device=device),(torch.max(torch.zeros(binarysize, device=device),mean_s)))
		return z

def CDFKuma(alpha, beta, threshold = 0.5):
	assert type(beta) == float or alpha.size() == beta.size()
	t = torch.zeros(alpha.size()).to(device) + threshold
	if type(beta) == float:
		return 1-torch.pow(1-torch.pow(t, F.softplus(alpha)), beta)
	return 1-torch.pow(1-torch.pow(t, F.softplus(alpha)), F.softplus(beta))

def LogLikelihood(observation_alpha, observation_beta,
	relevance_parameters, label):

	p_observation = 1-CDFKuma(observation_alpha, observation_beta)
	avg_observations = torch.tensor([[1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10]]).to(device)
	obs_likelihood = torch.pow(p_observation,avg_observations)*torch.pow(1-p_observation,1-avg_observations)
	# return torch.sum(torch.log(obs_likelihood))
	alpha, beta = torch.chunk(relevance_parameters, 2, dim=-1)
	# alpha = relevance_parameters
	# beta = observation_beta
	p_relevance = 1-CDFKuma(alpha, beta).squeeze(dim=-1)

	p_click = p_observation * p_relevance
	p_not_click = 1 - p_click
	likelihood = torch.pow(p_click,label)*torch.pow(p_not_click,1-label)

	return torch.mean(torch.sum(torch.log(likelihood), dim=1)), torch.sum(torch.log(obs_likelihood))


if __name__ == "__main__":
	mp.set_start_method('spawn')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	standard_dtype = torch.double
	torch.set_default_dtype(standard_dtype)
	if device == torch.device('cuda'):
		torch.cuda.empty_cache()

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

	# Generate clicks if necessary
	GENERATE = False
	try:
		assert not GENERATE
		assert os.path.isfile("train_clicks.p") and os.path.isfile("valid_clicks.p")
	except:
		with open(CLICK_MODEL_JSON) as fin:
			model_desc = json.load(fin)
			click_model = cm.loadModelFromJson(model_desc)

		# process dataset from file
		train_set = data_utils.read_data(INPUT_DATA_PATH, 'train', RANK_CUT)
		valid_set = data_utils.read_data(INPUT_DATA_PATH, 'valid', RANK_CUT)


		click_logs, rankings, features = generate_clicks(10000, click_model, train_set.gold_weights, train_set.featuredids)
		print("Train clicks generated!")
		zipped_all = zip(click_logs,rankings,features)
		pickle.dump(zipped_all, open("train_clicks.p","wb"))
		print("Saved train clicks in a pickle!")
		v_click_logs, v_rankings, v_features = generate_clicks(10000, click_model, valid_set.gold_weights, valid_set.featuredids)
		print("Valid clicks generated!")
		zipped_all = zip(v_click_logs,v_rankings,v_features)
		pickle.dump(zipped_all, open("valid_clicks.p","wb"))
		print("Saved valid clicks in a pickle!")

	# Do Hyper Parameter optimization
	bt_s = [16384]
	lr_d = [5e-4]
	lr_g = [2.5e-3]
	hs_d = [32]
	hs_g = [4]
	emb_d = [4]

	param_combinations = list(itertools.product(bt_s,lr_d,lr_g,hs_d,hs_g,emb_d))
	random.shuffle(param_combinations)

	param_combinations = [
		(1024, 1e-4, 5e-4, 16, 4, 4),
		(1024, 1e-4, 5e-4, 16, 8, 4),
		(1024, 1e-4, 5e-4, 16, 16, 4)
	]

	with mp.Pool(processes=3) as pool:
		results = pool.map(run, param_combinations)
	results = sorted(results, key=lambda tup: tup[1])

	with open('data-5000.txt', 'a') as fout:
		for result in results:
			fout.write(str(result) +'\n')

	# main()





