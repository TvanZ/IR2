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

model_filename = "./models/4096-0.01-0.05-32-0-4-fixed-relevance-paper.p"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
standard_dtype = torch.double
torch.set_default_dtype(standard_dtype)

ckpt = torch.load(model_filename)
g = ckpt["g_state_dict"]
alpha = g['binary_approximator.alpha']
beta = g['binary_approximator.beta']
current_best = ckpt["best_eval"]

print(alpha)
print(F.softplus(alpha))
print(beta)
print(F.softplus(beta))

def CDFKuma(alpha, beta, threshold = 0.5):
	t = torch.zeros(alpha.size()).to(device) + threshold
	return 1-torch.pow(1-torch.pow(t, F.softplus(alpha)), F.softplus(beta))

print(1-CDFKuma(alpha, beta))

