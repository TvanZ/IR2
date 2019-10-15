import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	DLA_logits = open("DLA_logits.txt", "r")
	DLA_fixer = DLA_logits.readlines()
	DLA_logits.close()
	print("opened a pickle!")
	propensity_logits = DLA_fixer[0]
	relevance_logits = DLA_fixer[1:]
	print(propensity_logits)
	print(relevance_logits)	

	prop_tensor = torch.tensor(propensity_logits)
	rel_tensor = torch.tensor(relevance_logits)

	# DO SOFTMAX <3
	#F.softmax


if __name__ == "__main__":
	if device == torch.device('cuda'):
		torch.cuda.empty_cache()
	main()