from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

from visualize import showPlot,plotPredictions
from models import RVAE,CVAE,HybridVAE

from ToyDatasets.timeSeries import Sinusoids

batch_size = 100
num_steps = 15
train_size = 1000
valid_size = 1000
num_classes = 0
# use just one
train_loader = DataLoader(Sinusoids(num_steps,virtual_size=train_size,quantization=num_classes),batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(Sinusoids(num_steps,virtual_size=valid_size,quantization=num_classes),batch_size=batch_size,shuffle=True)

def train(input_variable,model,optimizer,beta=1.0,alpha=0.2):
	model.train()
	optimizer.zero_grad()
	input_length = input_variable.size()[0]
	loss,diagnostics,data = model.getLoss(input_variable,beta=beta,alpha=alpha)
	loss.backward()
	optimizer.step()
	return loss.data[0],diagnostics['kl'][0],diagnostics['r_loss'][0],diagnostics['aux_loss'][0],data

def evaluate(input_variable,model):
	model.eval()
	input_length = input_variable.size()[0]
	loss,diagnostics,data = model.getLoss(input_variable)
	return loss.data[0],diagnostics['kl'][0],diagnostics['r_loss'][0],diagnostics['aux_loss'][0],data

def trainIters(model,n_iters,print_every=1000,plot_every=100,learning_rate=0.01):
	plot_losses = []
	print_loss_total = 0
	kl_loss_total = 0
	r_loss_total = 0
	aux_loss_total = 0
	correct_total = 0
	plot_loss_total = 0
	plot_losses_valid = []
	print_loss_valid_total = 0
	kl_loss_valid_total = 0
	r_loss_valid_total = 0
	aux_loss_valid_total = 0
	correct_valid_total = 0
	plot_loss_valid_total = 0

	optimizer = optim.Adam(model.parameters(),lr=learning_rate)
	beta_start_inc = int(n_iters*0.3)
	beta_weight = 0.05
	beta_max = 0.1
	beta_inc = 0.000002

	for iter in range(1,n_iters+1):
		for batch_idx,batch in enumerate(train_loader):
			input_variable = Variable(batch).type(torch.FloatTensor).transpose(1,0)
			loss,kl_loss,r_loss,aux_loss,data = train(input_variable,model,optimizer,beta=beta_weight)
			if iter > beta_start_inc and beta_weight < beta_max:
				beta_weight += beta_inc
			print_loss_total += loss
			kl_loss_total += kl_loss
			r_loss_total += r_loss
			aux_loss_total += aux_loss
			plot_loss_total += r_loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			kl_loss_avg = kl_loss_total / print_every
			r_loss_avg = r_loss_total / print_every
			aux_loss_avg = aux_loss_total / print_every
			print_loss_total = 0
			kl_loss_total = 0
			r_loss_total = 0
			aux_loss_total = 0
			print('\nTrain ({0:d} {1:d}%) loss: {2:.4f}, kl: {3:.4f}, r_loss: {4:.4f}, aux: {5:.4f}'.format(iter,int(iter/n_iters*100),print_loss_avg,kl_loss_avg,r_loss_avg,aux_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

		for batch_idx,batch in enumerate(valid_loader):
			input_variable = Variable(batch).type(torch.FloatTensor).transpose(1,0)
			loss,kl_loss,r_loss,aux_loss,data_valid = evaluate(input_variable,model)
			print_loss_valid_total += loss
			kl_loss_valid_total += kl_loss
			r_loss_valid_total += r_loss
			aux_loss_valid_total += aux_loss
			plot_loss_valid_total += r_loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_valid_total / print_every
			kl_loss_avg = kl_loss_valid_total / print_every
			r_loss_avg = r_loss_valid_total / print_every
			aux_loss_avg = aux_loss_valid_total / print_every
			print_loss_valid_total = 0
			kl_loss_valid_total = 0
			r_loss_valid_total = 0
			aux_loss_valid_total = 0
			print('Valid ({0:d} {1:d}%) loss: {2:.4f}, kl: {3:.4f}, r_loss: {4:.4f}, aux: {5:.4f}'.format(iter,int(iter/n_iters*100),print_loss_avg,kl_loss_avg,r_loss_avg,aux_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_valid_total / plot_every
			plot_losses_valid.append(plot_loss_avg)
			plot_loss_valid_total = 0

	showPlot(plot_losses,plot_losses_valid,yu=4)
	plotPredictions(data['x'],data['recon_x'],xu=num_steps,yu=1)
	plotPredictions(data_valid['x'],data_valid['recon_x'],xu=num_steps,yu=1)
	plt.show()

if __name__ == '__main__':
	hidden_size = 512
	latent_size = 256
	criterion = nn.MSELoss()
	model = RVAE(hidden_size=hidden_size,latent_size=latent_size,criterion=criterion,dropout_prob=0.4,teacher_forcing_ratio=0.5)
	#model = CVAE(hidden_size=hidden_size,latent_size=latent_size,criterion=criterion,dropout_prob=0.4)
	#model = HybridVAE(hidden_size=hidden_size,latent_size=latent_size,criterion=criterion,dropout_prob=0.4,teacher_forcing_ratio=0.5)
	trainIters(model=model,n_iters=1000,print_every=10,plot_every=10,learning_rate=1e-4)