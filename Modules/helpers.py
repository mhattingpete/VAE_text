from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np

class Normal:
	def __init__(self,mu,logvar):
		self.mu = mu
		self.logvar = logvar
		self.size = self.mu.size()

	def sample(self):
		var = torch.exp(0.5*self.logvar)
		eps = Variable(torch.FloatTensor(*self.size).normal_(),requires_grad=False)
		#z = eps.mul(var).add_(self.mu)
		z = self.mu + var*eps
		return z

def kl_loss(qz):
	kl = (-0.5*(qz.logvar-torch.exp(qz.logvar)-torch.pow(qz.mu,2)+1).sum(1)).mean()
	return kl

def kl_loss_multi(qz):
	kl = (-0.5*(qz.logvar-torch.exp(qz.logvar)-torch.pow(qz.mu,2)+1).sum(2)).sum(1).mean()
	return kl

# try to reshape the qz to have seq_len,batch,latent_size and then sum over seq_len and mean over batch

def nll_loss(recon_x,x):
	num_classes = recon_x.shape[2]
	x_pl = x.contiguous().view(-1).type(torch.LongTensor)
	recon_x_pl = recon_x.contiguous().view(-1,num_classes)
	return F.nll_loss(recon_x_pl,x_pl)

def mse_loss(recon_x,x):
	x_pl = x.contiguous().view(-1)
	recon_x_pl = recon_x.contiguous().view(-1)
	return F.mse_loss(recon_x_pl,x_pl)