from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import math

class Normal:
	def __init__(self,mu,logvar):
		self.mu = mu
		self.logvar = logvar
		self.size = self.mu.size()

	def sample(self):
		var = torch.exp(0.5*self.logvar) # might have to add F.softplus() around self.logvar
		eps = Variable(torch.FloatTensor(*self.size).normal_(),requires_grad=False)
		z = self.mu + var*eps
		return z

def kl_loss(qz):
	kl = (-0.5*(qz.logvar-torch.exp(qz.logvar)-torch.pow(qz.mu,2)+1).sum(1)).mean()
	return kl

def kl_loss_multi(qz):
	kl = (-0.5*(qz.logvar-torch.exp(qz.logvar)-torch.pow(qz.mu,2)+1).sum(2)).sum(0).mean()
	return kl

def nll_loss(recon_x,x):
	num_classes = recon_x.shape[2]
	x_pl = x.contiguous().view(-1).type(torch.LongTensor)
	recon_x_pl = recon_x.contiguous().view(-1,num_classes)
	return F.nll_loss(recon_x_pl,x_pl)

def mse_loss(recon_x,x):
	x_pl = x.contiguous().view(-1)
	recon_x_pl = recon_x.contiguous().view(-1)
	return F.mse_loss(recon_x_pl,x_pl)

def log_standard_gaussian(z):
	return (-0.5*math.log(2*math.pi)-z**2/2).sum(-1)

def log_gaussian(z,qz):
	logpdf = -0.5*math.log(2*math.pi)-qz.logvar/2-(z-qz.mu)**2/(2*torch.exp(qz.logvar))
	return logpdf.sum(-1)

def kl_div(z,q_param,p_param=None):
	qz = log_gaussian(z,q_param)
	if p_param is None:
		pz = log_standard_gaussian(z)
	else:
		pz = log_gaussian(z,p_param)
	return (qz - pz).sum(0).mean()