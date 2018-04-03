import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

from helpers import Normal

class SampleLayer(nn.Module):
	'''
	Sample layer that draws samples from a distribution this is done with a forward pass
	'''
	def loss(self):
		raise NotImplementedError

	def inputShape(self):
		raise NotImplementedError

	def outputShape(self):
		raise NotImplementedError

class NormalDistributed(SampleLayer):
	'''
	Draws samples from a normal distribution with a given mean and variance
	'''
	def __init__(self,latent_size):
		'''
		Inputs:
			latent_size: the number of features in the latent space
		'''
		super().__init__()
		self.latent_size = latent_size

	def forward(self,inputs):
		mu,logvar = inputs.chunk(2,1)
		qz = Normal(mu,logvar)
		samples = qz.sample()
		return samples,qz

	def getLoss(self,qz):
		kl = (-0.5*(qz.logvar-torch.exp(qz.logvar)-torch.pow(qz.mu,2)+1).sum(1))
		return kl

	def inputShape(self):
		return (-1,2*self.latent_size)

	def outputShape(self):
		return (-1,self.latent_size)