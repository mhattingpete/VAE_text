import torch
from torch import nn
from torch.autograd import Variable

from helpers import Normal

###########################
####   SAMPLE MODULES  ####
###########################

class SampleLayer(nn.Module):
	'''
	Sample layer that draws samples from a distribution this is done with a forward pass
	'''
	def forward(self):
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
		mu,logvar = inputs.chunk(2,-1)
		qz = Normal(mu,logvar)
		samples = qz.sample()
		return samples,qz

	def inputShape(self):
		return (-1,2*self.latent_size)

	def outputShape(self):
		return (-1,self.latent_size)

	def __repr__(self):
		return "{}(latent_size={})".format(self.__class__.__name__, self.latent_size)

"""
class AdvancedNormalDistributedOLD(SampleLayer):
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
		seq_len = inputs.size()[0]
		inputs = inputs.view(-1,self.latent_size)
		mu,logvar = inputs.chunk(2,1)
		qz = Normal(mu,logvar)
		samples = qz.sample().view(seq_len,-1,self.latent_size)
		return samples,qz

	def inputShape(self):
		return (-1,-1,2*self.latent_size)

	def outputShape(self):
		return (-1,-1,self.latent_size)
"""

class AdvancedNormalDistributed(SampleLayer):
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
		mu,logvar = inputs.chunk(2,-1)
		qz = Normal(mu,logvar)
		samples = qz.sample()
		return samples,qz

	def inputShape(self):
		return (-1,-1,2*self.latent_size)

	def outputShape(self):
		return (-1,-1,self.latent_size)

	def __repr__(self):
		return "{}(latent_size={})".format(self.__class__.__name__, self.latent_size)

class GaussianMerge(nn.Module):
	def __init__(self,latent_size):
		super().__init__()
		self.latent_size = latent_size

	def forward(self,inputs,mu2,logvar2):
		mu1,logvar1 = inputs.chunk(2,-1)
		precision1, precision2 = (1/torch.exp(logvar1), 1/torch.exp(logvar2))
		mu = ((mu1 * precision1) + (mu2 * precision2))/(precision1 + precision2)
		var = 1/(precision1 + precision2)
		logvar = torch.log(var + 1e-8)
		qz = Normal(mu,logvar)
		samples = qz.sample()
		return samples,qz

	def inputShape(self):
		return (-1,2*self.latent_size)

	def outputShape(self):
		return (-1,self.latent_size)

	def __repr__(self):
		return "{}(latent_size={})".format(self.__class__.__name__, self.latent_size)