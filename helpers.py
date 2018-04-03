from torch.autograd import Variable
import torch
import numpy as np

class Normal:
	def __init__(self,mu,logvar):
		self.mu = mu
		self.logvar = logvar
		self.size = self.mu.size()

	def sample(self):
		std = self.logvar.mul(0.5).exp_()
		eps = Variable(std.data.new(*self.size).normal_())
		z = eps.mul(std).add_(self.mu)
		return z

class WordDropout:
	def __init__(self,p,dummy):
		self.p = 1-p
		self.dummy = dummy

	def __call__(self,input):
		mask = Variable(torch.from_numpy(np.random.binomial(1,0.5,size=input.data.numpy().shape))).type(torch.FloatTensor)
		return input*mask + self.dummy*(1-mask)