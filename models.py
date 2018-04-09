import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import RNNEncoder,RNNDecoder,CNNEncoder,CNNDecoder,HybridDecoder
from sampleLayers import NormalDistributed
from helpers import Normal

###########################
####   SEQ2SEQ MODELS  ####
###########################

class RNNSeq2Seq(nn.Module):
	def __init__(self,rnn_size):
		super().__init__()
		"""
		Layer definitions
		"""
		# encode from input space to hidden space
		self.encoder = RNNEncoder(input_size=1,rnn_size=rnn_size)
		# decode from hidden space to input space
		self.decoder = RNNDecoder(input_size=1,rnn_size=rnn_size,output_size=1)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		dec = self.decoder(enc,num_steps)
		return dec

class CNNSeq2Seq(nn.Module):
	def __init__(self,conv_size):
		super().__init__()
		"""
		Layer definitions
		"""
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=1,conv_size=conv_size)
		# decode from hidden space to input space
		self.decoder = CNNDecoder(input_size=1,conv_size=conv_size,output_size=1)

	def forward(self,x):
		enc = self.encoder(x)
		dec = self.decoder(enc)
		return dec

class HybridSeq2Seq(nn.Module):
	def __init__(self,conv_size,rnn_size):
		super().__init__()
		"""
		Layer definitions
		"""
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=1,conv_size=conv_size)
		# decode from hidden space to input space
		self.decoder = HybridDecoder(input_size=1,conv_size=conv_size,rnn_size=rnn_size,output_size=1)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		dec = self.decoder(enc,num_steps)
		return dec

###########################
####     VAE MODELS    ####
###########################

class RNNVAE(nn.Module):
	def __init__(self,rnn_size,latent_size):
		super().__init__()
		"""
		Layer definitions
		"""
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = RNNEncoder(input_size=1,rnn_size=rnn_size)
		# encoded to latent layer
		self.h2z = nn.Sequential(
			nn.Linear(rnn_size,self.samplelayer.inputShape()[-1]),
			nn.ELU()
			)
		# latent to decoded layer
		self.z2h = nn.Sequential(
			nn.Linear(self.samplelayer.outputShape()[-1],rnn_size),
			nn.ELU()
			)
		# decode from hidden space to input space
		self.decoder = RNNDecoder(input_size=1,rnn_size=rnn_size,output_size=1)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec = self.decoder(dec,num_steps)
		return dec,qz

	def sample(self,num_samples,num_steps):
		h = Variable(torch.zeros(num_samples,self.samplelayer.inputShape()[-1]))
		mu,logvar = h.chunk(2,1)
		qz = Normal(mu,logvar)
		z = qz.sample()
		dec = self.z2h(z)
		dec = self.decoder(dec,num_steps)
		return dec

class CNNVAE(nn.Module):
	def __init__(self,conv_size,latent_size):
		super().__init__()
		"""
		Layer definitions
		"""
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=1,conv_size=conv_size)
		# encoded to latent layer
		self.h2z = nn.Sequential(
			nn.Linear(conv_size,self.samplelayer.inputShape()[-1]),
			nn.ELU()
			)
		# latent to decoded layer
		self.z2h = nn.Sequential(
			nn.Linear(self.samplelayer.outputShape()[-1],conv_size),
			nn.ELU()
			)
		# decode from hidden space to input space
		self.decoder = CNNDecoder(input_size=1,conv_size=conv_size,output_size=1)

	def forward(self,x):
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec = self.decoder(dec)
		return dec,qz

	def sample(self,num_samples):
		h = Variable(torch.zeros(num_samples,self.samplelayer.inputShape()[-1]))
		mu,logvar = h.chunk(2,1)
		qz = Normal(mu,logvar)
		z = qz.sample()
		dec = self.z2h(z)
		dec = self.decoder(dec)
		return dec

class HybridVAE(nn.Module):
	def __init__(self,conv_size,rnn_size,latent_size):
		super().__init__()
		"""
		Layer definitions
		"""
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=1,conv_size=conv_size)
		# encoded to latent layer
		self.h2z = nn.Sequential(
			nn.Linear(conv_size,self.samplelayer.inputShape()[-1]),
			nn.ELU()
			)
		# latent to decoded layer
		self.z2h = nn.Sequential(
			nn.Linear(self.samplelayer.outputShape()[-1],conv_size),
			nn.ELU()
			)
		# decode from hidden space to input space
		self.decoder = HybridDecoder(input_size=1,conv_size=conv_size,rnn_size=rnn_size,output_size=1)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec,aux_x = self.decoder(dec,num_steps)
		return dec,qz,aux_x

	def sample(self,num_samples,num_steps):
		h = Variable(torch.zeros(num_samples,self.samplelayer.inputShape()[-1]))
		mu,logvar = h.chunk(2,1)
		qz = Normal(mu,logvar)
		z = qz.sample()
		dec = self.z2h(z)
		dec,_ = self.decoder(dec,num_steps)
		return dec