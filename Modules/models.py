import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import RNNEncoder,AdvancedRNNEncoder,RNNDecoder,AdvancedRNNDecoder,CNNEncoder,AdvancedCNNEncoder,CNNDecoder,AdvancedCNNDecoder
from layers import HybridDecoder,AdvancedHybridDecoder,LadderEncoder,LadderDecoder,LadderCNNEncoder,LadderCNNDecoder
from layers import CNNEncoderSmall,CNNDecoderSmall,HybridDecoderSmall
from sampleLayers import NormalDistributed,AdvancedNormalDistributed
from helpers import kl_div

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
	def __init__(self,input_size,rnn_size,latent_size,output_size,use_softmax=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = False
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = RNNEncoder(input_size=input_size,rnn_size=rnn_size)
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
		self.decoder = RNNDecoder(input_size=input_size,rnn_size=rnn_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec = self.decoder(dec,num_steps)
		return dec,qz

	def sample(self,z,num_steps):
		dec = self.z2h(z)
		dec = self.decoder(dec,num_steps)
		return dec

class AdvancedRNNVAE(nn.Module):
	def __init__(self,input_size,rnn_size,latent_size,output_size,use_softmax=False,bidirectional=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = False
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = AdvancedNormalDistributed(latent_size=latent_size) # sample is of size [seq_len, batch_size, latent_size]
		# encode from input space to hidden space
		self.encoder = AdvancedRNNEncoder(input_size=input_size,rnn_size=rnn_size,bidirectional=bidirectional)
		# encoded to latent layer
		if bidirectional:
			directions = 2
		else:
			directions = 1
		self.h2z = nn.Sequential(
			nn.Linear(directions*rnn_size,self.samplelayer.inputShape()[-1]),
			nn.ELU()
			)
		# latent to decoded layer
		self.z2h = nn.Sequential(
			nn.Linear(self.samplelayer.outputShape()[-1],rnn_size),
			nn.ELU()
			)
		# decode from hidden space to input space
		self.decoder = AdvancedRNNDecoder(input_size=input_size,rnn_size=rnn_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec = self.decoder(dec,num_steps)
		return dec,qz

	def sample(self,z,num_steps):
		dec = self.z2h(z)
		dec = self.decoder(dec,num_steps)
		return dec

class CNNVAESmall(nn.Module):
	def __init__(self,input_size,conv_size,latent_size,output_size,use_softmax=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = False
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoderSmall(input_size=input_size,conv_size=conv_size)
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
		self.decoder = CNNDecoderSmall(input_size=input_size,conv_size=conv_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec = self.decoder(dec)
		return dec,qz

class CNNVAE(nn.Module):
	def __init__(self,input_size,conv_size,latent_size,output_size,use_softmax=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = False
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=input_size,conv_size=conv_size)
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
		self.decoder = CNNDecoder(input_size=input_size,conv_size=conv_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec = self.decoder(dec)
		return dec,qz

	def sample(self,z):
		'''
		h = Variable(torch.zeros(num_samples,self.samplelayer.inputShape()[-1]))
		mu,logvar = h.chunk(2,1)
		qz = Normal(mu,logvar)
		z = qz.sample()
		'''
		dec = self.z2h(z)
		dec = self.decoder(dec)
		return dec

class AdvancedCNNVAE(nn.Module):
	def __init__(self,input_size,conv_size,latent_size,output_size,use_softmax=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = False
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = AdvancedNormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = AdvancedCNNEncoder(input_size=input_size,conv_size=conv_size)
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
		self.decoder = AdvancedCNNDecoder(input_size=input_size,conv_size=conv_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec = self.decoder(dec)
		return dec,qz

	def sample(self,z):
		dec = self.z2h(z)
		dec = self.decoder(dec)
		return dec

class HybridVAESmall(nn.Module):
	def __init__(self,input_size,conv_size,rnn_size,latent_size,output_size,use_softmax=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = True
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoderSmall(input_size=input_size,conv_size=conv_size)
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
		self.decoder = HybridDecoderSmall(input_size=input_size,conv_size=conv_size,rnn_size=rnn_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec,aux_x = self.decoder(dec,num_steps)
		return dec,qz,aux_x

class HybridVAE(nn.Module):
	def __init__(self,input_size,conv_size,rnn_size,latent_size,output_size,use_softmax=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = True
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=input_size,conv_size=conv_size)
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
		self.decoder = HybridDecoder(input_size=input_size,conv_size=conv_size,rnn_size=rnn_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec,aux_x = self.decoder(dec,num_steps)
		return dec,qz,aux_x

	def sample(self,z,num_steps):
		'''
		h = Variable(torch.zeros(num_samples,self.samplelayer.inputShape()[-1]))
		mu,logvar = h.chunk(2,1)
		qz = Normal(mu,logvar)
		z = qz.sample()
		'''
		dec = self.z2h(z)
		dec,_ = self.decoder(dec,num_steps)
		return dec

class AdvancedHybridVAE(nn.Module):
	def __init__(self,input_size,conv_size,rnn_size,latent_size,output_size,use_softmax=False):
		super().__init__()
		"""
		Layer definitions
		"""
		self.aux_loss = True
		self.kl_loss = False
		# sample layer with normal distribution
		self.samplelayer = AdvancedNormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = AdvancedCNNEncoder(input_size=input_size,conv_size=conv_size)
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
		self.decoder = AdvancedHybridDecoder(input_size=input_size,conv_size=conv_size,rnn_size=rnn_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.h2z(enc)
		z,qz = self.samplelayer(enc)
		dec = self.z2h(z)
		dec,aux_x = self.decoder(dec,num_steps)
		return dec,qz,aux_x

	def sample(self,z,num_steps):
		dec = self.z2h(z)
		dec,_ = self.decoder(dec,num_steps)
		return dec

class LadderVAE(nn.Module):
	def __init__(self,input_size,hidden_sizes,latent_sizes,recon_hidden_size,output_size,use_softmax=False):
		super().__init__()
		self.aux_loss = False
		self.kl_loss = True
		self.sample_size = latent_sizes[-1]
		layer_sizes = [input_size,*hidden_sizes]
		encoder_layers = [LadderEncoder(input_size=layer_sizes[i-1],hidden_size=layer_sizes[i],latent_size=latent_sizes[i-1]) for i in range(1,len(layer_sizes))]
		self.encoder = nn.ModuleList(encoder_layers)
		decoder_layers = [LadderDecoder(input_size=latent_sizes[i],hidden_size=hidden_sizes[i-1],latent_size=latent_sizes[i-1]) for i in range(1,len(hidden_sizes))][::-1]
		self.decoder = nn.ModuleList(decoder_layers)
		self.z2h = nn.Sequential(
			nn.Linear(latent_sizes[0],recon_hidden_size),
			nn.ELU()
			)
		self.reconstruction = AdvancedRNNDecoder(input_size=input_size,rnn_size=recon_hidden_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		num_steps = x.size()[0]
		latents = []
		for encoder in self.encoder:
			x,z,qz = encoder(x)
			latents.append(qz)
		latents = list(reversed(latents))
		self.kl = 0
		for i,decoder in enumerate([-1,*self.decoder]):
			qz = latents[i]
			if i == 0:
				self.kl += kl_div(z,qz)
			else:
				z,kl_terms = decoder(z,qz)
				self.kl += kl_div(*kl_terms)
		z = self.z2h(z)
		dec = self.reconstruction(z,num_steps)
		return dec,None

	def sample(self,z):
		num_steps = z.size()[0]
		for decoder in self.decoder:
			z = decoder(z)
		z = self.z2h(z)
		dec = self.reconstruction(z,num_steps)
		return dec

class LadderCNNVAE(nn.Module):
	def __init__(self,input_size,hidden_sizes,latent_sizes,recon_hidden_size,output_size,use_softmax=False):
		super().__init__()
		self.aux_loss = False
		self.kl_loss = True
		self.sample_size = latent_sizes[-1]
		layer_sizes = [input_size,*hidden_sizes]
		encoder_layers = [LadderCNNEncoder(input_size=layer_sizes[i-1],hidden_size=layer_sizes[i],latent_size=latent_sizes[i-1]) for i in range(1,len(layer_sizes))]
		self.encoder = nn.ModuleList(encoder_layers)
		decoder_layers = [LadderCNNDecoder(input_size=latent_sizes[i],hidden_size=hidden_sizes[i-1],latent_size=latent_sizes[i-1]) for i in range(1,len(hidden_sizes))][::-1]
		self.decoder = nn.ModuleList(decoder_layers)
		self.z2h = nn.Sequential(
			nn.ConvTranspose1d(in_channels=latent_sizes[0],out_channels=recon_hidden_size,kernel_size=3),
			nn.ELU()
			)
		self.reconstruction = AdvancedRNNDecoder(input_size=input_size,rnn_size=recon_hidden_size,output_size=output_size,use_softmax=use_softmax)

	def forward(self,x):
		num_steps = x.size()[0]
		latents = []
		for encoder in self.encoder:
			x,z,qz = encoder(x)
			latents.append(qz)
		latents = list(reversed(latents))
		self.kl = 0
		for i,decoder in enumerate([-1,*self.decoder]):
			qz = latents[i]
			if i == 0:
				self.kl += kl_div(z,qz)
			else:
				z,kl_terms = decoder(z,qz)
				self.kl += kl_div(*kl_terms)
		z = self.z2h(z.transpose(0,1).transpose(1,2)).transpose(2,1).transpose(1,0)
		dec = self.reconstruction(z,num_steps)
		return dec,None

	def sample(self,z):
		num_steps = z.size()[0]
		for decoder in self.decoder:
			z = decoder(z)
		z = self.z2h(z.transpose(0,1).transpose(1,2)).transpose(2,1).transpose(1,0)
		dec = self.reconstruction(z,num_steps)
		return dec