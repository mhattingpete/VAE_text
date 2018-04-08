import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F

from layers import RNNEncoder,RNNDecoder,CNNEncoder,CNNDecoder,HybridDecoder
from sampleLayers import NormalDistributed

def weight_init(m):
	if type(m) == nn.Linear:
		nn.init.xavier_normal(m.weight.data)
	elif type(m) == nn.GRUCell:
		nn.init.xavier_normal(m.weight_hh.data)
		nn.init.xavier_normal(m.weight_ih.data)
	else:
		pass

class RVAE(nn.Module):
	def __init__(self,hidden_size,latent_size,criterion,dropout_prob=0.2,teacher_forcing_ratio=0.5):
		super().__init__()
		"""
		Layer definitions
		"""
		# sample from latent space
		self.sampler = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = RNNEncoder(input_size=1,hidden_size=hidden_size,dropout_prob=dropout_prob)
		# convert from hidden space to latent space
		self.hx2z = nn.Linear(2*hidden_size,self.sampler.inputShape()[-1])
		# convert from latent space to hidden space
		self.hz2x = nn.Linear(self.sampler.outputShape()[-1],hidden_size)
		# decode from hidden space to input space
		self.decoder = RNNDecoder(input_size=1,hidden_size=hidden_size,output_size=1,dropout_prob=dropout_prob)
		"""
		Function definitions
		"""
		# activation function
		self.relu = nn.ReLU()
		# dropout function
		self.dropout = nn.Dropout(p=dropout_prob)
		# loss function
		self.criterion = criterion
		"""
		Initialise weights
		"""
		self.apply(weight_init)

	# decoding latent space
	def z2x(self,z,num_steps,x=None):
		h = self.hz2x(z)
		h = self.relu(h)
		h = self.dropout(h)
		out = self.decoder(h,num_steps,x=x)
		return out

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.relu(enc)
		enc = self.dropout(enc)
		enc = self.hx2z(enc)
		enc = self.relu(enc)
		enc = self.dropout(enc)
		z,qz = self.sampler(enc)
		dec = self.z2x(z,num_steps,x=x)
		return dec,qz,z

	def getLoss(self,x,beta=1.0,alpha=None):
		num_steps = x.size()[0]
		recon_x,qz,z = self.forward(x)
		x = x.contiguous().view(-1)
		recon_x = recon_x.view(-1)
		kl = self.sampler.getLoss(qz)
		r_loss = self.criterion(recon_x,x)
		loss = r_loss+beta*kl
		x = x.view(num_steps,-1)
		recon_x = recon_x.view(num_steps,-1)
		diagnostics = {'kl':kl.data,'loss':loss.data,'r_loss':r_loss.data,'aux_loss':torch.FloatTensor([0])}
		data = {'recon_x':recon_x,'x':x,'qz':qz,'z':z}
		return loss,diagnostics,data

class CVAE(nn.Module):
	def __init__(self,hidden_size,latent_size,criterion,dropout_prob=0.2):
		super().__init__()
		"""
		Layer definitions
		"""
		# sample from latent space
		self.sampler = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=1,hidden_size=hidden_size,dropout_prob=dropout_prob)
		# convert from hidden space to latent space
		self.hx2z = nn.Linear(hidden_size,self.sampler.inputShape()[-1])
		# convert from latent space to hidden space
		self.hz2x = nn.Linear(self.sampler.outputShape()[-1],hidden_size)
		# decode from hidden space to input space
		self.decoder = CNNDecoder(input_size=1,hidden_size=hidden_size,output_size=1,dropout_prob=dropout_prob)
		"""
		Function definitions
		"""
		# activation function
		self.relu = nn.ReLU()
		# dropout function
		self.dropout = nn.Dropout(p=dropout_prob)
		# loss function
		self.criterion = criterion
		"""
		Initialise weights
		"""
		self.apply(weight_init)

	# decoding latent space
	def z2x(self,z):
		h = self.hz2x(z)
		h = self.relu(h)
		h = self.dropout(h)
		out = self.decoder(h)
		return out

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.relu(enc)
		enc = self.dropout(enc)
		enc = self.hx2z(enc)
		enc = self.relu(enc)
		enc = self.dropout(enc)
		z,qz = self.sampler(enc)
		out = self.z2x(z)
		return out,qz,z

	def getLoss(self,x,beta=1.0,alpha=None):
		num_steps = x.size()[0]
		recon_x,qz,z = self.forward(x)
		x = x.contiguous().view(-1)
		recon_x = recon_x.view(-1)
		kl = self.sampler.getLoss(qz)
		r_loss = self.criterion(recon_x,x)
		loss = r_loss+beta*kl
		x = x.view(num_steps,-1)
		recon_x = recon_x.view(num_steps,-1)
		diagnostics = {'kl':kl.data,'loss':loss.data,'r_loss':r_loss.data,'aux_loss':torch.FloatTensor([0])}
		data = {'recon_x':recon_x,'x':x,'qz':qz,'z':z}
		return loss,diagnostics,data

class HybridVAE(nn.Module):
	def __init__(self,hidden_size,latent_size,criterion,dropout_prob=0.2,teacher_forcing_ratio=0.5):
		super().__init__()
		"""
		Layer definitions
		"""
		# sample from latent space
		self.sampler = NormalDistributed(latent_size=latent_size)
		# encode from input space to hidden space
		self.encoder = CNNEncoder(input_size=1,hidden_size=hidden_size,dropout_prob=dropout_prob)
		# convert from hidden space to latent space
		self.hx2z = nn.Linear(hidden_size,self.sampler.inputShape()[-1])
		# convert from latent space to hidden space
		self.hz2x = nn.Linear(self.sampler.outputShape()[-1],hidden_size)
		# decode from hidden space to input space
		self.decoder = HybridDecoder(input_size=1,hidden_size=hidden_size,output_size=1,teacher_forcing_ratio=teacher_forcing_ratio,dropout_prob=dropout_prob)
		"""
		Function definitions
		"""
		# activation function
		self.relu = nn.ReLU()
		# dropout function
		self.dropout = nn.Dropout(p=dropout_prob)
		# loss function
		self.criterion = criterion
		"""
		Initialise weights
		"""
		self.apply(weight_init)

	# decoding latent space
	def z2x(self,z,num_steps,x=None):
		h = self.hz2x(z)
		out,dec = self.decoder(h,num_steps,x=x)
		return out,dec

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.hx2z(enc)
		z,qz = self.sampler(enc)
		out,dec = self.z2x(z,num_steps,x=x)
		return out,dec,qz,z

	def getLoss(self,x,beta=1.0,alpha=0.2):
		num_steps = x.size()[0]
		recon_x,aux_x,qz,z = self.forward(x)
		x = x.contiguous().view(-1)
		recon_x = recon_x.view(-1)
		kl = self.sampler.getLoss(qz)
		r_loss = self.criterion(recon_x,x)
		aux_loss = self.criterion(aux_x,x)
		loss = r_loss+beta*kl+alpha*aux_loss
		x = x.view(num_steps,-1)
		recon_x = recon_x.view(num_steps,-1)
		diagnostics = {'kl':kl.data,'loss':loss.data,'r_loss':r_loss.data,'aux_loss':aux_loss.data}
		data = {'recon_x':recon_x,'x':x,'qz':qz,'z':z}
		return loss,diagnostics,data


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