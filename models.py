import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F

from layers import RNNEncoder,RNNDecoder,CNNEncoder,CNNDecoder,HybridDecoder
from sampleLayers import NormalDistributed

def correctPreds(pred,labels):
	num_steps = labels.size()[0]
	labels = labels.type(torch.LongTensor)
	labels = labels.view(-1)
	pred = pred.view(-1)
	assert pred.size()[0] == labels.size()[0]
	correct = (pred.data == labels.data).sum()
	return correct

def weight_init(m):
	if type(m) == nn.Linear:
		nn.init.xavier_normal(m.weight.data)
	elif type(m) == nn.GRUCell:
		nn.init.xavier_normal(m.weight_hh.data)
		nn.init.xavier_normal(m.weight_ih.data)
	else:
		pass

class RVAE(nn.Module):
	def __init__(self,num_classes,hidden_size,latent_size,criterion,dropout_prob=0.2,teacher_forcing_ratio=0.5):
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
		vocab_size = num_classes + 1
		self.decoder = RNNDecoder(input_size=1,hidden_size=hidden_size,output_size=vocab_size,teacher_forcing_ratio=teacher_forcing_ratio,dropout_prob=dropout_prob)
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

	def loss(self,recon_x,x):
		x = x.type(torch.LongTensor).squeeze(2)
		loss = 0
		for di in range(x.size()[0]):
			loss += self.criterion(recon_x[di],x[di])
		return loss/x.size()[0]

	def getLoss(self,x,beta=1.0,alpha=None):
		recon_x,qz,z = self.forward(x)
		kl = self.sampler.getLoss(qz).mean()
		r_loss = self.loss(recon_x,x)
		loss = r_loss+beta*kl
		_,recon_x_pred = recon_x.max(dim=2)
		correct = correctPreds(recon_x_pred,x)
		diagnostics = {'kl':kl.data,'loss':loss.data,'r_loss':r_loss.data,'aux_loss':torch.FloatTensor([0]),'correct':correct}
		data = {'recon_x':recon_x_pred,'x':x.squeeze(2),'qz':qz,'z':z}
		return loss,diagnostics,data

class CVAE(nn.Module):
	def __init__(self,num_classes,hidden_size,latent_size,criterion,dropout_prob=0.2):
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
		vocab_size = num_classes
		self.decoder = CNNDecoder(input_size=1,hidden_size=hidden_size,output_size=vocab_size,dropout_prob=dropout_prob)
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
		#self.apply(weight_init)

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

	def loss(self,recon_x,x):
		x = x.type(torch.LongTensor).squeeze(2)
		loss = 0
		for di in range(x.size()[0]):
			loss += self.criterion(recon_x[di],x[di])
		return loss/x.size()[0]

	def getLoss(self,x,beta=1.0,alpha=None):
		recon_x,qz,z = self.forward(x)
		kl = self.sampler.getLoss(qz).mean()
		r_loss = self.loss(recon_x,x)
		loss = r_loss+beta*kl
		_,recon_x_pred = recon_x.max(dim=2)
		correct = correctPreds(recon_x_pred,x)
		diagnostics = {'kl':kl.data,'loss':loss.data,'r_loss':r_loss.data,'aux_loss':torch.FloatTensor([0]),'correct':correct}
		data = {'recon_x':recon_x_pred,'x':x.squeeze(2),'qz':qz,'z':z}
		return loss,diagnostics,data

class HybridVAE(nn.Module):
	def __init__(self,num_classes,hidden_size,latent_size,criterion,dropout_prob=0.2,teacher_forcing_ratio=0.5):
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
		vocab_size = num_classes + 2
		self.decoder = HybridDecoder(input_size=1,hidden_size=hidden_size,output_size=vocab_size,teacher_forcing_ratio=teacher_forcing_ratio,dropout_prob=dropout_prob)
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
		#self.apply(weight_init)

	# decoding latent space
	def z2x(self,z,num_steps,x=None):
		h = self.hz2x(z)
		h = self.relu(h)
		h = self.dropout(h)
		out,dec = self.decoder(h,num_steps,x=x)
		return out,dec

	def forward(self,x):
		num_steps = x.size()[0]
		enc = self.encoder(x)
		enc = self.relu(enc)
		enc = self.dropout(enc)
		enc = self.hx2z(enc)
		enc = self.relu(enc)
		enc = self.dropout(enc)
		z,qz = self.sampler(enc)
		out,dec = self.z2x(z,num_steps,x=x)
		return out,dec,qz,z

	def loss(self,recon_x,x):
		x = x.type(torch.LongTensor).squeeze(2)
		loss = 0
		for di in range(x.size()[0]):
			loss += self.criterion(recon_x[di],x[di])
		return loss/x.size()[0]

	def getLoss(self,x,beta=1.0,alpha=0.2):
		recon_x,recon_x_pre,qz,z = self.forward(x)
		kl = self.sampler.getLoss(qz).mean()
		r_loss = self.loss(recon_x,x)
		aux_loss = self.loss(recon_x_pre,x)
		loss = r_loss+beta*kl+alpha*aux_loss
		diagnostics = {'kl':kl.data,'loss':loss.data,'r_loss':r_loss.data,'aux_loss':aux_loss.data}
		_,recon_x_pred = recon_x.max(dim=2)
		data = {'recon_x':recon_x_pred,'x':x.squeeze(2),'qz':qz,'z':z}
		return loss,diagnostics,data