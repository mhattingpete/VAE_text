import torch
import torch.nn as nn
from torch.autograd import Variable
import random

from helpers import WordDropout

###########################
####    RNN MODULES    ####
###########################

class RNNEncoder(nn.Module):
	def __init__(self,input_size,hidden_size,bidirectional=True,dropout_prob=0.2):
		super().__init__()
		# interal variable sizes
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional
		# layer definitions
		self.gru = nn.GRU(input_size,hidden_size,num_layers=1,dropout=dropout_prob,bidirectional=bidirectional)

	def forward(self,x):
		h = self.initHidden()
		_,h = self.gru(x,h)
		if self.bidirectional:
			h = h.view(-1,2*self.hidden_size)
		else:
			h = h.view(-1,self.hidden_size)
		return h

	def initHidden(self):
		if self.bidirectional:
			directions = 2
		else:
			directions = 1
		result = Variable(torch.zeros(directions,1,self.hidden_size))
		return result

class RNNDecoder(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,distribute_z=True,teacher_forcing_ratio=0.5,dropout_prob=0.2):
		super().__init__()
		# interal variable sizes
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.distribute_z = distribute_z
		if self.distribute_z:
			step_input_size = input_size+hidden_size
		else:
			step_input_size = input_size
		# layer definitions
		self.gru = nn.GRUCell(step_input_size,hidden_size)
		self.h2x = nn.Linear(hidden_size,input_size)
		self.out = nn.Linear(hidden_size,output_size)
		# relu function
		self.relu = nn.ReLU()
		# dropout function
		self.dropout = nn.Dropout(p=dropout_prob)
		# softmax function
		self.softmax = nn.LogSoftmax(dim=2)

	def forward(self,z,input_length,x=None):
		# get batch size to create output
		batch_size = z.size()[0]
		# set the initial hidden state
		h = z
		# initial input tensor
		input = Variable(torch.ones(batch_size,self.input_size)*(self.output_size-1))
		# prepare output variable
		decoder_output = Variable(torch.zeros(input_length,batch_size,self.hidden_size))
		# use teacher forcing?
		use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
		# start running decode operation
		for di in range(input_length):
			if self.distribute_z:
				step_input = torch.cat([input,z],1)
			else:
				step_input = input
			h = self.gru(step_input,h)
			decoder_output[di,:,:] = h
			if use_teacher_forcing and self.training:
				input = x[di]
				input = self.dropout(input)
			else:
				input = self.relu(self.h2x(h))
		# prepare the output to correct shape
		output = self.out(decoder_output)
		# do softmax on output to normalize
		output = self.softmax(output)
		return output


###########################
####    CNN MODULES    ####
###########################


class CNNEncoder(nn.Module):
	def __init__(self,input_size,hidden_size,dropout_prob=0.2):
		super().__init__()
		# interal variable sizes
		self.hidden_size = hidden_size
		# layer definitions
		self.convLayers = nn.ModuleList([
			nn.Conv1d(in_channels=input_size,out_channels=hidden_size,kernel_size=3,stride=2),
			#nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
			nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=2),
			#nn.BatchNorm1d(hidden_size),
			nn.ReLU(),
			nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=2),
			#nn.BatchNorm1d(hidden_size),
			nn.ReLU()
			])

	def forward(self,x):
		x = x.transpose(0,1).transpose(1,2)
		for l in self.convLayers:
			x = l(x)
		x = x.view(x.size()[0],-1)
		return x

class CNNDecoder(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,dropout_prob=0.2):
		super().__init__()
		# interal variable sizes
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		# layer definitions
		self.convLayers = nn.ModuleList([
			nn.ConvTranspose1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=2),
			nn.ReLU(),
			nn.ConvTranspose1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=2),
			nn.ReLU(),
			nn.ConvTranspose1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=2),
			nn.ReLU()
			])
		self.out = nn.Linear(hidden_size,output_size)
		# softmax function
		self.softmax = nn.LogSoftmax(dim=2)

	def forward(self,z):
		# initial input tensor
		dec = z
		dec = dec.unsqueeze(2)
		for l in self.convLayers:
			dec = l(dec)
		dec = dec.transpose(1,2).transpose(0,1)
		output = self.out(dec)
		output = self.softmax(output)
		return output

class HybridDecoder(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,teacher_forcing_ratio=0.5,dropout_prob=0.2):
		super().__init__()
		# interal variable sizes
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.teacher_forcing_ratio = teacher_forcing_ratio
		step_input_size = input_size+hidden_size
		# layer definitions
		self.convLayers = nn.ModuleList([
			nn.ConvTranspose1d(in_channels=hidden_size,out_channels=256,kernel_size=3,stride=2),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.ConvTranspose1d(in_channels=256,out_channels=128,kernel_size=3,stride=2),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.ConvTranspose1d(in_channels=128,out_channels=hidden_size,kernel_size=3,stride=2),
			nn.BatchNorm1d(hidden_size),
			nn.ReLU()
			])
		self.gru = nn.GRUCell(step_input_size,hidden_size)
		self.h2x = nn.Linear(hidden_size,input_size)
		self.dec2out = nn.Linear(hidden_size,output_size)
		self.out = nn.Linear(hidden_size,output_size)
		# relu function
		self.relu = nn.ReLU()
		# dropout function
		self.dropout = WordDropout(p=dropout_prob,dummy=output_size-1)
		# softmax function
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self,z,input_length,x=None):
		# get batch size to create output
		batch_size = z.size()[0]
		# set the initial hidden state
		h = z
		# initial input tensor
		input = Variable(torch.ones(batch_size,self.input_size)*(self.output_size-2))
		# prepare output variable
		decoder_output = Variable(torch.zeros(input_length,batch_size,self.hidden_size))
		# use teacher forcing?
		use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
		dec = z
		dec = dec.unsqueeze(2)
		for l in self.convLayers:
			dec = l(dec)
		dec = dec.transpose(1,2).transpose(0,1)
		# start running decode operation
		for di in range(input_length):
			step_input = torch.cat([input,dec[di]],1)
			h = self.gru(step_input,h)
			decoder_output[di,:,:] = h
			if use_teacher_forcing and self.training:
				input = x[di]
				input = self.dropout(input)
			else:
				input = self.relu(self.h2x(h))
		# prepare the output to correct shape
		dec = self.dec2out(dec)
		output = self.out(decoder_output)
		return output,dec