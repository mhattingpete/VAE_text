import torch
import torch.nn as nn
from torch.autograd import Variable

###########################
####    RNN MODULES    ####
###########################

class RNNEncoder(nn.Module):
	def __init__(self,input_size,rnn_size):
		super().__init__()
		# interal variable sizes
		self.rnn_size = rnn_size
		# layer definitions
		self.gru = nn.GRU(input_size,rnn_size,num_layers=1)
		
	def forward(self,x):
		h = self.initHidden()
		_,h = self.gru(x,h)
		h = h.view(-1,self.rnn_size)
		return h

	def initHidden(self):
		result = Variable(torch.zeros(1,1,self.rnn_size))
		return result

class AdvancedRNNEncoder(nn.Module):
	def __init__(self,input_size,rnn_size,bidirectional=False):
		super().__init__()
		# interal variable sizes
		self.rnn_size = rnn_size
		# layer definitions
		self.gru = nn.GRU(input_size,rnn_size,num_layers=1,bidirectional=bidirectional)
		if bidirectional:
			self.directions = 2
		else:
			self.directions = 1
		
	def forward(self,x):
		h = self.initHidden()
		o,_ = self.gru(x,h)
		return o

	def initHidden(self):
		result = Variable(torch.zeros(self.directions,1,self.rnn_size))
		return result

class RNNDecoder(nn.Module):
	def __init__(self,input_size,rnn_size,output_size,use_softmax=False):
		super().__init__()
		# internal variable sizes
		self.input_size = input_size
		self.use_softmax = use_softmax
		if use_softmax:
			self.softmax = nn.LogSoftmax(dim=1)
		step_input_size = input_size + rnn_size
		# layer definitions
		self.gru = nn.GRUCell(step_input_size,rnn_size)
		self.out = nn.Linear(rnn_size,output_size)

	def forward(self,z,num_steps):
		# save the output
		predictions = []
		# get batch size to create output
		batch_size = z.size()[0]
		# set the initial hidden state
		h = z
		# initial input tensor
		previous_output = Variable(torch.zeros(batch_size,self.input_size))
		# start running decode operation
		for di in range(num_steps):
			step_input = torch.cat([previous_output,z],1)
			h = self.gru(step_input,h)
			out = self.out(h)
			if self.use_softmax:
				out = self.softmax(out)
				_,topi = out.data.topk(self.input_size)
				previous_output = Variable(topi).type(torch.FloatTensor)
			else:
				previous_output = out
			predictions.append(out)
		# prepare the output to correct shape
		output = torch.stack(predictions)
		return output

class AdvancedRNNDecoder(nn.Module):
	def __init__(self,input_size,rnn_size,output_size,use_softmax=False):
		super().__init__()
		# internal variable sizes
		self.input_size = input_size
		self.use_softmax = use_softmax
		if use_softmax:
			self.softmax = nn.LogSoftmax(dim=1)
		step_input_size = input_size + rnn_size
		# layer definitions
		self.gru = nn.GRUCell(step_input_size,rnn_size)
		self.out = nn.Linear(rnn_size,output_size)

	def forward(self,z,num_steps):
		# save the output
		predictions = []
		# get batch size to create output
		batch_size = z.size()[1]
		# set the initial hidden state
		h = z[-1]
		# initial input tensor
		previous_output = Variable(torch.zeros(batch_size,self.input_size))
		# start running decode operation
		for di in range(num_steps):
			step_input = torch.cat([previous_output,z[di]],1)
			h = self.gru(step_input,h)
			out = self.out(h)
			if self.use_softmax:
				out = self.softmax(out)
				_,topi = out.data.topk(self.input_size)
				previous_output = Variable(topi).type(torch.FloatTensor)
			else:
				previous_output = out
			predictions.append(out)
		# prepare the output to correct shape
		output = torch.stack(predictions)
		return output

###########################
####    CNN MODULES    ####
###########################

class CNNEncoder(nn.Module):
	def __init__(self,input_size,conv_size):
		super().__init__()
		# internal variable sizes
		self.conv_size = conv_size
		# layer definitions
		self.convLayers = nn.Sequential(
			nn.Conv1d(in_channels=input_size,out_channels=conv_size,kernel_size=5),
			nn.ELU(),
			nn.Conv1d(in_channels=conv_size,out_channels=conv_size,kernel_size=5),
			nn.ELU(),
			nn.Conv1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.Conv1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.Conv1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.Conv1d(in_channels=conv_size,out_channels=conv_size,kernel_size=2)
			)

	def forward(self,x):
		x = x.transpose(0,1).transpose(1,2)
		x = self.convLayers(x)
		x = x.view(x.size()[0],-1)
		return x

class AdvancedCNNEncoder(nn.Module):
	def __init__(self,input_size,conv_size):
		super().__init__()
		# internal variable sizes
		self.conv_size = conv_size
		# layer definitions
		self.convLayers = nn.Sequential(
			nn.Conv1d(in_channels=input_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.Conv1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.Conv1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3)
			)

	def forward(self,x):
		x = x.transpose(0,1).transpose(1,2)
		x = self.convLayers(x).transpose(1,2).transpose(0,1)
		return x

class CNNDecoder(nn.Module):
	def __init__(self,input_size,conv_size,output_size,use_softmax=False):
		super().__init__()
		self.use_softmax = use_softmax
		if use_softmax:
			self.softmax = nn.LogSoftmax(dim=2)
		# layer definitions
		self.convLayers = nn.Sequential(
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=2),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=5),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=output_size,kernel_size=5)
			)

	def forward(self,z):
		# initial input tensor
		dec = z
		dec = dec.unsqueeze(2)
		dec = self.convLayers(dec).transpose(2,1).transpose(1,0)
		if self.use_softmax:
			dec = self.softmax(dec)
		return dec

class AdvancedCNNDecoder(nn.Module):
	def __init__(self,input_size,conv_size,output_size,use_softmax=False):
		super().__init__()
		self.use_softmax = use_softmax
		if use_softmax:
			self.softmax = nn.LogSoftmax(dim=2)
		# layer definitions
		self.convLayers = nn.Sequential(
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=output_size,kernel_size=3)
			)

	def forward(self,z):
		# initial input tensor
		dec = z.transpose(0,1).transpose(1,2)
		dec = self.convLayers(dec).transpose(2,1).transpose(1,0)
		if self.use_softmax:
			dec = self.softmax(dec)
		return dec

###########################
####   HYBRID MODULES  ####
###########################

class HybridDecoder(nn.Module):
	def __init__(self,input_size,conv_size,rnn_size,output_size,use_softmax=False):
		super().__init__()
		self.use_softmax = use_softmax
		if use_softmax:
			self.softmaxConv = nn.LogSoftmax(dim=2)
			self.softmax = nn.LogSoftmax(dim=1)
		# interal variable sizes
		self.input_size = input_size
		step_input_size = input_size + output_size
		# layer definitions
		self.convLayers = nn.Sequential(
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=2),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=5),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=output_size,kernel_size=5)
			)
		self.gru = nn.GRUCell(step_input_size,rnn_size)
		self.out = nn.Linear(rnn_size,output_size)

	def forward(self,z,num_steps):
		# save the output
		predictions = []
		# get batch size to create output
		batch_size = z.size()[0]
		# set the initial hidden state
		h = z
		# transform z using conv layers
		dec = z.unsqueeze(2)
		dec = self.convLayers(dec).transpose(2,1).transpose(1,0)
		if self.use_softmax:
			dec = self.softmaxConv(dec)
		# initial input tensor
		previous_output = Variable(torch.zeros(batch_size,self.input_size))
		# start running decode operation
		for di in range(num_steps):
			step_input = torch.cat([previous_output,dec[di]],1)
			h = self.gru(step_input,h)
			out = self.out(h)
			if self.use_softmax:
				out = self.softmax(out)
				_,topi = out.data.topk(self.input_size)
				previous_output = Variable(topi).type(torch.FloatTensor)
			else:
				previous_output = out
			predictions.append(out)
		# prepare the output to correct shape
		output = torch.stack(predictions)
		return output,dec

class AdvancedHybridDecoder(nn.Module):
	def __init__(self,input_size,conv_size,rnn_size,output_size,use_softmax=False):
		super().__init__()
		self.use_softmax = use_softmax
		if use_softmax:
			self.softmaxConv = nn.LogSoftmax(dim=2)
			self.softmax = nn.LogSoftmax(dim=1)
		# interal variable sizes
		self.input_size = input_size
		step_input_size = input_size + output_size
		# layer definitions
		self.convLayers = nn.Sequential(
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=conv_size,kernel_size=3),
			nn.ELU(),
			nn.ConvTranspose1d(in_channels=conv_size,out_channels=output_size,kernel_size=3),
			)
		self.gru = nn.GRUCell(step_input_size,rnn_size)
		self.out = nn.Linear(rnn_size,output_size)

	def forward(self,z,num_steps):
		# save the output
		predictions = []
		# get batch size to create output
		batch_size = z.size()[1]
		# set the initial hidden state
		h = z[-1]
		# transform z using conv layers
		dec = z.transpose(0,1).transpose(1,2)
		dec = self.convLayers(dec).transpose(2,1).transpose(1,0)
		if self.use_softmax:
			dec = self.softmaxConv(dec)
		# initial input tensor
		previous_output = Variable(torch.zeros(batch_size,self.input_size))
		# start running decode operation
		for di in range(num_steps):
			step_input = torch.cat([previous_output,dec[di]],1)
			h = self.gru(step_input,h)
			out = self.out(h)
			if self.use_softmax:
				out = self.softmax(out)
				_,topi = out.data.topk(self.input_size)
				previous_output = Variable(topi).type(torch.FloatTensor)
			else:
				previous_output = out
			predictions.append(out)
		# prepare the output to correct shape
		output = torch.stack(predictions)
		return output,dec