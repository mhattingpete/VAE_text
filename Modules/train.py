from torch.autograd import Variable
import torch
import numpy as np

# local imports
from visualize import showPlot,plotPredictions

def train(data_loader,valid_data_loader,model,optimizer,scheduler,loss_fn,kl_loss_fn,n_iters=1000,use_softmax=False,max_beta=1e-2,print_every=100,plot_pred=True,max_batches=None,print_partial=False):
	train_data = {"x":[],"recon_x":[]}
	valid_data = {"x":[],"recon_x":[]}
	plot_losses = []
	plot_losses_valid = []
	beta_weights = np.linspace(0,max_beta,n_iters)
	alpha_weight = 0.3 if model.aux_loss else 0
	aux_alternative = Variable(torch.FloatTensor([0]),requires_grad=False)
	valid_batch_loader = iter(valid_data_loader)
	# start training and evaluating
	for i in range(1,n_iters+1):
		#if i > beta_start_inc and beta_weight < beta_max:
		#    beta_weight += beta_inc
		beta_weight = float(beta_weights[i-1])
		# train
		print_loss = 0
		print_r_loss = 0
		print_kl_loss = 0
		print_aux_loss = 0
		divisor = 0
		for batch_idx,batch in enumerate(data_loader):
			print(batch_idx,batch)
			if (max_batches and divisor >= max_batches):
				break
			divisor += 1
			x = Variable(batch).type(torch.FloatTensor).transpose(1,0)
			model.train()
			optimizer.zero_grad()
			if model.aux_loss:
				recon_x,qz,aux_x = model(x)
				aux_loss = -loss_fn(aux_x,x)
			else:
				recon_x,qz = model(x)
				aux_loss = aux_alternative
			# calculate loss
			r_loss = -loss_fn(recon_x,x)
			if model.kl_loss:
				kl = model.kl
			else:
				kl = kl_loss_fn(qz)
			elbo_loss = (r_loss - (beta_weight * kl)) + (alpha_weight * aux_loss)
			loss = -elbo_loss
			train_data["x"] = x.squeeze(2)
			if use_softmax:
				_,topi = recon_x.data.topk(1)
				train_data["recon_x"] = Variable(topi).squeeze(2)
			else:
				train_data["recon_x"] = recon_x.squeeze(2)
			print_loss += elbo_loss.item()
			print_r_loss += r_loss.item()
			print_kl_loss += kl.item()
			print_aux_loss += aux_loss.item()
			loss.backward()
			optimizer.step()
			if max_batches and print_partial:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					i,batch_idx*len(batch),max_batches*len(batch),
					100.*batch_idx/max_batches,elbo_loss.item()))
			elif print_partial:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					i,batch_idx*len(batch),len(data_loader.dataset),
					100.*batch_idx/len(data_loader),elbo_loss.item()))
		if i % print_every == 0:
			plot_losses.append(print_loss/divisor)
			print('\nTrain ({0:d} {1:d}%) elbo: {2:.4f} r_loss: {3:.4f} kl: {4:.4f} aux_loss: {5:.4f} beta {6:.2e}'.format(i,int(i/n_iters*100),print_loss/divisor,print_r_loss/divisor,print_kl_loss/divisor,print_aux_loss/divisor,beta_weight))
		# eval
		try:
			batch = next(valid_batch_loader)
		except StopIteration:
			valid_batch_loader = iter(valid_data_loader)
			batch = next(valid_batch_loader)
		x = Variable(batch).type(torch.FloatTensor).transpose(1,0)
		model.eval()
		if model.aux_loss:
			recon_x,qz,aux_x = model(x)
			aux_loss = -loss_fn(aux_x,x)
		else:
			recon_x,qz = model(x)
			aux_loss = aux_alternative
		r_loss = -loss_fn(recon_x,x)
		if model.kl_loss:
			kl = model.kl
		else:
			kl = kl_loss_fn(qz)
		elbo_loss = (r_loss - (beta_weight * kl)) + (alpha_weight * aux_loss)
		loss = -elbo_loss
		valid_data["x"] = x.squeeze(2)
		if use_softmax:
			_,topi = recon_x.data.topk(1)
			valid_data["recon_x"] = Variable(topi).squeeze(2)
		else:
			valid_data["recon_x"] = recon_x.squeeze(2)
		scheduler.step(loss.item())
		if i % print_every == 0:
			plot_losses_valid.append(elbo_loss.item())
			print('Valid ({0:d} {1:d}%) elbo: {2:.4f} r_loss: {3:.4f} kl: {4:.4f} aux_loss: {5:.4f} beta {6:.2e}'.format(i,int(i/n_iters*100),elbo_loss.item(),r_loss.item(),kl.item(),aux_loss.item(),beta_weight))
	if use_softmax:
		num_classes = recon_x.shape[2]
		num_steps = recon_x.shape[0]
		showPlot(plot_losses,plot_losses_valid,yl=-2.5)
		if plot_pred:
			plotPredictions(train_data['x'],train_data['recon_x'],xu=num_steps,yu=num_classes)
			plotPredictions(valid_data['x'],valid_data['recon_x'],xu=num_steps,yu=num_classes)
	else:
		num_steps = recon_x.shape[0]
		showPlot(plot_losses,plot_losses_valid,yl=-2.5)
		if plot_pred:
			plotPredictions(train_data['x'],train_data['recon_x'],xu=num_steps,yu=1)
			plotPredictions(valid_data['x'],valid_data['recon_x'],xu=num_steps,yu=1)

def train_twitter(data_loader,valid_data_loader,model,optimizer,scheduler,loss_fn,kl_loss_fn,n_iters=1000,use_softmax=False,max_beta=1e-2,print_every=100,plot_pred=True,max_batches=None,print_partial=False):
	train_data = {"x":[],"recon_x":[]}
	valid_data = {"x":[],"recon_x":[]}
	plot_losses = []
	plot_losses_valid = []
	beta_weights = np.linspace(0,max_beta,n_iters)
	alpha_weight = 0.3 if model.aux_loss else 0
	aux_alternative = Variable(torch.FloatTensor([0]),requires_grad=False)
	valid_batch_loader = iter(valid_data_loader)
	# start training and evaluating
	for i in range(1,n_iters+1):
		#if i > beta_start_inc and beta_weight < beta_max:
		#    beta_weight += beta_inc
		beta_weight = float(beta_weights[i-1])
		# train
		print_loss = 0
		print_r_loss = 0
		print_kl_loss = 0
		print_aux_loss = 0
		divisor = 0
		for batch_idx,batch in enumerate(data_loader):
			if (max_batches and divisor >= max_batches):
				break
			divisor += 1
			x = Variable(torch.FloatTensor(batch[0])).unsqueeze(2).transpose(1,0)
			model.train()
			optimizer.zero_grad()
			if model.aux_loss:
				recon_x,qz,aux_x = model(x)
				aux_loss = -loss_fn(aux_x,x)
			else:
				recon_x,qz = model(x)
				aux_loss = aux_alternative
			# calculate loss
			r_loss = -loss_fn(recon_x,x)
			if model.kl_loss:
				kl = model.kl
			else:
				kl = kl_loss_fn(qz)
			elbo_loss = (r_loss - (beta_weight * kl)) + (alpha_weight * aux_loss)
			loss = -elbo_loss
			train_data["x"] = x.squeeze(2)
			if use_softmax:
				_,topi = recon_x.data.topk(1)
				train_data["recon_x"] = Variable(topi).squeeze(2)
			else:
				train_data["recon_x"] = recon_x.squeeze(2)
			print_loss += elbo_loss.item()
			print_r_loss += r_loss.item()
			print_kl_loss += kl.item()
			print_aux_loss += aux_loss.item()
			loss.backward()
			optimizer.step()
			if max_batches and print_partial:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					i,batch_idx*len(batch),max_batches*len(batch),
					100.*batch_idx/max_batches,elbo_loss.item()))
			elif print_partial:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					i,batch_idx*len(batch),len(data_loader.dataset),
					100.*batch_idx/(len(data_loader.dataset)*len(batch)),elbo_loss.item()))
		if i % print_every == 0:
			plot_losses.append(print_loss/divisor)
			print('\nTrain ({0:d} {1:d}%) elbo: {2:.4f} r_loss: {3:.4f} kl: {4:.4f} aux_loss: {5:.4f} beta {6:.2e}'.format(i,int(i/n_iters*100),print_loss/divisor,print_r_loss/divisor,print_kl_loss/divisor,print_aux_loss/divisor,beta_weight))
		# eval
		try:
			batch = next(valid_batch_loader)[0]
		except StopIteration:
			valid_batch_loader = iter(valid_data_loader)
			batch = next(valid_batch_loader)[0]
		x = Variable(torch.FloatTensor(batch)).unsqueeze(2).transpose(1,0)
		model.eval()
		if model.aux_loss:
			recon_x,qz,aux_x = model(x)
			aux_loss = -loss_fn(aux_x,x)
		else:
			recon_x,qz = model(x)
			aux_loss = aux_alternative
		r_loss = -loss_fn(recon_x,x)
		if model.kl_loss:
			kl = model.kl
		else:
			kl = kl_loss_fn(qz)
		elbo_loss = (r_loss - (beta_weight * kl)) + (alpha_weight * aux_loss)
		loss = -elbo_loss
		valid_data["x"] = x.squeeze(2)
		if use_softmax:
			_,topi = recon_x.data.topk(1)
			valid_data["recon_x"] = Variable(topi).squeeze(2)
		else:
			valid_data["recon_x"] = recon_x.squeeze(2)
		scheduler.step(loss.item())
		if i % print_every == 0:
			plot_losses_valid.append(elbo_loss.item())
			print('Valid ({0:d} {1:d}%) elbo: {2:.4f} r_loss: {3:.4f} kl: {4:.4f} aux_loss: {5:.4f} beta {6:.2e}'.format(i,int(i/n_iters*100),elbo_loss.item(),r_loss.item(),kl.item(),aux_loss.item(),beta_weight))
	if use_softmax:
		num_classes = recon_x.shape[2]
		num_steps = recon_x.shape[0]
		showPlot(plot_losses,plot_losses_valid,yl=-2.5)
		if plot_pred:
			plotPredictions(train_data['x'],train_data['recon_x'],xu=num_steps,yu=num_classes)
			plotPredictions(valid_data['x'],valid_data['recon_x'],xu=num_steps,yu=num_classes)
	else:
		num_steps = recon_x.shape[0]
		showPlot(plot_losses,plot_losses_valid,yl=-2.5)
		if plot_pred:
			plotPredictions(train_data['x'],train_data['recon_x'],xu=num_steps,yu=1)
			plotPredictions(valid_data['x'],valid_data['recon_x'],xu=num_steps,yu=1)
