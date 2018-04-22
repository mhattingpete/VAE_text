from torch.autograd import Variable
import torch
import numpy as np

# local imports
from visualize import showPlot,plotPredictions

def train(data_loader,dataset_size,valid_batch_loader,model,optimizer,scheduler,loss_fn,kl_loss_fn,n_iters=1000,use_softmax=False,print_every=100):
    train_data = {"x":[],"recon_x":[]}
    valid_data = {"x":[],"recon_x":[]}
    plot_losses = []
    plot_losses_valid = []
    beta_weights = np.linspace(0,0.04,n_iters)
    alpha_weight = 0.3 if model.aux_loss else 0
    aux_alternative = Variable(torch.FloatTensor([0]),requires_grad=False)
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
        for _,batch in enumerate(data_loader):
            divisor += 1
            x = Variable(batch).type(torch.FloatTensor).transpose(1,0)
            model.train()
            optimizer.zero_grad()
            if model.aux_loss:
                recon_x,qz,aux_x = model(x)
                aux_loss = loss_fn(aux_x,x)
            else:
                recon_x,qz = model(x)
                aux_loss = aux_alternative
            # calculate loss
            r_loss = loss_fn(recon_x,x)
            kl = kl_loss_fn(qz)
            loss = r_loss + beta_weight * kl + alpha_weight * aux_loss
            train_data["x"] = x.squeeze(2)
            if use_softmax:
                _,topi = recon_x.data.topk(1)
                train_data["recon_x"] = Variable(topi).squeeze(2)
            else:
                train_data["recon_x"] = recon_x.squeeze(2)
            print_loss += loss.data[0]
            print_r_loss += r_loss.data[0]
            print_kl_loss += kl.data[0]
            print_aux_loss += aux_loss.data[0]
            loss.backward()
            optimizer.step()
        if i % print_every == 0:
            plot_losses.append(print_loss/divisor)
            print('\nTrain ({0:d} {1:d}%) loss: {2:.4f} r_loss: {3:.4f} kl: {4:.4f} aux_loss: {5:.4f} beta {6:.2e}'.format(i,int(i/n_iters*100),print_loss/divisor,print_r_loss/divisor,print_kl_loss/divisor,print_aux_loss/divisor,beta_weight))
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
            aux_loss = loss_fn(aux_x,x)
        else:
            recon_x,qz = model(x)
            aux_loss = aux_alternative
        r_loss = loss_fn(recon_x,x)
        kl = kl_loss_fn(qz)
        loss = r_loss + beta_weight * kl + alpha_weight * aux_loss
        valid_data["x"] = x.squeeze(2)
        if use_softmax:
            _,topi = recon_x.data.topk(1)
            valid_data["recon_x"] = Variable(topi).squeeze(2)
        else:
            valid_data["recon_x"] = recon_x.squeeze(2)
        scheduler.step(loss.data[0])
        if i % print_every == 0:
            plot_losses_valid.append(loss.data[0])
            print('Valid ({0:d} {1:d}%) loss: {2:.4f} r_loss: {3:.4f} kl: {4:.4f} aux_loss: {5:.4f} beta {6:.2e}'.format(i,int(i/n_iters*100),loss.data[0],r_loss.data[0],kl.data[0],aux_loss.data[0],beta_weight))
    if use_softmax:
        num_classes = recon_x.shape[2]
        showPlot(plot_losses,plot_losses_valid,yu=2.5)
        plotPredictions(train_data['x'],train_data['recon_x'],xu=num_steps,yu=num_classes)
        plotPredictions(valid_data['x'],valid_data['recon_x'],xu=num_steps,yu=num_classes)
    else:
        showPlot(plot_losses,plot_losses_valid,yu=0.5)
        plotPredictions(train_data['x'],train_data['recon_x'],xu=num_steps,yu=1)
        plotPredictions(valid_data['x'],valid_data['recon_x'],xu=num_steps,yu=1)
