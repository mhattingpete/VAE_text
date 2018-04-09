import matplotlib.pyplot as plt
import numpy as np

from ToyDatasets.timeSeries import Sinusoids

def plotDataset(dataset,xu=None,yu=1,max_print=100):
	fig = plt.figure()
	i = 1
	n = int(np.floor(np.sqrt(max_print)))
	nsq = n*n
	for s in dataset:
		if i == 1 and xu == None:
			xu = s.shape[0]
		if i > nsq:
			break
		ax = fig.add_subplot(n,n,i)
		ax.plot(s)
		ax.set_xlim([0,xu])
		ax.set_ylim([0,yu])
		ax.axis('off')
		i += 1

def plotPredictions(x,recon_x,xu=None,yu=1,max_print=100):
	x = x.data.numpy()
	recon_x = recon_x.data.numpy()
	fig = plt.figure()
	n = int(np.floor(np.sqrt(max_print)))
	nsq = n*n
	for i in range(x.shape[1]):
		if i == 1 and xu == None:
			xu = x[:,i].shape[0]
		if i >= nsq:
			break
		ax = fig.add_subplot(n,n,i+1)
		ax.plot(x[:,i],'r')
		ax.plot(recon_x[:,i],'b')
		ax.set_xlim([0,xu])
		ax.set_ylim([0,yu])
		ax.axis('off')

def plotSamples(x,xu=None,yu=1,max_print=100):
	x = x.data.numpy()
	fig = plt.figure()
	n = int(np.floor(np.sqrt(max_print)))
	nsq = n*n
	for i in range(x.shape[1]):
		if i == 1 and xu == None:
			xu = x[:,i].shape[0]
		if i >= nsq:
			break
		ax = fig.add_subplot(n,n,i+1)
		ax.plot(x[:,i],'r')
		ax.set_xlim([0,xu])
		ax.set_ylim([0,yu])
		ax.axis('off')

def showPlot(points,points_valid,yu=200):
	plt.figure()
	plt.plot(points,'ro')
	plt.plot(points_valid,'b*')
	plt.ylim([0,yu])

if __name__ == '__main__':
	num_steps = 10
	dataset = Sinusoids(num_steps)
	plotDataset(dataset,xu=num_steps,max_print=100)
	plt.show()