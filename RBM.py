import numpy as np
import matplotlib.pyplot as plt
import random

class RBM(object):

	def __init__(self,num_visible,num_hidden):
		#Initializes the weights and biases of a restricted Boltzmann machine randomly by sampling from a Gaussian distribution
		self.num_visible = num_visible
		self.num_hidden = num_hidden
		self.W = np.random.normal(0,0.05,(num_visible,num_hidden))
		self.a = np.random.normal(-0.5,0.05,(1,num_visible))
		self.b = np.random.normal(-0.2,0.05,(1,num_hidden))

	def _logistic(self,x):
		return np.divide(1, 1+np.exp(-x))	

	def grad_E(self,v,h):
		#This function returns the average of the partial derivatives of energy with respect to all the parameters, where the average is taken over some mini-batch of samples
		#INPUTS:
			# v: An ndarray of dimension (batch_size x num_visible). A state of the visible nodes.  Each row contains a different training example from a mini-batch.
			# h: An ndarray of dimension (batch_size x num_hidden).  A state of the hidden nodes.  Each row contains a different training example from a mini-batch.
		#OUTPUTS:
			# dW: a numpy array of size (num_visible,num_hidden) which is the partial derivative of energy with respect to the (i,j)'th weight evaluated at (v,h)
			# da: a numpy array of size (1, num_visible) which is the partial derivative of energy with respect to the visible biases evaluated at (v,h)
			# db: a numpy array of size (1, num_hidden) which is the partial derivative of energy with respect to the hidden biases evaluated at (v,h)
		
		dW = -np.mean(v[:,:,np.newaxis]*h[:,np.newaxis,:],axis=0)
		da = -np.mean(v,axis=0)
		db = -np.mean(h,axis=0)

		return dW, da, db

	#This function is not used for anything at the moment.
	def E(self,v,h):
		#Computes the energy of the state (v,h)
		#assert type(v) is np.ndarray, "Input 'v' must be a numpy array"
		return -v @ self.W @ h.T - v @ self.a.T - h @ self.h.T

	def sample_hidden_given_visible(self,d):
		#This function samples the conditional distribution p(h|v,W,a,b)
		#INPUTS:
			# d: Is a numpy array containing the minibatch of data to input into the RBM.  It should be formatted so that the rows are each individual data entries, and the columns represent the visible variables
		#OUTPUT: 
			# h_sample is a numpy ndarray of dimension (num rows d x num_hidden)

		#Compute the probabilities of the hidden variables given the data
		p_given_data = self._logistic(d @ self.W + self.b)
		#Sample the distributions corresponding to each element of the minibatch and return the result
		return np.random.binomial(1,p_given_data)

		#This code is slow.  Optimized above
		#h_sample = np.array([[1 if random.random() <= row[i] else 0 for i in range(len(row))] for row in p_given_data])

	def sample_visible_given_hidden(self,h):
		#This function samples the conditional distribution p(v|h,W,a,b).  Very similar to the previous function.  Only real difference is a transpose in the matrix multiplication of the logistic function, and using the visible bias

		#Compute the probabilities of the hidden variables given the data
		p_given_h = self._logistic(h @ self.W.T + self.a)
		#Sample the distribution
		return np.random.binomial(1,p_given_h)

		#The below was originally what I was doing, but it turned out to be very slow.  It was replaced with the above code. Motto: Vectorization is good!
		#v_sample = np.array([[1 if random.random() <= row[i] else 0 for i in range(len(row))] for row in p_given_h])
	
	def sample_from_joint(self,v0=None,num_iters = 2):
		#This function uses block Gibbs sampling to generate a sample from the joint distribution P(v,h|W,a,b)
		#INPUTS:
			#v0 (optional): np.ndarray is a starting point for the Gibbs sampling. (For the purposes of the specific problem, v0 will represent an element of the MNST dataset).
							# When not specified, v0 is initialized randomly.  
			#num_iters (optional): integer argument that controls iteration depth for the Gibbs sampling
		#OUTPUTS:
			# (v_free, h_free) is the approximate sample from the joint distribution P(v,h|W,a,b)
			# h_data is a sample of P(h|v0,W,a,b)
			
			if v0 is None:
				v = np.random.binomial(n=1, p=0.5, size=(1,self.num_visible))
			else:
				v = v0	

			h_data = self.sample_hidden_given_visible(v)
			h_free = h_data

			#Perform the Gibbs sampling
			for _ in range(num_iters):
				#Propagate forward
				v_free = self.sample_visible_given_hidden(h_free)
				#Propagate backward
				h_free = self.sample_hidden_given_visible(v_free)

			return v_free, h_free, h_data

	def save(self, file_name):
		#This function records the weights and biases of an RBM into a .csv file
		np.savetxt(file_name + ' a.csv',self.a,delimiter=",")
		np.savetxt(file_name + ' b.csv',self.b,delimiter=",")
		np.savetxt(file_name + ' W.csv',self.W,delimiter=",")

	def load(self, file_name):
		#This function updates the current RBM to one that has been saved to a file previously.
		self.a = np.loadtxt(file_name + ' a.csv',delimiter=",").reshape((1,self.num_visible))
		self.b = np.loadtxt(file_name + ' b.csv',delimiter=",").reshape((1,self.num_hidden))
		self.W = np.loadtxt(file_name + ' W.csv',delimiter=",")

	def plot_marginal_sample(self,gibbs_sample_no=2):

		#Perform Gibbs sampling starting from a random input
		v = self.sample_from_joint(num_iters=gibbs_sample_no)[0]

		#Convert v to a pixel format
		v = (v*255).reshape((28,28))

		# Plot
		plt.title('Here is a sample from our Restricted Boltzmann Machine')
		plt.imshow(v, cmap='gray')
		plt.show()

	def train(self,data,num_epochs=10, eps=0.001,batch_size=20):
		#INPUTS: 
			#data: The dataset to train the RBM on
			#RBM: A class object of type 'RBM' on which the training is done
		#OPTIONAL INPUTS:
			#batch_size: Size of mini-batches to use in stochastic gradient descent
			#num_epochs: Number of times to iterate over the entire data set
			#eps: Learning rate for gradient descent.
			
		#Debugging: Get test mini-batch of 100 examples
		#tmb = np.random.binomial(1, data[0:99,:])

		num_batches = int(len(data[:,0])/batch_size)

		for epoch in range(num_epochs):
			print('Epoch '+str(epoch+1))
			#Randomly reorder the dataset
			np.random.shuffle(data)

			#Debugging: Check activation of hidden nodes
			#self.plot_hidden_activation_probabilities(tmb)

			#Iterate over all the minibatches
			for i in range(num_batches):
				
				#Slow. Was changed.
				#mini_batch = np.array([[1 if random.random() <= p else 0 for p in data[row,:]] for row in range(batch_size*i, batch_size*(i+1))])
				#Get the mini_batch
				mini_batch = np.random.binomial(1, data[batch_size*i:batch_size*(i+1),:])

				#Sample the joint distribution and p(h|data,W,a,b) using the mini batch
				v_free, h_free, h_data = self.sample_from_joint(v0=mini_batch,num_iters=2)

				#Compute the derivatives of log(p(v|W,a,b)) using the minibatch stochastic gradient descent.  
				#Contribution from the joint part
				dW_pos, da_pos, db_pos = self.grad_E(v_free,h_free)
				#Contribution from the conditional part
				dW_neg, da_neg, db_neg = self.grad_E(mini_batch,h_data) 

				#Update the parameters.  The in-place addition saves time
				self.W += eps*(dW_pos-dW_neg)
				self.a += eps*(da_pos-da_neg)
				self.b += eps*(db_pos-db_neg)
				
				#Give some output to monitor progress
				if i*batch_size % 20000 == 0:
					print('------------------------------------------')
					print('Currently training on sample: ' + str(i*batch_size) + ' out of ' + str(data.shape[0]) + ' in epoch ' + str(epoch+1))
					print('Sample weight[0,0]: ' + str(self.W[0,0]))
					print('Sample visible bias[0]: ' + str(self.a[0,0]))
					print('Sample hidden bias[0]: ' + str(self.b[0,0]))


############################################## THE FOLLOWING FUNCTIONS ARE USED FOR DEBUGGING AND HYPERPARAMETER TUNING ###################################################################		

	def plot_hidden_activation_probabilities(self, mini_batch):
		#This functions is used for debugging the training.  It makes a plot of the hidden neuron activations across a number of training examples.
		pixels = self._logistic(mini_batch @ self.W + self.b)*256
		plt.figure(1)
		plt.subplot(211)
		plt.imshow(pixels,cmap='gray')
		plt.subplot(212)
		plt.hist(pixels.reshape((1,self.num_hidden*mini_batch.shape[0]))[0,:]/255,bins='auto')
		plt.show()		

	def plot_filters(self,num_filters):
		#This function plots the learned weight filters.
		plt.figure(1)
		#Want 5 filters per row
		num_rows = int(np.rint(num_filters/5))
		
		for i in range(num_rows*5):
			plt.subplot(num_rows,5,i+1)
			pixels = self.W[:,i].reshape((28,28))
			plt.imshow(pixels,cmap='gray')


		plt.show()

	def plot_weight_bias_histogram(self):
		#This function is used for debugging the hyperparameters.  It plots a histogram of the weights and biases.
		plt.figure(1)

		#Weights
		plt.subplot(311)
		plt.title('Weights')
		plt.hist(self.W.reshape(1,self.num_visible*self.num_hidden)[0,:],bins='auto')
		#Visible biases
		plt.subplot(312)
		plt.title('Visible biases')
		plt.hist(self.a[0,:],bins='auto')
		#Hidden biases
		plt.subplot(313)
		plt.title('Hidden biases')
		plt.hist(self.b[0,:],bins='auto')

		plt.show()