import numpy as np
import csv

def return_mnist_as_ndarray():
	# This function reads in the MNIST dataset out of a CSV file, and stores the results in a numpy ndarray for further processing.  Note that we are storing the probability distributions of pixel intensities (dividing by 255).
	# These probability distributions will be sampled later when we construct minibatches.
	with open('mnist_train.csv', 'r') as mnist_csv:
		d = np.zeros((60000,784))
		row_index = 0
		for data in csv.reader(mnist_csv):
			d[row_index,:] = np.array(data[1:],dtype='float32')/255
			row_index += 1
		return d