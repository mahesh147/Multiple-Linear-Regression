# Imporitng libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sys

# creating the regression class
class Linear_regression:
	
	def __init__(self,train_data,train_label,epochs=1000,alpha=0.1):
		self.train_data = self.feature_normalization(train_data)
		self.new_train_data = np.insert(self.train_data,0,1,axis=1)
		self.train_label = train_label
		self.weights = np.zeros((self.new_train_data.shape[1],1))
		self.epochs = epochs
		self.alpha = alpha	

  # The hypothesis     
  
	def hypothesis(self):		
		return np.dot(self.new_train_data,self.weights)	
	
  # Function to calculate error   
  
	def cost(self,predicted,labels):
		return (1/(2*np.size(labels)))*np.sum((predicted-labels)**2)		
	
#   function to get the gradient
  
	def derivative(self):
		gradient = (1/np.size(self.train_label))*np.dot(self.new_train_data.T,(self.hypothesis()-self.train_label))
		return gradient

	def train(self):
		self.cost_vals = []
		m = len(self.train_label)
		for i in range(self.epochs):
			loss = self.cost(self.hypothesis(),self.train_label)						
			self.weights = self.weights - (self.alpha)*self.derivative()							
			self.cost_vals.append(loss)
			print('\r loss: {} %'.format(loss),end=' ')
			sys.stdout.flush()		

		print('\n')	
		plt.plot(self.cost_vals)
		plt.xlabel('Iterations')
		plt.ylabel('Cost')
		plt.show()
		return self.weights,self.cost_vals

	def predict(self,data,labels):
		data = np.insert(data,0,1,axis=1)
		predicted = np.dot(data,self.weights)
		cost_total = self.cost(predicted,labels)
		return predicted,cost_total
		
	def feature_normalization(self,data):
		new_data = (data-np.mean(data))/np.std(data)
		return new_data	

if __name__ == '__main__':
  
  # Reading data and applying Multiple linear regression algorithm.   
	data = pd.read_csv("50_Startups.csv",sep=',',header=None)
	train_data = np.array(data.iloc[1:,:3])
	train_data = train_data.astype(np.float)

	train_label = np.array(data.iloc[1:,4]).reshape(-1,1)
	train_label = train_label.astype(np.float)

	gd = Linear_regression(train_data,train_label,epochs=100000,alpha=0.1)
	r = gd.train()
	print(r[0])
	print('-------------')
