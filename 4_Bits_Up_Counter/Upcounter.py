In the first part of the assignment, you will implement a simple 4-bit counter using a neural network. 
The network for the counter consists of 4 input nodes, nh hidden nodes, and 4 output nodes. 
The input to the counter is a number between 0 and 15 in binary (e.g., 11=%1011) and the output of the counter network is the binary presentation of the input number + 1 (e.g., 12=%1100). 
The architecture of the network for the counter. 
Try different number of nodes in the hidden layer from 1 to 7 hidden nodes. 
Randomly select four of the possible sixteen input patterns as validation set. 
Compare the accuracy of your system on the test set and the training set.





"""
Created on Wed May 13 19:13:10 2020

@author: Rohit
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
     [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
     [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1]]) # 16 * 4

y = np.array([[0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0],
     [0,1,0,1], [0,1,1,0], [0,1,1,1], [1,0,0,0],
     [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0]]) # 16 * 4

print(X.shape)
print(y.shape)
def sigmoid(x):
  return 1/(1+np.exp(-x))

 
def sigmoid_prime(x):
  return sigmoid(x)*(1-sigmoid(x))

INPUT_SIZE = 4
HIDDEN_SIZE = 7
OUTPUT_SIZE = 4

w_hidden = np.random.uniform(size = (INPUT_SIZE, HIDDEN_SIZE)) # 4 * 7
w_output = np.random.uniform(size = (HIDDEN_SIZE, OUTPUT_SIZE)) # 7 * 4
bias1 = np.random.uniform( size = ( X.shape[0], HIDDEN_SIZE ) )
bias2 = np.random.uniform( size = ( X.shape[0], OUTPUT_SIZE ) )

learning = 0.01

for i in range(10000):
    z = np.dot(X, w_hidden) #+ bias1# 16 * 7
    act_hidden = sigmoid(z)  # 16 * 7
    output = np.dot(act_hidden, w_output) #+ bias2 # 16 * 4
    
    error_out = output - y # 16 * 4
        
    error_hidden = np.dot(error_out, w_output.T) * sigmoid_prime(z) # 16 * 7
    w_output = w_output - learning * np.dot(act_hidden.T, error_out)
    #bias2 -= learning * error_out
   
    
    w_hidden = w_hidden - learning * np.dot(X.T, error_hidden) # 4 * 7
    #bias1 -= learning * error_hidden
    
print(output)

def model():
    X = np.array([[1,1,0,0], [1,1,0,1], [1,1,1,0], [1,0,0,0], [1,1,1,1], [0,0,0,1], [0,0,1,0], [0,0,1,1], [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1]])
    y = np.array([[1,1,0,1], [1,1,1,0], [1,1,1,1], [1,0,0,1], [0,0,0,0], [0,0,1,0], [0,0,1,1], [0,1,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0]])
    z = np.dot(X, w_hidden) #+ bias1# 16 * 7
    act_hidden = sigmoid(z)  # 16 * 7
    output = np.dot(act_hidden, w_output) #+ bias2
    print(output)
    
model()
    
    
    
    
