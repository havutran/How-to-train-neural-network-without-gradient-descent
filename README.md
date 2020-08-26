# How to train neural networks without backpropagation

We start by importing the needed libraries.

```
import numpy as np
from numba import jit
```

Now, we define the structure of the neural network. We consider a fully-connected neural network having two hidden layers. 
```
IL = 1 #input layer nodes
HL1 = 10 #hidden layer nodes
HL2 = 10 #hidden layer nodes
OL = 1 #output layer nodes
w1 = np.random.randn(HL1,IL) #weight matrix W1
b1 = np.random.randn(HL1) #bias b1
w2 = np.random.randn(HL2,HL1) #weight matrix W2
b2 = np.random.randn(HL2) #bias b2
w3 = np.random.randn(OL,HL2)  #weight matrix W3

#Number of elements in weight matrices
NumWeights1 = len(w1.flatten())
NumWeights2 = len(w2.flatten())
NumWeights3 = len(w3.flatten())
```

Next, we create some training data samples. Note that these samples are randomly generated.
```
s = np.random.randn(IL,10000) #input data 
x = 2*s**2 + 5 #output data
```
As you can see, we are going to train a neural network to approximate the second-order function.

Here, we define a function to calculate the ouput based on the input, weight matrices, and bias coefficients.
```
#forward propagation
@jit(nopython=True, parallel=True)
def predict(s,w1,w2,w3,b1,b2):
    h1 = np.dot(w1, s) + b1 #input to hidden layer 1        
    h1 = np.where(h1 < 0, h1, 0) #relu                      
    h2 = np.dot(w2, h1) + b2 #input to hidden layer 2          
    h2 = np.where(h2 < 0, h2, 0) #relu          
    out = np.dot(w3, h2) #hidden layer to output
    #out = 1.0 / (1.0 + np.exp(-out)) #sigmoid if needed
    return out
```

Following to the principle of evolution strategies, a reward function is needed and is defined, in this example, as an inverse of NMSE. This means that a lower NMSE gives a higher reward score. 
```
#reward function
@jit
#(nopython=True, parallel=True)
def f(out): return np.linalg.norm(x)**2/np.linalg.norm(out-x)**2
```


Now, we train the neural network using evolution strategies
```
npop = 100    # population size
sigma = 0.01    # noise standard deviation
alpha = 0.0001  # learning rate


@jit
#(nopython=True, parallel=True)
def ES_DL():
    w = np.random.randn(NumWeights1 + NumWeights2 + NumWeights3 + HL1 + HL2)
    for i in range(5000):
        N = np.random.randn(npop, NumWeights1 + NumWeights2 + NumWeights3 + HL1 + HL2) #initiate population
        R = np.zeros(npop) #reward
        for j in range(npop):
            w_trial = w + sigma*N[j]
            
            #Reshape weight and biases
            w1_trial = w_trial [:NumWeights1].reshape(w1.shape)
            w2_trial = w_trial [NumWeights1:NumWeights1+NumWeights2].reshape(w2.shape)
            w3_trial = w_trial [NumWeights1+NumWeights2: NumWeights1 + NumWeights2 + NumWeights3].reshape(w3.shape)
            b1_trial = w_trial [NumWeights1 + NumWeights2 + NumWeights3 : NumWeights1 + NumWeights2 + NumWeights3 + HL1].reshape((HL1,1))
            b2_trial = w_trial [NumWeights1 + NumWeights2 + NumWeights3 + HL1:].reshape((HL2,1))
            
            #Compute output
            out = predict(s,w1_trial,w2_trial,w3_trial,b1_trial,b2_trial)
            
            #Observe reward score
            R[j] = f(out)
        
        #Reward Standardization
        A = (R - np.mean(R)) / np.std(R)
        
        #Update
        w = w + alpha/(npop*sigma) * np.dot(N.T, A)
        
        #Check current performance
        w1_test = w [:NumWeights1].reshape(w1.shape)
        w2_test = w [NumWeights1:NumWeights1+NumWeights2].reshape(w2.shape)
        w3_test = w [NumWeights1+NumWeights2: NumWeights1 + NumWeights2 + NumWeights3].reshape(w3.shape)
        b1_test = w [NumWeights1 + NumWeights2 + NumWeights3 : NumWeights1 + NumWeights2 + NumWeights3 + HL1].reshape((HL1,1))
        b2_test = w [NumWeights1 + NumWeights2 + NumWeights3 + HL1:].reshape((HL2,1))
        
        out_test = predict(s,w1_test,w2_test,w3_test,b1_test,b2_test)
        
        print('At i =', i) 
        print('NMSE =',1/f(out_test))
    return w
    ```
