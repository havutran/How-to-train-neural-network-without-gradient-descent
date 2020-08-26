# How to train neural networks without gradient descent

## Introduction

Recently, stochastic gradient descent is a well-known technique and has been considered as a universal choice for optimizing deep learning models. However, it is not the only choice [R1].

In this post, we will discuss about how to train neural networks without gradient descent. Alternatively, a method, so-called evolution strategies (ES), is promoted as a solution. It has been gaining significant attention from the research community due to its simplicity and efficiency in dealing with Reinforcement Learning problems [R2].

In principle, the ES algorithm belongs to a category of population-based optimization methods, motivated by natural selection. Natural selection gives credence to the following: the individuals with characteristics helpful to their survival can outlast and then these characteristics will be transferred to the next generation. On this basis, each iteration of ES consists of two phases: (i) generating a population of actions, and (ii) observing the returned reward and selecting "elite" actions that well fit the objective to make an update for the next iteration.

Some advantages of ES were discussed in [R3]

#### References
[R1] [Lil'Log - Evolution Strategies ](https://lilianweng.github.io/lil-log/2019/09/05/evolution-strategies.html)

[R2] [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)

[R3][Evolution Strategies - Medium.com](https://medium.com/swlh/evolution-strategies-844e2694e632)

## Implementation
We start by importing the needed libraries.

```
import numpy as np
from numba import jit
from timeit import default_timer as timer
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
@jit
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
def f(out): return np.linalg.norm(x)**2/np.linalg.norm(out-x)**2
```


Below is how we train the neural network using evolution strategies
```
npop = 100    # population size
sigma = 0.01    # noise standard deviation
alpha = 0.0001  # learning rate


@jit
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
    
Now, let's run the program and measure the execution time if needed
```
start = timer() 
w = ES_DL()
print("Execution time:", timer()-start) 
```

The obtained result is
```
At i = 4999
NMSE = 0.0001287464438334327
Execution time: 585.0522013
```
which is not bad at all.

Note that this is just a simple example and the performance can be further improved by tuning hyper-parameters properly.
