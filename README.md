# How to train neural networks without backpropagation

We start by importing the needed libraries.

```
import numpy as np
from numba import jit
```

Now, we define the stucture of the neural network. We consider a fully-connected neural network having two hidden layers. 
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
#(nopython=True, parallel=True)
def predict(s,w1,w2,w3,b1,b2):
    h1 = np.dot(w1, s) + b1 #input to hidden layer 1        
    h1 = np.where(h1 < 0, h1, 0) #relu                      
    h2 = np.dot(w2, h1) + b2 #input to hidden layer 2          
    h2 = np.where(h2 < 0, h2, 0) #relu          
    out = np.dot(w3, h2) #hidden layer to output
    #out = 1.0 / (1.0 + np.exp(-out)) #sigmoid if needed
    return out
```
