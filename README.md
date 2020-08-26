

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
