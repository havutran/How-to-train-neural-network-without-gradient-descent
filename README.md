

We start by importing the needed libraries.

```
import numpy as np
from numba import jit
```

Next, we create some training data samples. Note that these samples are randomly generated.

```
s = np.random.randn(IL,10000) #input data
x = 2*s**2 + 5 #output data
```
As you can see, we are going to train a neural network to approximate the second-order function.

```
IL = 1 #input layer nodes
HL1 = 10 #hidden layer nodes
HL2 = 10 #hidden layer nodes
OL = 1 #output layer nodes
w1 = np.random.randn(HL1,IL) #/ np.sqrt(IL)
b1 = np.random.randn(HL1)
w2 = np.random.randn(HL2,HL1)
b2 = np.random.randn(HL2)
w3 = np.random.randn(OL,HL2) #/ np.sqrt(HL3)

NumWeights1 = len(w1.flatten())
NumWeights2 = len(w2.flatten())
NumWeights3 = len(w3.flatten())
```
