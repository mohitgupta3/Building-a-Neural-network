# Building-a-Neural-network

Learn to Build a Neural Network from Scratch.

## Hello world,
In this repo, we will learn about Neural networks and also how to build Neural Networks by playing with some toy codes.

In simple words, Neural networks are "computer system modelled on the human brain and nervous system".
Neural Networks comprizes of Artificial neurons. 

## But, what the hack is an Artificial neuron?
Well, there is no rocket science,
Consider this image below to understand.
![Relation b/w Biological and Artificial neuron](neuron.png)

In that Artificial neuron, you can see an encircled equation in the middle, that equation is nothing but an Activation function, if you want to learn about them [here](https://github.com/Optimist-Prime/A-Story-of-Activation-Functions) is the link.

Assuming that you totally understood above image, we may proceed further.

## Now lets take look at some Mathematics.

![Some Mathematical Notations](Math Notation Cheat Sheet.png)

### Now lets build our first neural network in python.
only dependency here is numpy.
Refer to code _Simple_NN.py_

```python
import numpy as np

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
```
The code above will be used to predict the output of three inputs in the table given below.
![Table to train Simple Network](Table1.PNG)

We could solve this problem by simply measuring statistics between the input values and the output values. If we did so, we would see 
that the leftmost input column is perfectly correlated with the output. Backpropagation, in its simplest form, measures statistics like
this to make a model. Let's jump right in and use it to do this.

## 2-Layer Neural Network

Refer to the code _Two_L_NN.py_.
```python
import numpy as np

# sigmoid activation function
def nonlin(x,deriv=False):
if(deriv==True):
return x*(1-x)
return 1/(1+np.exp(-x))

# input dataset
X = np.array([  [0,0,1],
[0,1,1],
[1,0,1],
[1,1,1] ])

# output dataset           
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
for iter in range(10000):
# forward propagation
  l0 = X
  l1 = nonlin(np.dot(l0,syn0))
  # how much did we miss?
  l1_error = y - l1
  # multiply how much we missed by the
  # slope of the sigmoid at the values in l1
  l1_delta = l1_error * nonlin(l1,True)
  # update weights
  syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1
```

Output of Above Function will be as.
```
Output After Training:
[[ 0.00966449]
 [ 0.00786506]
 [ 0.99358898]
 [ 0.99211957]]
```

As you can see in the "Output After Training", it works!!! (A GREAT ROUND OF APPLAUSE FOR YA!!!) Before I describe processes, 
I recommend playing around with the code to get an intuitive feel for how it works.

Refer to _Two_L_NN.ipynb_ file. Above Code is explained there in detail.

## 3 Layered Neural network

```python
import numpy as np


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
        return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]])

y = np.array(
    [[0],
    [1],
    [1],
    [0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    # how much did we miss the target value?
    l2_error = y - l2
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))
       # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
```

Output will be.

```
Error:0.496410031903
Error:0.00858452565325
Error:0.00578945986251
Error:0.00462917677677
Error:0.00395876528027
Error:0.00351012256786
```
This code is Explained in _Three_L_NN.ipynb_.

Not making this __README.md__ too log to read, lets end this Tutorial here.

I hope that, today you learnt a lot about Neural Networks. 

__Wait!!__ Is it all about Neural networks?
Are you a master now?

No, It's just a beginning, there is a lot of stuff to learn. However you get a nice start, so lets smile togather and start _coading and Experimenting_. Yeahh, also learning.

you may [mail](mohit.gupta2jly@gmail.com) me if you have any issues. Be specific and I will try to write in as easy as possible and brief way, cuz I hate typing.

__NOTE :__ We will Learn to build Convolutional-Neural-Networks in next Repo, stay tuned.
