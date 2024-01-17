## PLEASE READ: Before starting to go through this file, make sure you are familiar with the content in the introduction, in the notes.md
## Basic imports of commonly used Python Libraries
## Math, as the name suggests, is used for common mathematical operations.
## Numpy is a more powerful library used for more complex mathematical operations.
## Mathplotlib is used for graphing and other plotting functions (essentially for mathematical visualation operations)

import math
import numpy as np
import matplotlib.pyplot as plt
## This is defining a regular function, f(x). The value of f(x) is 3x^2 - 4x + 5.
def f(x):
    return 3*x**2 - 4*x + 5

## Return the value of the function f evaluated at 3 (20)
y = f(3.0)
print(y)


## An array of numbers from -5 to 5, in increments of 0.25
## -5, 4.75, 4.5, and so on and so forth
xs = np.arange(-5,5,0.25)
print(xs)
## Evaluate the function f at each value of the array. Returns a new array with all of the values
ys = f(xs)
print(ys)
## Plot both arrays
plt.plot(xs, ys)

## Look at the notes for the definiton of a derivative and limit

h = 0.001 # Random value of h
x = 3.0
## Evaluation of the slope/derivative at x = 3.0 and h = 0.001
k = (f(x + h) - f(x))/h 
print(k)
##Do the same for -3
x = -3.0
## Evaluation of the slope/derivative at x = 3.0 and h = 0.001
k = (f(x + h) - f(x))/h 
print(k)

## Do the same for h = 0.00001
h = 0.00001
x = -3.0
## Evaluation of the slope/derivative at x = 3.0 and h = 0.001
k = (f(x + h) - f(x))/h 
print(k)

## Repeating over time, you will see that the derivative = 0 at x = 2/3


## Slightly more complex case, with additional inputs 
#Inputs
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)

## Our goal is to evalauate the derivative of d with respect to the other values

h = 0.001
# New Inputs
a = 2.0
b = -3.0
c = 10.0

## Here, the value of d2 will be less than d1, making the eventual slope negative 
## as the value went down. This is because b is negative, which means making a larger will make the overall value smaller.

d1 = a*b + c
a += h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1)/h)

## Changing b results in a positive slope, as b is initially negative, and
## adding some positive value to it will make it larger.
d1 = a*b + c
b += h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1)/h)

## In the case of c being changed, the slope will be 1 as the change will simply
## be the value of h we add to c.
d1 = a*b + c
c += h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1)/h)


## We are now moving to more complex expressions, similar to those used in neural networks.
## This will involve some slightly more complex python, so please review the notes on classes and 


## Define a new class Value, for any numerical value
## Classes are the objects that make Python an object oriented language
class Value:
    def _init_(self,data, _children=(), _op = '', label = ''):
        self.data = data
        self.grad = 0.0
        self._op = _op
        self._prev = set(_children)
        self.label = label
    ## _repr_ makes the return value prettier
    def _repr_(self):
        return f"Value(data = {self.data})"
    def _add_(self, other):
        out = Value(self.data + other.data, (self,other), '+')
        return out
    def _mul_(self,other):
        out = Value(self.data * other.data, (self,other), '*')
        return out
    def tanh(self):
       n = self.data
       t = math.exp(2en)
    


## This will result in an error if the add function is not defined
a = Value(2.0)
b = Value(3.0)
c = a + b
print(c)

a = Value(2.0)
b = Value(3.0)
c = Value(10.0)
d = a * b + c
print(d)

## Now, we will try to gain an understanding of how different values lead to one another. We will define children, which are essentially arguments to a function

a = Value(2.0)
b = Value(3.0)
c = Value(10.0)
d = a * b + c
print(d._prev) 
## Should result in the values 6 and 10

## The code below is not needed to understand the concepts. This is essentially using a library called graphviz to map out all of the operations that lead to any final value. 
## This is so you can see each individual data point
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

## Lets also add a label for better visibility

a = Value(2.0, label = 'a')
b = Value(3.0, label = 'b')
c = Value(10.0, label = 'c')
d = a * b + c; d.label = 'd'
e = d + c; e.label = 'e'
h = Value(-1.0, label = 'f')
L  = h * d

## Manual backpropgation: introducing calculus
## Derivative of L with respect to a
def lol():
    a = Value(2.0, label = 'a')
    b = Value(3.0, label = 'b')
    c = Value(10.0, label = 'c')
    d = a * b; d.label = 'd'
    e = d + c; e.label = 'e'
    h = Value(-1.0, label = 'f')
    L  = h * d
    L1 = L.data
    a = Value(2.0 + h, label = 'a')
    b = Value(3.0, label = 'b')
    c = Value(10.0, label = 'c')
    d = a * b + c; d.label = 'd'
    e = d + c; e.label = 'e'
    h = Value(-1.0, label = 'f')
    L  = h * d
    L2 = L.data
    print(L2 - L1/h)





## Manual Backpropogation Part 2, using a neuron
## Review the definition of a neuron in the notes first please!


## A neuron basically works by taking various weights and input values, combining them mathematically with one another and getting an expected output value


plt.plot(hp.arrange(-5,5,0.2), np.tanh(np.arrange(-5,5,0.2))s)

x1 = Value(2.0, label='x1')
x2 = Value(0.0, label = 'x2')
w1 = Value('-1.0', label = 'w1')
w2 = Value('1.0', label = 'w2')
b = Value(0.7, label = 'b')
x1w1 = x1 * w1; x1w1.label = 'x1w1'
x2w2 = x2 * w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1 * w1 + x2 * w2; x1w1x2w2.label = 'x1w1x2w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh()
draw_dot(n)


## Now that we have gone over the basics, I will be explaining the code
## less and will be diving more into the concepts surrounding basic backpropogation

## Please turn over to the notes for better explanation of some of the concepts. 

## We will be starting with a review of the chain rule, reviewing some common errors, and defining a loss function.


## We will no longer be reviewing the code; if you want full access to the code written during the actual lecture, please check Andrej Karpathy's Jupyter Notebook here:https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd