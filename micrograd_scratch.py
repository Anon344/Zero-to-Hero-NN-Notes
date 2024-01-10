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
class Value:
    def _init_(self,data):
        self.data = data
    def _repr_(self):
        return f"Value(data = {self.data})"
    def _add_(self, other):
        out = Value(self.data + other.data)
        return out
    def _mul_(self,other):
        out = Value(self.data * other.data)
        return out


## This will result in an error if the add function is not defined
a = Value(2.0)
b = Value(3.0)
c = a + b
print(c)