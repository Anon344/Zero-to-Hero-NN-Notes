## What are neural networks?
Neural networks, in their simplest form, can be represented as circuits that taken in various inputs, apply various operations to them, and get an output. These outputs are determined mathematically, and over time, various patterns emerge from these calculations. Neural networks are different from traditional computational circuits in the sense that they actively learn and improve over time, very much like a biological brain. They are also an integral part of machine learning. 

Neural Networks train through two ways: forward and back propagation. Forward propogation is when the network predicts the output of a function without knowing how the function actually works, while backpropgation is when it derives the input by working backwards from a given output.

## What is Autograd/Micrograd?
Autograd is a library written in the programming language Python. It allows for efficient and easier calculations of common mathematical operations, namely differentiation. Autograd is key in allowing neural networks to train effciently: often, neural networks need to evaluate complex functions 

## Derivative Definition
A derivative is the rate of growth or the slope at any given point a evaluated at the function. In more simple terms, given some value x evaluated for a function f, the derivative measures the mathematcial response of the function when x is increased by an arbitary value h.



## Python Classes/OOP
A class is simply a broader collection of individual objects. It contains a set of instructions or functions that every object in it can do. For example, to track a different number of dogs, we can all assign them to the class "Dog". We know that all dogs have 4 legs, are furry, and are warm-blooded. They may also be able to do different things, such as bark. These can defined as functions within Python within the class. 
The keyword class is used to define a class in Python. Often, there is also an _init_ function that intializes any individual member of the class. 
