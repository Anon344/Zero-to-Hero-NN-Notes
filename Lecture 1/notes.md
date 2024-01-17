# Lecture 1: What is a neural network? Learning by building Micrograd

## Intoduction
<p>These collection of files is guided walkthrough/tutorial intended to enable the average person to understand artificial intelligence and machine learning at deep technical level. The only prerequisites that one must possess to understand this guide are a legitimate interest in how the generative AI tools we use today work, and some level of familiarity with basic mathematics. </p>

<p> To follow through this guide as a programmer, make sure that you have the latest version of Python installed. Python is a generalized, object-oriented programming language. It is often used in machine learning, artificial intelligence, and more. We will talk more about Python later. You can download Python [here] (https://www.python.org/downloads/) for your personal operating system. Note that you can choose to instead simply follow the comments in the code and follow along to these lecture notes. If that is what you would rather do, feel free to skip the next section</p>

<p> Next, make sure to download the required libraries. Libraries in Python are like abstracted collections of code that make it easier for the user to perform various operations. For example, a library could define sum, so that instead of typing in 1 + 3, you could simply type sum(1,3). Over time, this makes creating complex projects that require a lot of calculations a lot easier. 
To download the libraries, open your terminal. On most operating systems, you can do this by going to the search bar and typing CMD. Once there, type in the following command:
pip3 install math, numpy, mathplotlib
</p>

Now, let's get stared with the content! 
## What are neural networks?
Neural networks, in their simplest form, can be represented as circuits that taken in various inputs, apply various operations to them, and get an output. These outputs are determined mathematically, and over time, various patterns emerge from these calculations. Neural networks are different from traditional computational circuits in the sense that they actively learn and improve over time, very much like a biological brain. They are also an integral part of machine learning. 

Neural Networks train through two ways: forward and back propagation. Forward propogation is when the network predicts the output of a function without knowing how the function actually works, while backpropgation is when it derives the input by working backwards from a given output.
## What is Autograd/Micrograd?
Autograd is a library written in the programming language Python. It allows for efficient and easier calculations of common mathematical operations, namely differentiation. Autograd is key in allowing neural networks to train effciently: often, neural networks need to evaluate complex functions, and autograd allows for these functions to be evaluated in an abstract manner. Micrograd, which is the focus of this lecture, is essentially a simplified version of autograd designed for smaller, simpler neural networks to be easily built. 

## Derivative Definition
A derivative is the rate of growth or the slope at any given point a evaluated at the function. In more simple terms, given some value x evaluated for a function f, the derivative measures the mathematcial response of the function when x is increased by an arbitary value h.



## Python Classes/OOP
A class is simply a broader collection of individual objects. It contains a set of instructions or functions that every object in it can do. For example, to track a different number of dogs, we can all assign them to the class "Dog". We know that all dogs have 4 legs, are furry, and are warm-blooded. They may also be able to do different things, such as bark. These can defined as functions within Python within the class. 
The keyword class is used to define a class in Python. Often, there is also an _init_ function that intializes any individual member of the class. 

## What is a neuron?
<p>Neural Networks are comprised of neurons. A neuron simply consists of some inputs, with the synapse being comprised of weights. The cell body receives the product (weight * input) of each weight and input combination. The body also has its own bias, which is mathematically combined with the weight-input product. This result then goes through an activation function, which is typically a "squashing function". Examples of squashing functions include tanh, sinoid, etc. Let's see this in action in the code</p>

## Chain Rule Review
<p>When we are looping back across multiple nodes (ie, trying to find the individual gradients for value that is the final result of multiple operations and combinations), we have to use the chain rule. The chain rule essentially helps us find the final derivative starting from a value that is the result of multiple functions, or a composite function. The chain rule states that, for a composite function f(g(x)) (note that the function f is acting on the output of the function g, that is, f's input is g's output), the derivative is f'g(x) * g'. We are taking the derivative of f while keeping g constant, and then multiplying that by the derivative of g. For example, the derivative of 
(x-1)^2 will be 2(x-1) * 1 (2x is the derivative of x^2, here x is the function x -1, while 1 is the derivative of x-1). Recall that the derivative essentially measures the rate of growth of a function. For a ML model, it allows us to measure the gradient, which is a good measure of how accurately the model is performing over time.<p>  

## Mistakes, topological sort, and looking forward
<p> When defining a class in Python, we have to make sure that all of the possible operations, along with all of their potential inputs, are properly accounted for. In our current definition of the Value class (review the class Value in the code-based notes), we have assumed that all operations will automatically be done between two members of the predefined class. That means, when we try, for example, to add a, which is a member of the Value class, and 2.0, which is just a regular number, the add operation will attempt to treat 2.0 like it is a member of the Value class, which will result in an error. To fix this, we can define alternative functions such as r_add, r_mul, and so and so forth within the value class. These functions are meant to ensure that normal values can interact with members of the value class properly.</p>

<p>When actually training a full scale neural network, we would like to take advantage of efficient algorithms, especially when trying to evaluate the gradient descent (each individual gradient/derivative value for all of the parts of a complex mathematical expression), we can use tools such as the topological sort, which will allow for us to efficiently order the nodes that make up the expression. A topological sort essentially orders two nodes that are connected mathematically in such a way that a node "a" will always come before a node "b". The loss function will determine how well the network is predicting  Feel free to view the wikipedia article on topological sort to learn about it in more detail. Ultimately, training a neural application will requires dozens of low level optimizations for effciency, but we have covered the basics of how they fundamentally work in this lecture. As a review, neural networks are essentially circuits that process/map inputs to given outputs, with the biggest difference being that they learn and try to match their computed output to expected outputs. In the lecture, we created Micrograd with Python. Micrograd is a simple library for computing the gradients associated with the values in a neural network, and is very helpful when designing a new neural network from scratch. </p>



