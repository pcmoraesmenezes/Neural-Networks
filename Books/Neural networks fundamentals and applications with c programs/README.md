# Neural Networks - Fundamentals and Applications with C Programs 📚

## Authors

Oswaldo Ludwig Jr. And Eduard Montgomery


## Chapters 📚

## Chapter 1: Introduction

Artificial Neural Networks (ANN) are a computational model that is inspired by the way biological neural networks in the human brain process information. The key element of this model is the novel structure of the information processing system. It is composed of a large number of highly interconnected processing elements (neurons) working in unison to solve specific problems. ANNs, like people, learn by example. An ANN is configured for a specific application, such as pattern recognition or data classification, through a learning process. Learning in biological systems involves adjustments to the synaptic connections that exist between the neurons. This is true of ANNs as well.

### About Artificial Intelligence

The concept of intelligence is difficult to define, this concept are described as an undefined concept. Below are a list of habilites that define an machine as intelligent:

- ability to make inferences and solve problems

- ability to plan

- ability to accumulate and apply knowledge

- ability to understand natural language

- ability to learn with and without supervision

- ability to perceive sensory information

--- 

The two last itens are most related to ANNs.

#### Knowledge Representation

The accumulated knowledge in a machine can take a declarative context or procedural context. Declarative knowledge is the knowledge that can be expressed in a declarative sentence or as a fact. Procedural knowledge is the knowledge that can be expressed as a procedure or a recipe. Declarative context can be strongly represented in Prolog.

### Reasoning   

We can express reasoning as the capability to search process in a state of possible actions and states. Actions are maintained by operators, and operators are used to change the state of the system. Each state can maintain a finite number of actions.

### Learning

- Generalization: The ability to apply knowledge to new situations.

- Discrimination: The ability to distinguish between different situations.

- Induction: The ability to infer general rules from specific examples.

- Instruction: The ability to learn from a teacher.

Basically we can define a machine learning process as the following above. Machine learning has two main types: The first is based on symbolic representation, involves representing knowledge through symbols or rules and using sequential logical inference to learn and make decisions. This method requires a search engine to integrate new knowledge into the existing base and deduce implicit facts. It is used in systems such as medical diagnosis and natural language processing, but faces challenges in encoding knowledge into rules and dealing with ambiguities. This type of learning is considered supervised learning, because it's required a set of labeled data to learn the relationships between symbols or rules and desired outcomes. Labeled data provides examples for the algorithm to learn to associate specific inputs with correct outputs. The second type of machine learning is based on connectionist representation, which involves representing knowledge through interconnected nodes and using parallel processing to learn and make decisions. This type requires an algorithm to adjust the weights of connections between nodes to minimize errors and improve performance. It is used in systems such as pattern recognition and data classification, but faces challenges in training large networks and dealing with overfitting.

### Short History of Neural Networks

Developed in 40's by Walter Pits and McCulloch. The idea behind was make an interconnection beetwen biological neurons and eletronical circuits.

In 1956, Artificial Inteligence(IA) was divided in two main areas: symbolic representation and connectionist representation. The first one was based on symbolic representation, and the second one was based on connectionist representation.

In 1957, Frank Rosenblatt developed the Perceptron. However Marvin Minsky and Seymor Papert showed that the Perceptron was limited to solve linear problems. a perceptron with a single layer can't solve problems usign XOR operator, for example.

### Benefits and Limitations of Neural Networks

Thanks to the massively parallel structure of neural networks and their excellent generalization ability, it is possible to produce suitable outputs for inputs that were not present during training. This capability distinguishes itself from many other algorithms. For complex problems, the network is not capable of learning alone, making it necessary to break the complex problem into simple problems. This requires integration with a traditional algorithm. Learning then cannot be something completely connectionist until now.

A network may be able to adjust its weights in real time, distinguishing itself from non-stationary universes. However this can be a problem, because an neural network requires a lot of computational power and a lot of data to learn. They are also called data-hungry algorithms.

One of the biggest disadvantages of neural networks is the fact that many types of them be "black box" algorithms, which means that is impossible to know how the network is making decisions. This can be a problem in some areas, like medicine, for example. Also it's impossible know the weight relation between the inputs and outputs. It's only possible to know if the network works well making analysis using square mean error, for example.

---
### Exercises

1. What does artificial intelligence mean?

2. What abilities should a smart machine have?

3. Make an review about fundamental concepts of smart systems.

4. What's learning?

5. Describe the advantages and disadvantages of neural networks.

6. What's XOR problem?

### Answers

1. Artificial intelligence is a branch of computer science that aims to create machines capable of performing tasks that require human intelligence.

2. A smart machine should be able to make inferences and solve problems, plan, accumulate and apply knowledge, understand natural language, learn with and without supervision, perceive sensory information.

3. A smart system is a system that can make decisions and take actions based on knowledge and reasoning. It can be based on symbolic representation or connectionist representation. It can learn from examples or from a teacher. It can generalize, discriminate, induce, and instruct.

4. Learning is an undefined concept that involves the acquisition of knowledge or skills through experience, study, or teaching. The most common concept suggests that the simplest of changes that occur in a system, where all the weights are changed to adjust to this new change, is considered a learning process.

5. The advantages of neural networks are their massively parallel structure, excellent generalization ability, and real-time weight adjustment. The disadvantages are their data-hungry nature, the need to break complex problems into simple problems, and their black box nature.

6. The XOR problem is a problem that a perceptron with a single layer cannot solve. The perceptron is limited to solving linear problems, and the XOR problem is a non-linear problem. The XOR problem is a problem that requires a multi-layer perceptron to solve.

## Chapter 2 - Neurocomputing Principles

Neurocomputing is based on the principles of biological neural networks. Understanding the process about how the brain works is essential to understand how artificial neural networks work. 

### Neurons

Human brain has a lot of neurons, and each neuron can create synapse with a lot of others neurons. If we consider an connection as a binary bit (0 or 1), we can say that we has a $X = x*y$ Where:

- $X$ Memory capacity

- $x$ Number of neurons

- $y$ Number of synapses

![Biological Neuron](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/1.png)

The image above show us a biological neuron. The principals components in biological neurons are:

- Dendrites: They are the input of the neuron. They receive the signals from other neurons.

- Neuron Body: It's the processing unit of the neuron. It's responsible to process the signals received by the dendrites.

- Axon: It's the output of the neuron. It's responsible to send the signals to other neurons.

Eletrical signs are sent by the axon to the dendrites. To a sign be considered valid, it must be greather than aproximately $50mV$ in the neuron body. If the sign is lower than $50mV$, the neuron doesn't send the sign to the axon.

The system is essentially considered as non linear. 

Before the eletrical sign pass to the next neuron must pass by a synapse. Here the pass is made by chemical signs, and the sign must be greather than an threshold to pass to the next neuron. Each human brain network has it own threshold.

A neuron receives signs from countless dendrites, and the neuron ponders it and sends it to the axon. It's valid to say that each passage has it own sign that can be amplified or not. Human brain has a lot of specialized functions, and each function is made by a specific network of neurons, which it's responsible to give weights to the signs.

Weights are adjusted based on train received by brain during life.

![Mathematical neuron representation](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/2.png)

The image above show us a mathematical representation of a neuron. The input can be one or more signals, and the output is the result of the ponderation of the input signals. The ponderation is made by the weights. The output is the result of the ponderation of the input signals. The output is made by a only function, called activation function. The activation function can be linear or non-linear. The most common activation function is the sigmoid function. Those output can be a input to another neuron.

Input signs came simultaneously to the neuron, and it's because that the neuron is considered a parallel processing unit.

Dendrites and axons are mathematical represented by the synapses and weights. The weight is represented by $w$, and the input signal is represented by $x$. When we have a input, $x$, and a weight, $w$, the output is $y = w*x$. The output is the result of the ponderation of the input signal.

$w_{1}x_{1},  w_{2}x_{2}, w_{3}x_{3} , ... ,w_{n}x_{n}$

The neuron, however make a sum of those ponderations, and the output is the result of the sum of the ponderations. The output is represented by $v = \sum_{i=0}^{n} w_{i}x_{i}$

This function is called as combination function or activation function.

This value $v$ is passed to an transfer function, which is responsible to avoid the progressive increase. Those functions has it own maximum and minimum value.

A perceptron with two layers with a non linear transfer function (sigmoid) can solve any problem.

We can see that in a mathematical neuron model, the neuron is statistical, so it doesn't consider the natural neuron dynamics.

Dealing with a network tha has more than one layer we adjust weights using the gradient descent algorithm. The gradient descent algorithm is responsible to adjust the weights to minimize the error and find the minimum value of the error. Thanks to the derivative process, if the transfer function had a constant value, the derivative would not find any information in the search for the best value.

Some of the most common transfer functions are:

- Sigmoid function: $\phi(x) = \frac{1}{1 + e^{-v}}$

- Gaussian function: $\phi(x) = e^{-v^{2}}$

- Hyperbolic tangent function: $\phi(x) = \frac{e^{v} - e^{-v}}{e^{v} + e^{-v}}$ or $\phi(x) = tanh(v)$

We also can use a bias in the neuron. The bias is a constant value that is added to the ponderation. The bias is represented by $b$. The output is represented by $y = w*x + b$. Thanks to the bias, we can adjust the ponderation to the left or to the right. Without the bias, the function could be a zero and the problem or $XOR$ could not be solved.

### Neural Networks

By combining the neurons with one or more layers and the synapses we can create a Artificial Neural Network (ANN). 

![Artificial Neural Networks](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/3.png)

The image above show us an ANN. We can find an input layer, which is the first layer of the network, and receives the input signals. The input layer is connected to the hidden layer, which is responsible to guarantee non linear ponderation. The hidden layer is connected to another hidden layer or to the output layer. The output layer is responsible to give the final output of the network.

By adjusting weights the network can learn (memorize) the input data patterns.