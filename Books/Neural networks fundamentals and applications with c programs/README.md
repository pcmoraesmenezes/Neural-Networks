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

### Network Structure

The network structure of a neural network varies considerably, and the way the structure is arranged relates to the learning algorithm that will be used to train the network.

When the network has all outputs connected to all inputs, we have a full connected network. If we have a output sign connected to a input sign, we have a feedforward network.

The structure of a network is free, and the number of layers and neurons can be adjusted. The number of layers and neurons are defined by the problem that the network will solve.

### Learning Process

The learning process of a neural network can be supervised or unsupervised. In supervised learning, the network receives a set of labeled data, and the network adjusts the weights to minimize the error. In unsupervised learning, the network receives a set of unlabeled data, and the network adjusts the weights to find patterns in the data. The unsupervised learning splits the data in clusters.

### Learning Algorithms

The learning is made by a iterative process that adjusts the weights to minimize the error. The most common learning algorithms are:

- Learning by correction: The network adjusts the weights to minimize the error.

- Competitive learning: Kohonen's algorithm.

- Learning bases in memory

- Hebbian learning: The network adjusts the weights to maximize the correlation between the input and output.

- Boltzmann learning:

### Credit attribution problem

Credit attribution problem refers to how the network can determines the contribution of each unity(neuron, synapse, layer, etc), of the network to the final output. This attribution is guided by the global error gradient.

When we are dealing with a single layer network, the weight can be made directly. However, dealing with a multi layer network, the weight adjust must be more complex.

### Exercises

1. What are the differences between a biological neuron and a mathematical neuron?

2. What are the differences between supervised and unsupervised learning?

3. What is feedback?

4. What is the principle of neurons functions based on Hebb's law?

5. What are the principles algorithms of learning?

6. What is, and for what is used a bias?

### Answers

1. A biological neuron is a cell that receives, processes, and transmits information through electrical and chemical signals. A mathematical neuron is a computational model that receives input signals, processes them, and produces an output signal. A biological neuron has dendrites, a neuron body, and an axon. A mathematical neuron has input signals, weights, a combination function, a transfer function, and an output signal.

2. Supervised learning is a type of machine learning that involves training a model on a labeled dataset. The model learns to map input data to output data. Unsupervised learning is a type of machine learning that involves training a model on an unlabeled dataset. The model learns to find patterns in the data.

3. Feedback is a process in which the output of a system is fed back as input to the system. Feedback can be positive or negative. Positive feedback amplifies the output of a system. Negative feedback stabilizes the output of a system.

4. The principle of neurons functions based on Hebb's law is that neurons that fire together wire together. This means that when two neurons are activated at the same time, the connection between them is strengthened. However the two neurons must be in the same layer. If the neurons are in different layers, the connection between them is weakened.

5. The principle algorithms of learning are learning by correction, competitive learning, learning based on memory, Hebbian learning, and Boltzmann learning.

6. A bias is a constant value that is added to the weighted sum of the input signals in a neuron. The bias allows the neuron to adjust the weighted sum to the left or to the right. Without the bias, the weighted sum could be zero, and the network would not be able to solve the XOR problem.

## Chapter 3 - Neural Network Project

As we saw the structure of a neural network is free, and the number of layers and neurons can be adjusted. The number of layers and neurons are defined by the problem that the network will solve. This is made by a projectist, which is responsible to define the network structure. To the projectist define the network structure, it must follow the steps below:

1. Collect and data selection.

2. Define the network structure.

3. Training.

4. Test the network.

5. Integrate the network with the application.

### Collect and data selection

The first step is to collect the data that will be used to train the network. The data must be representative of the problem that the network will solve. 

The data must be split in two sets: training set and test set. The training set is used to train the network, and the test set is used to test the network.

Is recommended a random split of the data to avoid bias.

The number of synapses of a ANN's is the function of the number of inputs.

Also the recommended data value to train a network is a function of synapses and bias. More (synapses and bias) it have, more data is required to train the network.

### Define the network structure

Can be explained in three steps:

1. Which configuration of the network will be used? Example of configurations are: single layer, multi layer, feedforward, recurrent, etc. This step is a neural choice paradigm

2. How many layers and neurons will be used?

3. Learning algorithm, learning ratio, and error function.

4. Transfer function

### Training

This step consist in adjust the weights to minimize the error. The projectist choices:

1. Starting values of the weights

2. Which learning algorithm will be used

3. Time of training

Usually the weights are initialized with random values. The learning algorithm is responsible to adjust the weights to minimize the error. The time of training is the number of iterations that the network will be trained. The goal is achieve minimum error.

### Test the network

As explained before, the data is split into two sets. In this step we pick the test data, which is used to test and evalute the network. This data are not used to train the network.

In this step we must check the weights also and bias. Low values indicates prunning and high values indicates overfitting.

### Integrate the network with the application

The last step is to integrate the network with the application. The application must be able to receive the input data, send it to the network, and receive the output data. The application must also be able to display the output data to the user. 

The system must be checked periodically and be maintained.

### Exercises

1. Describe the step by step of a neural network project.

2. What can generate computational costs in a neural network project?

3. How many and what are the configuration steps of a neural network?

4. In which phase of the project the data is checked to verify overfitting?

5. How integrate the network with the application?

6. How is choosen the learning algorithm?

### Answers

1. The step by step of a neural network project is as follows: collect and data selection, define the network structure, training, test the network, integrate the network with the application.

2. Computational costs in a neural network project can be generated by the number of layers and neurons in the network, the number of synapses and bias in the network, the number of data in the training set, the number of iterations in the training process...

3. The configuration steps of a neural network are: which configuration of the network will be used, how many layers and neurons will be used, which learning algorithm will be used, which learning ratio will be used, which error function will be used, which transfer function will be used.

4. The data is checked to verify overfitting in the test the network phase.

5. The network is integrated with the application by creating an interface that allows the user to input data, sending the input data to the network, receiving the output data from the network, and displaying the output data to the user.

6. The learning algorithm is chosen based on the problem that the network will solve. The learning algorithm must be able to adjust the weights to minimize the error.

## Chapter 4 - Perceptron

The Perceptron is the simplest neural network. It's a single layer network, and it's used to solve linear problems. The Perceptron is a feedforward network, and it's composed of a single layer of neurons. The Perceptron is used to solve linear problems, and it's not capable of solving non-linear problems.

### Perceptron Structure

As metioned before, the perceptron has only a single layer. It can have multiple inputs and outputs. Those inputs are connected to the output by synapses. Each synapse has it own weight and can have a bias. Also all inputs are connected to a single neuron. Calling the inputs as $x$ that can be represented as $x_{1}, x_{2}, x_{3}, ..., x_{n}$, the weights as $w$ that can be represented as $w_{1}, w_{2}, w_{3}, ..., w_{n}$, and the bias as $b$, we can call the transfer function as $v = \sum_{i=1}^{n} w_{i}x_{i} + b$ which is the seighted sum of inputs plus the bias. Next we go to a activation function, which each problem has it own function. The most common activation function is the sigmoid function. The sigmoid function is represented as $\phi(x) = \frac{1}{1 + e^{-v}}$. The output of the perceptron is the result of the activation function. The output can be represented as $y = \phi(v)$.

![Perceptron Structure](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/4.avif)

The image above show us the structure of a perceptron. The inputs goes from the range of $x_{1}, x_{2}, x_{3}, ..., x_{n}$, and the weights goes from the range of $w_{1}, w_{2}, w_{3}, ..., w_{n}$. The output is the result of the activation function.

### Perceptron Learning Algorithm 

As we saw before, after adjusting the weights, the perceptron can give a desire output with a minimal error. The proccess of adjust weights is iterative and is done by a learning algorithm called **delta rule**. The processing used by this rule is called ADALINE (Adaptive Linear Neuron). 

We initialize randomly the weights and the net sum is calculated. Based on that we compare the desire output with the real output. If the error is not aceptable, we adjust the weights. The weights are adjusted proportionally to the error and sign value.

The delta rule is represented as:

$w(i,j)_{T+1} = w(i,j)_{T} + nE(j)_{T}x(i)$. Where:

- $w(i,j)_{T+1}$ is the new weight

- $w(i,j)_{T}$ is the old weight

- $n$ is the learning ratio

- $E(j)_{T}$ is the error for the neuron $j$

- $i$ is the index for the input

- $j$ is the index for the neuron

- $T$ is iteration

- $x(i)$ is the input

The error is calculated as $E(j) = d(j) - y(j)$. Where:

- $E(j)$ is the error for the neuron $j$

- $d(j)$ is the sign calculated by the network to the neuron $j$

- $y(j)$ is the output calculated by the network to the neuron $j$

So the error is given by the difference between the desire signal to the neuron j $(d(j))$ and the output signal calculated by the network $(y(j))$.

The mean error for all neurons in output layer is calculated as $E = \frac{1}{n}\sum_{j=1}^{n} |E(j)|$. Where:

- $E$ is the mean error

- $n$ is the number of neurons in the output layer

- $E(j)$ is the error for the neuron $j$

The mean error for all training data is calculated as $E = \frac{1}{n}\sum_{i=1}^{n} |E(T)|$. Where:

This value can be used to stop the training process.

### Example 1

In this example, the perceptron will have two input sign and one output. This net after the train will be able to do a binary classification for four individuals. The individuals and the class are:

- Bach: Compositor

- Beethoven: Compositor

- Einstein: Scientist

- Kepler: Scientist

The next step is to codify the data. The data can be codified as:

| Individual | Input 1 | Input 2 |
|------------|---------|---------|
| Bach       | 0       | 0       |
| Beethoven  | 0       | 1       |
| Einstein   | 1       | 0       |
| Kepler     | 1       | 1       |

We are also going to use the bias.

After those considerations we can track the net, initial we will consider the synapses as 0 and also the bias.

![Simples perceptron](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/simple_perceptron.png)

The first entry is for KEPLER (1 1), however the output is 0, which means that the net is returning a compositor.

The bias is adjusted to 1, and the calculation for the activation function is defined like: $v = \sum_{i=1}^{n} w_{i}x_{i} + b$. The output is 1, which means that the net is returning a scientist. So:

$v = 0 * 1 + 0 * 1 + 0*1 = 0$

The transferer function has the following value: 
$$
\phi(v) = 
\begin{cases} 
1, & \text{if } v > 0 \\
0, & \text{if } v \leq 0 
\end{cases}
$$ 

$v \leq 0$, $\phi = 0$. So the net is returning a compositor.

The output isn't correct and comparing it with the desire output we have an error. The error is calculated by: $E = 1 - 0 = 1$.

Defining arbitrary a learning ratio $n = 1$ the adjust of weights and bias will be:

$w(i)_{T+1} = w(i)_{T} +nE_{T} x(i)$

The new weights will be:

$w_{1} = 0 + 1 * 1 * 1 = 1$

$w_{2} = 0 + 1 * 1 * 1 = 1$

$b = 0 + 1 * 1 = 1$

So the new neural after the first train will be:

![Perceptron after first train](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/perceptron%20after%20first%20train.png)

Now we are consider the input (0 1) for BEETHOVEN. The sum of weights is $2$, and after the transfer function is $1$. The output is $1$, which means that the net is returning a scientist. The desire output is $0$, and the error is $-1$. The new weights and bias are:

$w_{1} = 1+1*(-1)*0 = 1$

$w_{2} = 1+1*(-1)*1 = 0$

$b = 1 + 1 * (-1) * 1 = 0$

Now we are going to send (1 0) to net.

![Einsten input](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/sign_for_einsten.png)

Now adjusting the weights we discover that the error is now define $E = 1-1 = 0$

$w_{1} = 1+1*(0)*1= 1$

$w_{2} = 0 + 1*(0)*1 = 0$

There's no adjust in the net, so trying a new input (0 0) we discover that will be no adjust in the net again. So our net is trained and ready.

### Example 2

Now in this example will have two entry and two outputs. The goal is:

- Einstein: 0 1 1 0

- Machado de Assis: 1 0 1 0

- Raquel de Queiroz: 1 0 0 1

- Marie Curie: 0 1 0 1

We can codify in this way: Einstein = 11; Machado de Assis = 10; Raquel de Queiroz = 00; Marie Curie = 01; Autor = 0; Cientist = 1 ; Man = 0; Woman = 1

In this way we are going to use two elements in entry and two neurons in output. Also a bias have value $1$. Learning Ratio is defined, $n=1$

![Two outputs](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/two_outputs_perceptron.png)

We start the train of this net showing the sign (0 1) for Marie Curie. THe output will be (0 0).

$v_{1} = 0 * 1 + 0 * 0 + 0 * 1= 0$

$v_{2} = 0 * 1 + 0 * 0 + 0 * 1 = 0$

The transfer function, which is the same as the previous example, will return $0$ for all neurons. WOMEN is defined as $1$ and CIENTIST as $1$, so the error is $1$ for both neurons. The new weights and bias are:

$w_{11} = 0 +1 *1 *0 = 0$

$w_{12} = 0 +1 *1 *0 = 0$

$w_{21} = 0 +1 *1 *1 = 1$

$w_{22} = 0 +1 *1 *1 = 1$

$b_{1} = 1 + 1 * 1 * 0 = 1$

$b_{2} = 1 + 1 * 1 * 1 = 1$

After the trains the net will be:

- $w_{11} = -2$

- $w_{12} = 0$

- $w_{21} = 1$

- $w_{22} = 1$

- $b_{1} = 1$

- $b_{2} = 0$

### XOR Problem

The XOR problem is a problem that a perceptron with a single layer cannot solve. The perceptron is limited to solving linear problems, and the XOR problem is a non-linear problem. The XOR problem is a problem that requires a multi-layer perceptron to solve.

### C implementation

You can see the C implementation of a perceptron in the file [Perceptron](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/perceptron/perceptron.c)

### Exercises

1. Describe the working of a perceptron, your's advantages and limitations.

2. What is the delta rule?

3. Why xor problem cannot be solved by a perceptron?

4. Describes the steps to train a perceptron.

5. Using the C implementation build a perceptron with 6 inputs and 4 outputs.

6. Using the C implementation to build a perceptron to generate a truth table with 8 inputs for AND, OR and NAND

7. In a collection of 6 books, 4 are about engineering and 2 about literature, 3 of the engineering books and 1 of the literature books are written by mann, the other books are written by woman. Create a perceptron to classify the books.

### Answers

1. The perceptron is the most simple neural network. It's a single layer network, and it's used to solve linear problems. The perceptron is a feedforward network and it's not capable to solve non linear problems. The advantages of the perceptron are its simplicity and lower computational cost. The limitations of the perceptron are its inability to solve non-linear problems and its inability to solve the XOR problem.

2. The delta rule is a learning algorithm used to adjust the weights of a perceptron to minimize the error. The delta rule adjusts the weights proportionally to the error and the sign value.

3. The XOR problem cannot be solved by a perceptron because the perceptron is limited to solving linear problems, and the XOR problem is a non-linear problem. The XOR problem requires a multi-layer perceptron to solve.

4. The steps to train a perceptron are: initialize the weights and the bias, calculate the net sum, calculate the output, calculate the error, adjust the weights and the bias, repeat the process until the error is acceptable.

5. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/perceptron/exercise_5.c)

6. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/perceptron/exercise_6.c)

7. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/perceptron/exercise_7.c)

## Chapter 5 - Multi Layer Perceptron

Due the limitations of perceptron it was necessary to introduce layers in the network. Those layers primarily are based on human brain. This layers are responsible to solve non-linear problems.

### Description

Multi Layer Perceptron (MLP) is a net with one or more hidden layers. It has $i$ entrys and $j$ outputs. The net is feedforward. MLP is a generalization of a perceptron and the training is made by supervisioned algorithm. The learning algorithm is know as backpropagation.

![MLP](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Images/Example-multilayer-perceptron-MLP-model-of-a-multi-classification-artificial-neural.png)

The MLP sign propagation works as a perceptron.

### Backpropagation Algorithm

You can find the explanation to this algorithm [Here](https://github.com/pcmoraesmenezes/Calculo/blob/main/Explica%C3%A7%C3%A3o%20de%20Algoritmo/Backpropagation.md)

### C implementation

You can see the C implementation of a perceptron in the file [Perceptron](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/MLP/main.c)

### Exercises

1. Describe the backpropagation algorithm.

2. Describe the steps of a mlp train

3. Using the C implementation solve the XOR problem

4. Add the momentum to the C implementation

5. Consider those functions: $y = 3x + z$ and $y = 5x² - z$. Create a MLP to solve this problem.

6. Find the numerical results for the functions $y = \sqrt{x}$ and $y = x³$. Create a MLP to solve this problem.

### Answers

1. The backpropagation algorithm is a learning algorithm used to adjust the weights of a multi-layer perceptron to minimize the error. The backpropagation algorithm adjusts the weights by propagating the error backwards through the network.

2. The steps of a MLP train are: initialize the weights and the bias, calculate the net sum, calculate the output, calculate the error, adjust the weights and the bias, repeat the process until the error is acceptable.

3. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/MLP/xor.c)

4. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/MLP/momentum.c)

5. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/MLP/5.c)

6. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/MLP/6.c)

## Chapter 6 - Konohen Network

New type of learning proccess called competitive learning. Is also know as self-organizing map. This network aims to reduce the dimensionality of the data.

### Description

The Konohen network is a unsupervised learning network that is also a self-organizing map. The concept of a competitive is related to: Based on the input value the neurons compete with each other and the neuron who wons has the weight adjusted. Also, the neurons that are close to the winner neuron have the weight adjusted too. This network is also a feedforward network.

### Net Algorithm

You can find the explanation to this algorithm [Here](https://github.com/pcmoraesmenezes/Calculo/blob/main/Explica%C3%A7%C3%A3o%20de%20Algoritmo/Konohen's%20net.md)


### C implementation

You can see the C implementation of a perceptron in the file [Konohen](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Konohen/main.c)

### Exercises

1. Describe the principle of konohe's net use, and explain how it's done the test.

2. What's is a winner neuron?

3. Describe the cooperative learning and adaptive learning.

4. Make a konohen's net to verify based on three input which number class belongs.

5. Modify the konoken's net to the neuron be activity from two inputs, defined by the function y = x+1 and y = 2x

### Answers

1. The principle of Konohen's net is to reduce the dimensionality of the data. The test is done by presenting the input data to the network and observing the output data.

2. The winner neuron is the neuron that wins the competition in a Konohen network. The winner neuron has the weight adjusted.

3. Cooperative learning is a type of learning in which the neurons in a Konohen network compete with each other to win the competition. Adaptive learning is a type of learning in which the weights of the neurons in a Konohen network are adjusted based on the input data.

4. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Konohen/4.c)

5. The solution can be found [Here](/Books/Neural%20networks%20fundamentals%20and%20applications%20with%20c%20programs/Konohen/5.c)

