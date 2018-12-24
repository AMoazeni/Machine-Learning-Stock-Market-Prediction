# Introduction

<br></br>
Take me to the [code and Jupyter Notebook](https://github.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/blob/master/Jupyter%20Notebook/ML%20-%20Stock%20Market%20Prediction.ipynb) for Stock Market Prediction!

<br></br>
This article explores a Machine Learning algorithm called Recurrent Neural Network (RNN), it's a common Deep Learning technique used for continuous data pattern recognition. Recurrent Neural Network take into account how data changes over time, it's typically used for time-series data (stock prices, sensor readings, etc). Recurrent Neural Network can also be used for video analysis.


<br></br>
You are provided with a dataset consisting of stock prices for Google Inc, used to train a model and predict future stock prices as shown below.


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/01%20-%20Google%20Stock%20Price%20Prediction.png" alt="Google-Stock"></div>


<br></br>
For improved predictions, you can train this model on stock price data for more companies in the same sector, region, subsidiaries, etc. Sentiment analysis of the web, news, and social media may also be useful in your predictions. The open-source developer Sentdex has created a really useful tool for [S&P 500 Sentiment Analysis](http://sentdex.com/financial-analysis/).


<br></br>
<br></br>

# Recurrent Neural Networks

<br></br>
As we try to model Machine Learning to behave like brains, weights represent long-term memory in the Temporal Lobe. Recognition of patterns and images is done by the Occipital Lobe which works similar to Convolution Neural Networks. Recurrent Neural Networks are like short-term memory which remembers recent memory and can create context similar to the Frontal Lobe. The Parietal Lobe is responsible for spacial recognition like Botlzman Machines. Recurrent Neural Networks connect neurons to themselves through time, creating a feedback loop that preserves short-term and long-term memory awareness.

<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/02%20-%20Brain%20Diagram.png" width="400" alt="Brain"></div>


<br></br>
The following diagram represents the old-school way to describe RNNs, which shows a Feedback Loop (temporal loop) structure that connects hidden layers to themselves and the output layer which gives them a short-term memory.


### Compact Form Representation

<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/03%20-%20Old%20RNN%20Representation.png" alt="RNN"></div>



### Expanded Form Representation

<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/04%20-%20Expanded%20%20RNN%20Representation.png" width="400" alt="RNN-2"></div>


<br></br>
A more modern representation shows the following RNN types and use examples: 

1. One-To-Many: Computer description of an image. CNN used to classify images and then RNN used to make sense of images and generate context.

2. Many-To-One: Sentiment Analysis of text (gague the positivity or negativity of text)

3. Many-to-Many: Google translate of language who's vocabulary changes based on the gender of the subject. Also subtitling of a movie.


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/05%20-%20RNN%20Examples.png" width="600" alt="RNN-Example"></div>


<br></br>
Check out Andrej Karpathy's Blog (Director of AI at Tesla) on [Github](http://karpathy.github.io/) and [Medium](https://medium.com/@karpathy/).


<br></br>
Here is the movie script writen by an AI trained with an LSTM Recurrent Neural Network: [Sunspring by Benjamin the Artificial Intelligence](https://www.youtube.com/watch?v=LY7x2Ihqjmc).



<br></br>
<br></br>

# RNN Gradient Problem (Expanding or Vanishing)

<br></br>
The gradient is used to update the weights in an RNN by looking back a certain number of user defined steps. The lower the gradient, the harder it is to update the weights (vanishing gradient) of nodes further back in time. Especially because previous layers are used as inputs for future layers. This means old neurons are training much slower that more current neurons. It's like a domino effect.

<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/06%20-%20RNN%20Vanish%20Gradient.png" alt="RNN-Vanishing"></div>



<br></br>
<br></br>

# Expanding Gradient Solutions


### 1. Truncated Back-propagation
Stop back-propagation after a certain point (not an optimal because not updating all the weights). Better than doing nothing which can produce an irrelevant network.


### 2. Penalties
The gradient can be penalized and artificially reduced.


### 3. Gradient Clipping
A maximum limit for the gradient which stops it from rising more.



<br></br>
<br></br>

# Vanishing Gradient Solutions

### 1. Weight Initialization
You can be smart about how you initialize weights to minimize the vanishing gradient problem.


### 2. Echo State Network
Designed to solve vanishing gradient problem. It's a recurrent neural network with a sparsely connected hidden layer (with typically 1% connectivity). The connectivity and weights of hidden neurons are fixed and randomly assigned.


### 3. Long Short-Term Memory Networks (LSTM)
Most popular RNN structure to tackle this problem.



<br></br>
<br></br>

# LSTM

<br></br>
When the weight of an RNN gradient 'W_rec' is less than 1 we get Vanishing Gradient, when 'W_rec' is more than 1 we get Expanding Gradient, thus we can set 'W_rec = 1'.


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/07%20-%20LSTM.png" alt="LSTM"></div>


<br></br>
- Circles represent Layers (Vectors).

- 'C' represents Memory Cells Layers.

- 'h' represents Output Layers (Hidden States).

- 'X' represents Input Layers.

- Lines represent values being transferred.

- Concatenated lines represent pipelines running in parallel.

- Forks are when Data is copied.

- Pointwise Element-by-Element Operation (X) represents valves (from left-to-right: Forget Valve, Memory Valve, Output Valve).

- Valves can be open, closed or partially open as decided by an Activation Function.

- Pointwise Element-by-Element Operation (+) represent a Tee pipe joint, allowing stuff through if the corresponding valve is activated.

- Pointwise Element-by-Element Operation (Tanh) Tangent function that outputs (values between -1 to 1).

- Sigma Layer Operation Sigmoid Activation Function (values from 0 to 1).


<br></br>
<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/08%20-%20LSTM%20Cell.png" alt="LSTM-Cell"></div>



<br></br>
<br></br>

# LSTM Step 1

New Value 'X_t' and value from previous node 'h_t-1' decide if the forget valve should be opened or closed (Sigmoid).

<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/09%20-%20LSTM%20Step%201.png" width="600" alt="LSTM-Step-1"></div>



<br></br>
<br></br>

# LSTM Step 2

New Value 'X_t' and value from Previous Node 'h_t-1'. Together they decide if the memory valve should be opened or closed (Sigmoid). To what extent to let values through (Tanh from -1 to 1).

<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/10%20-%20LSTM%20Step%202.png" width="600" alt="LSTM-Step-2"></div>



<br></br>
<br></br>

# LSTM Step 3

Decide the extent to which a memory cell 'C_t' should be updated from the previous memory cell 'C_t-1'. Forget and memory valves used to decide this. You can update memory completely, not at all or only partially.

<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/11%20-%20LSTM%20Step%203.png" width="600" alt="LSTM-Step-3"></div>



<br></br>
<br></br>

# LSTM Step 4

New value 'X_t' and value from previous node 'h_t-1' decides which part of the memory pipeline, and to what extent they will be used as an Output 'h_t'.

<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/12%20-%20LSTM%20Step%204.png" width="600" alt="LSTM-Step-4"></div>



<br></br>
<br></br>

# LSTM Variation 1 (Add Peep holes)

Sigmoid layer activation functions now have additional information about the current state of the Memory Cell. So valve decisions are made, taking into account memory cell state.

<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/13%20-%20LSTM%20Var%201.png" width="600" alt="LSTM-Var-1"></div>



<br></br>
<br></br>

# LSTM Variation 2 (Connect Forget & Memory Valves)

Forget and memory valves can make a combined decision. They're connected with a '-1' multiplier so one opens when the other closes.

<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/14%20-%20LSTM%20Var%202.png" width="600" alt="LSTM-Var-2"></div>



<br></br>
<br></br>

# LSTM Variation 3 (GRU: Gated Recurring Units)

The memory pipeline is replaced by the hidden pipeline. Simpler but less flexible in terms of how many things are being monitored and controlled.

<div align="center"><img src="https://raw.githubusercontent.com/AMoazeni/Machine-Learning-Stock-Market-Prediction/master/Jupyter%20Notebook/Images/15%20-%20LSTM%20Var%203.png" width="600" alt="LSTM-Var-3"></div>




# Code

<br></br>
Download the code and run it with 'Jupyter Notebook' or copy the code into the 'Spyder' IDE found in the [Anaconda Distribution](https://www.anaconda.com/download/). 'Spyder' is similar to MATLAB, it allows you to step through the code and examine the 'Variable Explorer' to see exactly how the data is parsed and analyzed. Jupyter Notebook also offers a [Jupyter Variable Explorer Extension](http://volderette.de/jupyter-notebook-variable-explorer/) which is quite useful for keeping track of variables.


<br></br>
```shell
$ git clone https://github.com/AMoazeni/Machine-Learning-Stock-Market-Prediction.git
$ cd Machine-Learning-Stock-Market-Prediction
```

<br></br>
<br></br>
<br></br>
<br></br>
