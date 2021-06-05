# LSTM for muscle activation prediction
Work at Institute for Parallel and Distributed Systems (IPVS) at the University of Stuttgart.

# Motivation
The goal is to predict the activation of the 5 arm muscles of an opposed person given the normalized angle, angular velocity, angular acceleration of the arm, and the weight which will be held. Furthermore, this program facilitates the evaluation. The error measures will be evaluated to decide how the project will be continued. If certain intervals constantly have a high error measure, the original idea was to train another neural network for this interval.

# Structure
The main file is "networktest.py". "functions.py" contains several functions which are imported in "networktest.py". The file "test repeated_use_of_sorted_function.py" demonstrates how the sorting function used in "networktest.py" works in a simple example.
"networktest.py" basically consists of the following parts:
1. Preprocessing and loading of the future input data of the neural network
2. Building and training of the LSTM or dense neural network
3. Visualization and evaluation of the result

# Explanations
1. Preprocessing and loading of the future input data of the neural network

Firstly, one can use a preprogrammed function that loads the input and already separates it into a training and test set. 
Subsequently, the input can be sorted with the sort_data() function taking an array with 4 integers between 0 and 3 as input. The values 1, 2, and 3 can not exist more than once. These values indicate the priority for sorting the data. The position of the value in the array indicates to which input the priority refers. A value of 0 means that no sorting takes place for the input at the same position as the value 0. For example, sort_data([2, 3, 0, 1]) indicates that the input will be sorted with the highest priority according to the 4th input value namely the weight. The weights which have similar values will then be sorted by the angle and finally by the angular velocity.
One also has the option to extract samples every n timesteps for the training and validation data. This can shorten the computation time.

2. Building and training of the LSTM or dense neural network

50 network architectures are proposed for each number of layers between 3 and 8. The proposed network architectures specify the number of neurons per layer.
One can also manually choose the number of neurons by using an array. Multiple neural networks can be tested in one run by extending the array into the 2nd dimension.
Then, a dense neural network or/and an LSTM network can be trained.

3. Evaluation of the neural networks and of the result

The model will be evaluated with eval_models which is located in "functions.py". The command model.evaluate() returns an array having the structure ['loss', 'mean_squared_error', 'mean_absolute_error']. So can focus on analyzing different error measures. The error measure will be saved with additional information concerning the computation time and without in a *.txt file. Furthermore, the network architectures will be saved as *.png files. Subsequently, the mean absolute errors (mae) of the sorted validation data will be plotted for different angle intervals. Additionally, the mae is plotted for each muscle and angle range individually.

![image](https://github.com/NRiess/Prediction-of-Muscle-Activation/blob/main/Structure.png)

# Outlook
In the long run, the neural network will be implemented in a Hololens application to visualize the muscle activation of an opposed person. It can then be used, for example, for the education of psychotherapists. Furthermore, the neural network will also be implemented on the iPhone 12 pro since it has a lidar sensor and has therefore a good depth perception. The implementation on the iPhone would make the program available to more people since it is much cheaper than the Hololens 2.
