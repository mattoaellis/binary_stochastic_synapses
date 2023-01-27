# Binary Stochastic Synapses

Code for modelling binary stochastic synapse networks.

This code is written using PyTorch, Scikit-Learn and numpy for certain functions. A conda yaml file is provided but the required Python packages are:
- matplotlib
- numpy
- python=3
- scipy
- pytorch
- torchvision
- cudatoolkit=10.2
- scikit-learn
- torchaudio

## Running the code

This code will run a training loop for the binary stochastic newtwork using a downscaled copy of the MNIST dataset. The code requres the number of training samples to be provided as an input, with the value given as a power of 2 (e.g specifying 0 will give 1 sample) . For example:
```
>python ./mnist_test.py 0
```
will run the training using a single synaptic sample.

The code will train 5 different models and save their trained weights to '{Nsy}_BSS_model_{i}.pt'. The test and validation errors are stored as '{Nsy}_{test,valid}_err{i}'.