# MNIST Digit Classification Using CNN Neural Network in PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch for classifying MNIST digits. The CNN is trained on a dataset with 60,000 images and has been evaluated to achieve an accuracy of 96.7% on unseen data and an accuracy of 97.1% on seen data.

## Preprocessing
The MNIST dataset is preprocessed using the preprocessData() function. The function performs the following steps:

- Reads the MNIST dataset from the PyTorch built-in dataset.
- Normalises the pixel values to be between 0 and 1.
- Splits the dataset into train and test data using 80% of the data for training and 20% for testing.
- Creates DataLoader objects for loading train and test data in batches using torch.utils.data.DataLoader().

MNIST Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv

## Neural Network Architecture
The neural network architecture is defined in the MNISTCNN class. The class takes the following arguments during initialization:

- channels: Number of input channels (1 for grayscale images).
- output_classes: Number of output classes (10 for MNIST digits).
- kernel_size: Size of convolutional kernel.
- cnn_hidden_layers: List of integers specifying the sizes of the convolutional hidden layers.
- ann_hidden_layers: List of integers specifying the sizes of the feed-forward hidden hidden layers.

The neural network has the following architecture:

- Convolutional layer with 64 output channels, kernel size of 9, stride of 1, and padding of 1.
- Max pooling layer with kernel size of 2 
- Convolutional layer with 128 output channels, kernel size of 9, stride of 1, and padding of 1.
- Fully connected layer with 500 output neurons.
- Fully connected layer with 100 output neurons.
- Output layer with 10 output neurons (one for each digit).

The neural network has a convolutional layer, pooling layers, fully connected layers, and an output layer. ReLU activation function is applied to the convolutional and hidden layers, and softmax activation function is applied to the output layer.

The neural network model is trained using the trainModel() method, which takes the following arguments:

- dataloader: DataLoader for loading training data.
- epochs: Number of epochs for training.
- criterion: Loss function for optimisation.
- optimiser: Optimisation algorithm for updating model parameters (Adam Optimiser).

## Packages Used
PyTorch, NumPy, Matplotlib, Pandas 
