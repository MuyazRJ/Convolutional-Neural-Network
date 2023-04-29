import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from cnn import MNISTCNN
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def processData(batch_size):
    train_data = pd.read_csv('mnist_train.csv').sample(frac = 1).reset_index(drop = True)
    test_data = pd.read_csv('mnist_test.csv')

    train_data_labels = torch.tensor(train_data.iloc[:, 0].values).reshape(-1, 1).to(device)
    train_data = torch.tensor(train_data.iloc[:, 1:].values).reshape(-1, 1, 28, 28).to(device) / 255

    test_data_labels = torch.tensor(test_data.iloc[:, 0].values).reshape(-1, 1).to(device)
    test_data = torch.tensor(test_data.iloc[:, 1:].values).reshape(-1, 1, 28, 28).to(device) / 255

    train_data = DataLoader(dataset = TensorDataset(train_data, train_data_labels), batch_size = batch_size, shuffle = False)
    test_data = DataLoader(dataset = TensorDataset(test_data, test_data_labels), batch_size = batch_size, shuffle = False)

    return train_data, test_data

def drawImage(images):
    for image in images: 
        number = np.squeeze(image[0]).reshape(28, 28, 1)
        plt.imshow(number)
        plt.title(f"Predicted {image[1].item()} Actual: {image[2].item()}")
        plt.show()

def loadModel(filename, model, optimiser):
    checkpoint = torch.load(f'{filename}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    return model, optimiser

model = MNISTCNN(channels = 1, kernel_size = [9, 9], cnn_layers = [64, 128], ann_layers = [500, 100], output_size = 10)
train_data, test_data = processData(batch_size = 10000)

optimiser = optim.Adam(model.parameters(), lr = 0.05)
criterion = nn.CrossEntropyLoss()

model.trainNetwork(dataloader = train_data, optimiser = optimiser, criterion = criterion, epochs = 50)
print(f"Accuracy on unseen data: {model.getAccuracy(test_data)}")

images = model.wrongPredictions(test_data, 10)
drawImage(images)

model.saveModel(filename = 'model_checkpoint', optimiser = optimiser)