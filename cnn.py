import torch 
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNISTCNN(nn.Module):

    def __init__(self, channels, kernel_size, cnn_layers, ann_layers, output_size):
        super().__init__()  

        self.input_layer = nn.Conv2d(channels, cnn_layers[0], kernel_size = kernel_size[0]).to(device)
        self.pool = nn.MaxPool2d(kernel_size = 2).to(device)
        ann_input = (28 - kernel_size[0] + 1) / 2
        self.cnn_layers = []
        self.ann_layers = []
        
        for i in range(len(cnn_layers) - 1): 
            ann_input = int(np.floor((ann_input - kernel_size[i + 1] + 1) / 2))
            self.cnn_layers.append(nn.Conv2d(cnn_layers[i], cnn_layers[i + 1], kernel_size = kernel_size[i + 1]).to(device))

        self.final_cnn_pixels = ann_input
        ann_input = cnn_layers[-1] * ann_input ** 2
        self.ann_layers.append(nn.Linear(ann_input, ann_layers[0]).to(device))

        for i in range(len(ann_layers) - 1): self.ann_layers.append(nn.Linear(ann_layers[i], ann_layers[i + 1]).to(device))
        self.output_layer = nn.Linear(ann_layers[-1], output_size).to(device)
    
    def forward(self, input):
        x = self.pool(torch.relu(self.input_layer(input)).to(device))
        ann_inputs = self.cnn_layers[-1].out_channels * self.final_cnn_pixels ** 2 
        
        for layer in self.cnn_layers: x = self.pool(torch.relu(layer(x)).to(device))
        x = x.reshape(-1, ann_inputs)
        for layer in self.ann_layers: x = torch.relu(layer(x)).to(device)
        
        output = torch.sigmoid(self.output_layer(x))
        return output
    
    def trainNetwork(self, dataloader, optimiser, criterion, epochs):
        for epoch in range(epochs):
            training_loss = 0 

            for (X, y) in dataloader:
                self.train()
                optimiser.zero_grad()
                predictions = self.forward(X).to(device)
                loss = criterion(predictions, y.squeeze().long())
                training_loss += loss
                loss.backward()
                optimiser.step()

            if epoch % 2 == 0: 
                training_loss /= len(dataloader)
                print(f"Epoch: {epoch} Loss: {training_loss} Accuracy: {self.getAccuracy(dataloader)}")
    
    def getAccuracy(self, dataloader):
        self.eval()
        correct = 0  

        for (data, labels) in dataloader: 
            predictions = self.forward(data).to(device)  
            true_false_predictions = predictions.argmax(dim=1) == labels.squeeze().long().to(device)
            correct += (true_false_predictions).sum()
        return correct / len(dataloader.dataset)  
    
    def wrongPredictions(self, dataloader, image_amount):
        self.eval()
        images = []

        for (data, labels) in dataloader:  
            predictions = self.forward(data).to(device) 
            model_num_predictions = predictions.argmax(dim=1)
            true_false_predictions = predictions.argmax(dim=1) == labels.squeeze().long().to(device)
            false_predictions = torch.nonzero(~true_false_predictions).cpu().tolist()

            for index in false_predictions:
                images.append([data[index].cpu().numpy(), model_num_predictions[index].cpu().numpy(), labels[index].cpu().numpy()])
                if len(images) >= image_amount: return images

    def saveModel(self, filename, optimiser):
        self.eval()
        torch.save({'model_state_dict': self.state_dict(), 'optimiser_state_dict': optimiser.state_dict()}, f'{filename}.pth')