#import torch
#import torch.nn as nn
import numpy as np
import pandas
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle

with open('lstm_data', 'rb') as f:
	data = np.array(pickle.load(f))

split = int(.8*len(data))
np.random.shuffle(data)

train_set = data[:split]
train_features = train_set[:, 0]
train_labels = train_set[:, 1]
test_set = data[split:]
test_features = test_set[:, 0]
test_labels = test_set[:, 1]

for feature in train_features:
	print(feature.shape)
	exit()


class LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for i in range(epochs):
    for features, label in train_set:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(features)

        single_loss = loss_function(y_pred, label)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')