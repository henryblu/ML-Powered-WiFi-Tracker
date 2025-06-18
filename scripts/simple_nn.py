import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CSV_FILE = '../data/Table.csv'

target_mac = "5A:49:80:66:41:BB" # replace with your target device mac
df = pd.read_csv(CSV_FILE)

df = df.drop(columns='CSI_DATA')
df = df.rename(columns={'seq_ctrl': 'CSI'})

# Take relevant data
df = df[['mac', 'rssi', 'CSI']] # Relevant columns
df = df[df['mac'] == target_mac] # Take only target mac address into account

data = df[['CSI']].to_numpy()

n_data = len(data)
CSI_dim = 128
print("n_data:", n_data)

def get_CSI(CSI_RAW):
    string = CSI_RAW
    res = string.split(' ')
    numbers = []
    for token in res:
        if '[' in token or ']' in token:
            if '[' in token:
                temp = token.split('[')
                if len(temp[1]) > 0:
                    number = int(temp[1])
                    numbers.append(number)
            if ']' in token:
                temp = token.split(']')
                if len(temp[1]) > 0:
                    number = int(temp[1])
                    numbers.append(number)
        else:
            number = int(token)
            numbers.append(number)
    return numbers

X = np.zeros((n_data, CSI_dim))
for i, value in enumerate(data):
    string = value[0]
    asd = np.array(get_CSI(string))
    X[i,:] = np.array(get_CSI(string))

Y = np.zeros((n_data))

# shuffle inputs
indsh = (np.arange(n_data))
np.random.shuffle(indsh)

X = X[indsh, :]
Y = Y[indsh]

X_train = X[:int(0.7 * n_data),:]
y_train = Y[:int(0.7 * n_data)]

X_val = X[int(0.7 * n_data):int(0.8 * n_data),:]
y_val = Y[int(0.7 * n_data):int(0.8 * n_data)]

X_test = X[int(0.8 * n_data):,:]
y_test = Y[int(0.8 * n_data):]

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        x = self.feature[idx, :]
        y = self.label[idx, :]
        return x, y

# Create Dataset objects
train_data = CustomDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_data = CustomDataset(torch.Tensor(X_val), torch.Tensor(y_val))
test_data = CustomDataset(torch.Tensor(X_test), torch.Tensor(y_test))

# Create DataLoader for training, validation, and test sets
batch_size = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
############################-----------------###########################