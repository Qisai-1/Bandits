import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pandas as pd
import numpy as np
import itertools

from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt

# load subset of data used to train bilinear model
subset_df = pd.read_csv("subset_data.csv") 
subset_df = subset_df.iloc[: , 1:]

# one hot encoding the Pedigree 
Pedigree = subset_df.iloc[: , 1]
Pedigree = Pedigree.values
Pedigree = Pedigree.reshape(-1,1)
encoder = OneHotEncoder()
one_hot = encoder.fit_transform(Pedigree)
seed_one_hot = one_hot.toarray()
subset_df["Seed"] = list(seed_one_hot)

# extract field names, seed names and field x seed name combinations
field_id = subset_df['Field-Location_x'].unique().tolist()
seed_id = subset_df['Pedigree_x'].unique().tolist()
seedxfield = list(itertools.product(seed_id, field_id))
print('Example of seed and field combination:', seedxfield[0])

# build dataframe with number of samples regarding to the field and seed 
unique_seed_field = subset_df.groupby(['Field-Location_x', 'Pedigree_x']).size().reset_index().rename(columns={0:"count"})

# generating list of valid seed x field combinations
valid_seedxfield = unique_seed_field[["Field-Location_x","Pedigree_x"]].values

digits = load_digits()
data = digits.data
labels = digits.target

obs_raw = subset_df[['Loc-ID-Onehot','Seed','% Sand', '% Silt', '% Clay', 'v_daily_avg_temp', 'v_daily_avg_radiation', 'v_daily_avg_photo_p']]
obs_raw = obs_raw.to_numpy()

for i in range(len(obs_raw)):
    obs_raw[i][0] = obs_raw[i][0].strip('][').split(', ')

obs = []
for i in range(len(obs_raw)):
  obs.append(list([int(id) for id in obs_raw[i][0]])  + list(obs_raw[i][1])  + list(obs_raw[i][2:]))
obs = np.asarray(obs)

scaler = StandardScaler()
obs[:,-6:] = scaler.fit_transform(obs[:,-6:])

labels = subset_df['Plot Yield (ibs/ft2)'].values

# Set device to use GPU if available, else use CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Load your input data as a numpy array
input_data = obs
number_local = len(obs_raw[0][0])
number_seed_types = len(obs_raw[0][1])
latent_dim = 8
# Define the autoencoder architecture

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_data.shape[1], out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=8, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=input_data.shape[1]),
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        decoded[:,:number_local]  = torch.softmax(decoded[:,:number_local], dim=1)
        #temp  = torch.argmax(decoded[:,:number_local] , dim=1)
        #decoded[:,:number_local] = torch.eye(number_local)[temp].to(decoded[:,:number_local].device)
        

        decoded[:,number_local:number_seed_types+number_local] = torch.softmax(decoded[:,number_local:number_seed_types+number_local], dim=1)
        #temp  = torch.argmax(decoded[:,number_local:number_seed_types+number_local] , dim=1)
        #decoded[:,number_local:number_seed_types+number_local] = torch.eye(number_seed_types)[temp].to(decoded[:,number_local:number_seed_types+number_local].device)

        return decoded



# Initialize the model, loss function, and optimizer
autoencoder = Autoencoder().to(device)
loss_MSE = nn.MSELoss()
m = nn.Sigmoid()
loss_BCE = nn.BCELoss()


optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert numpy array to PyTorch tensor
input_data = torch.tensor(input_data, dtype=torch.float32).to(device)

# Split the dataset into training and validation sets
X_train, X_val = train_test_split(input_data, test_size=0.2, random_state=42)




# Train the autoencoder
best_val_loss = float('inf')
early_stop_counter = 0
patience = 100
loss_curve_train = []
num_epochs = 2000
runs = 0 
weight_MSE = 0.9
loss_curve_val = [] 
for epoch in range(num_epochs):
    # Forward pass
    output = autoencoder(X_train)

    #Two loss function 
    loss_m = loss_MSE(output[:,-6:], X_train[:,-6:])
    loss_b = loss_BCE(m(output[:,:-6]), X_train[:,:-6])
    print(loss_b)

    loss = weight_MSE * loss_m + (1-weight_MSE) *loss_b

    loss_curve_train.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute the validation loss
    with torch.no_grad():
        val_outputs = autoencoder(X_val)

        loss_val_m = loss_MSE(val_outputs[:,-6:], X_val[:,-6:])
        loss_val_b = loss_BCE(m(val_outputs[:,:-6]), X_val[:,:-6])
        val_loss = weight_MSE * loss_val_m + (1-weight_MSE) *loss_val_b
        loss_curve_val.append(val_loss.item())
        print('Validate Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'
            .format(epoch+1, num_epochs, loss.item(), val_loss.item()))

    # Print the loss
    print('Training Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))



    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
    if early_stop_counter >= patience:
        print('Early stopping after {} epochs'.format(epoch+1))
        break

    # if runs > 5 :
    #         if loss - loss_curve[-2] < 10e-10000:
    #             break 
    # runs +=1 

# Generate encoded data
encoded_data = autoencoder.encoder(input_data).detach().cpu().numpy()
encoded_data = torch.tensor(encoded_data, dtype=torch.float32).to(device)
decoder_data = autoencoder.decoder(encoded_data).detach().cpu().numpy()


# Plot the loss curve
plt.plot(loss_curve_val, label='Val loss')
plt.plot(loss_curve_train, label='Train loss')

plt.title('Autoencoder Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()