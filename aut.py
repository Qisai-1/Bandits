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


def normalize_data(data): 
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(obs[:,-6:])
    return X_normalized


obs[:,-6:] = normalize_data(obs[:,-6:])

Yield = subset_df['Plot Yield (ibs/ft2)'].values.reshape(-1,1)

Yield = normalize_data(Yield)





# Set device to use GPU if available, else use CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Load your input data as a numpy array
input_data = obs
number_local = len(obs_raw[0][0])
number_seed_types = len(obs_raw[0][1])
latent_dim = 8
# Define the autoencoder architecture




class Encoder(nn.Module):
    def __init__(self, 
                num_input_size : int,  
                latent_dim : int):
        super().__init__()

        self.fc1 = nn.Linear(num_input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
                num_input_size : int,
                latent_dim : int):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_input_size)
  
    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        x[:,:number_local] = torch.softmax(x[:,:number_local],dim=1)
        x[:,:number_local] = nn.Sigmoid()(x[:,:number_local])
        
        #temp  = torch.argmax(x[:,:number_local] , dim=1)
        #x[:,:number_local] = torch.eye(number_local)[temp].to(x[:,:number_local].device)
        
        x[:,number_local:number_seed_types+number_local] = torch.softmax(x[:,number_local:number_seed_types+number_local], dim=1)
        x[:,number_local:number_seed_types+number_local] = nn.Sigmoid()(x[:,number_local:number_seed_types+number_local])
        #temp = torch.argmax(x[:,number_local:number_seed_types+number_local] , dim=1)
        #x[:,number_local:number_seed_types+number_local] = torch.eye(number_seed_types)[temp].to(x[:,number_local:number_seed_types+number_local].device)

        return x 



class Autoencoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 8,
                 num_input_size : int = 53,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 ):
        super(Autoencoder, self).__init__()

        self.encoder = encoder_class(num_input_size,latent_dim)
        self.decoder = decoder_class(num_input_size,latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Define the architecture of the DNN for linear mapping 
class Linear_Mapping(nn.Module):
    def __init__(self,
                latent_dim : int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 16)  # Input layer
        self.fc2 = nn.Linear(16, 1)   # Output layer

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))  # ReLU activation in input layer
        x = self.fc2(x)  # Output layer
        return x


# Initialize the model, loss function, and optimizer
autoencoder = Autoencoder(latent_dim,input_data.shape[1] ).to(device)
mapping = Linear_Mapping(latent_dim).to(device)
loss_MSE = nn.MSELoss()
loss_BCE = nn.BCELoss()


optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert numpy array to PyTorch tensor
input_data = torch.tensor(input_data, dtype=torch.float32).to(device)

Yield = torch.tensor(Yield, dtype=torch.float32).to(device)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(input_data, Yield,  test_size=0.2, random_state=42)




# Train the autoencoder
best_val_loss = float('inf')
early_stop_counter = 0
patience = 20
loss_curve_train = []
num_epochs = 2000
runs = 0 
weight_MSE = 0.3
loss_curve_val = [] 
for epoch in range(num_epochs):
    # Forward pass
    output = autoencoder(X_train)
    # Linear mapping 

    encoder_output = autoencoder.encoder(X_train)
    Yield_result = mapping(encoder_output)

    #Two loss function for autoencoder 
    loss_m = loss_MSE(output[:,-6:], X_train[:,-6:])


    loss_b_local = loss_BCE(output[:,:number_local], X_train[:,:number_local])
    loss_b_seed = loss_BCE(output[:,number_local:number_seed_types+number_local], X_train[:,number_local:number_seed_types+number_local])

    loss_b = 0.5 * loss_b_local + 0.5 * loss_b_seed

    #Loss function for linear mapping
    loss_Liner_map = loss_MSE(Yield_result, y_train)


    loss = weight_MSE * loss_m + (1-weight_MSE) *loss_b + (1-weight_MSE)/2 * loss_Liner_map

    print(" MSE loss for autoencoder {}".format(loss_m))
    print(" BCE loss for autoencoder {}".format(loss_b))
    print(" MSE loss for Linear model {}".format(loss_Liner_map))

    loss_curve_train.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute the validation loss
    with torch.no_grad():
        val_outputs = autoencoder(X_val)

        encoder_val_output = autoencoder.encoder(X_val)
        Yield_val_result = mapping(encoder_val_output)

        loss_val_m = loss_MSE(val_outputs[:,-6:], X_val[:,-6:])
        loss_val_b = loss_BCE(val_outputs[:,:-6], X_val[:,:-6])


        loss_val_Liner_map = loss_MSE(Yield_val_result, y_val)


        val_loss = weight_MSE * loss_val_m + (1-weight_MSE) *loss_val_b + 0.5 * loss_val_Liner_map
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
decoder_data = autoencoder.decoder(encoded_data).detach().cpu()

temp  = torch.argmax(decoder_data[:,:number_local] , dim=1)
decoder_data[:,:number_local] = torch.eye(number_local)[temp].to(decoder_data[:,:number_local].device)

temp = torch.argmax(decoder_data[:,number_local:number_seed_types+number_local] , dim=1)
decoder_data[:,number_local:number_seed_types+number_local] = torch.eye(number_seed_types)[temp].to(decoder_data[:,number_local:number_seed_types+number_local].device)



# Predicted Yield 

Yield_pred = mapping(encoded_data)

# Plot the loss curve
plt.plot(loss_curve_val, label='Val loss')
plt.plot(loss_curve_train, label='Train loss')

plt.title('Autoencoder Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()