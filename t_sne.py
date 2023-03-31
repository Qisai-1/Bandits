import time
import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
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

labels = subset_df['Plot Yield (ibs/ft2)'].values
# obs = []
# for i in range(len(obs_raw)):
#   loc = np.asarray(obs_raw[i][0])
#   seed = np.asarray(obs_raw[i][1])
#   result = np.concatenate((loc, seed,obs_raw[i][2:]))
#   obs.append(result)

# obs = np.asarray(obs)



pca_30 = PCA(n_components=30)
pca_result_30 = pca_30.fit_transform(obs)

print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_30.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=3, random_state=42)
tsne_pca_data = tsne.fit_transform(pca_result_30)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# create a 3D plot of the transformed data

df_sne = pd.DataFrame()
df_sne['tsne-pca50-one'] = tsne_pca_data[:,0]
df_sne['tsne-pca50-two'] = tsne_pca_data[:,1]
df_sne['tsne-pca50-three'] = tsne_pca_data[:,2]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_pca_data[:,0], tsne_pca_data[:,1], tsne_pca_data[:,2], c=labels)

# show the plot
plt.show()


tsne = TSNE(n_components=2, random_state=42)
tsne_pca_data_2D = tsne.fit_transform(pca_result_30)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(tsne_pca_data_2D, labels, test_size=0.4,random_state=1)
# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X_train, y_train)
print('Coefficients: ', model.coef_)


fig, ax = plt.subplots()
plt.style.use('fivethirtyeight')
  
## plotting residual errors in training data
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  
## plotting residual errors in test data
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
  
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
  
## plotting legend
plt.legend(loc = 'upper right')
  
## plot title
plt.title("Residual errors")
plt.xlim(0.1, 0.3)

## method call for showing the plot
plt.show()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(tsne_pca_data_2D[:,0], tsne_pca_data_2D[:,1], linear_result.predict(tsne_pca_data_2D), alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plt.show()

