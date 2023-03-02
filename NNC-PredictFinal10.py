import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
plt.style.use("bmh")
import seaborn as sns




# Load data from CSV and check read
data = pd.read_csv('high_diamond_ranked_10min.csv')

# Split data into input (X) and output (Y) variables
X = data.drop(['blueWins', 'gameId'], axis=1).values
Y = data['blueWins'].values

# Apply PCA to reduce the number of features
pca = PCA(n_components=38)
X = pca.fit_transform(X)

# Scale the input variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data into stes
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=41)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.66, random_state=41)

# Classify array 1-dimension only
y_train = Y_train.ravel()
y_val = Y_val.ravel()

# Reshape the Y arrays 
Y_train = Y_train.reshape(-1, 1)
Y_val = Y_val.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# Shape of the used data
print(data.describe())

print(data.shape)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# Correlation Matrix WIP
#sns.set(rc={'figure.figsize':(15,15)})
#corr = data.corr()
#sns.heatmap(corr[((corr >= 0.3) | (corr <= -0.3)) & (corr != 1)], cannot=False, linewidths=.5, fmt= '.2f')
#plt.title('Corelation Matrix')

# Define the neural network model
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=800)

# Train the model
model.fit(X_train, y_train.ravel())

# Evaluate the model 
accuracy = model.score(X_test, Y_test)
print('Accuracy:', accuracy)
accuracy = model.score(X_val, y_val.ravel())
print('Validation accuracy:', accuracy)

# Choose a random game ID from the dataset
random_game_id = random.choice(data['gameId'])

# Retrieve the corresponding row from the dataset
new_data = data.loc[data['gameId'] == random_game_id].drop(['blueWins', 'gameId'], axis=1).values

# Apply PCA and scaling to the new data
new_data = pca.transform(new_data)
new_data = scaler.transform(new_data)

# Predict the winner for the new data using the trained NN
prediction = model.predict(new_data)

# Print the prediction and the game ID of a random game
if prediction == 1:
    print("Team Blue wins")
else:
    print("Team Red wins")
print("Game ID:", random_game_id)