import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load data from CSV
data = pd.read_csv('high_diamond_ranked_10min.csv')

# Split data into input (X) and output (Y) variables
X = data.drop(['blueWins', 'gameId'], axis=1).values
Y = data['blueWins'].values

# Scale the input variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Define the neural network model
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)

# Train the model
model.fit(X_train, Y_train)

# Evaluate the model on the testing data
accuracy = model.score(X_test, Y_test)
print('Accuracy:', accuracy)
print('test')

# Predict the winner for new data
new_data = scaler.transform([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
prediction = model.predict(new_data)

if prediction == 1:
    print("Team Blue wins")
else:
    print("Team Red wins")

print('test2')