import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
​
# Sample data for training the neural network
data = [
    ("PULLOUT(WAGON, ROOM)", "Battery removed from room"),
    ("PULLOUT(WAGON, ROOM)", "Bomb exploded"),
    # Add more examples as needed
]
​
# Create a mapping of implications to outcomes
implications = {
    "PULLOUT(WAGON, ROOM)": {
        "relevant": ["Battery removed from room"],
        "irrelevant": ["Bomb exploded", "Color of room's walls unchanged", "Wheels turning more revolutions"],
    },
    # Add more implications as needed
}
​
# Prepare training data
X_train = [input_str for input_str, _ in data]
y_train = [outcome_str for _, outcome_str in data]
​
# Convert input strings to numerical data using Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_numeric = tokenizer.texts_to_sequences(X_train)
X_train_numeric = pad_sequences(X_train_numeric)
​
# Convert outcome strings to numerical data using one-hot encoding
y_train_categorical = np.zeros((len(y_train), len(implications)))
​
for i, outcome_str in enumerate(y_train):
    for j, category in enumerate(implications.values()):
        if outcome_str in category["relevant"]:
            y_train_categorical[i, j] = 1
​
# Build the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_numeric.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification
​
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
​
# Train the model
model.fit(X_train_numeric, y_train_categorical[:, 0], epochs=15, batch_size=1, verbose=1)
​
# Use the trained model to make predictions
input_str = "PULLOUT(WAGON, ROOM)"
input_numeric = tokenizer.texts_to_sequences([input_str])
input_numeric = pad_sequences(input_numeric, maxlen=X_train_numeric.shape[1])
prediction = model.predict(input_numeric)
​
# Convert the prediction to a binary outcome
predicted_outcome = "relevant" if prediction > 0.5 else "irrelevant"
​
print(f"The predicted outcome for action '{input_str}' is: {predicted_outcome}")