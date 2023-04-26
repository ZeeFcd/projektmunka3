import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the CSV dataset using pandas
dataset = pd.read_csv('training.csv')

# Split the dataset into training and testing data
train_data = dataset.sample(frac=0.8, random_state=0)
test_data = dataset.drop(train_data.index)

# Split the training and testing data into input features and labels
x_train = train_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_train = train_data['ONCOGENIC'].to_numpy()
x_test = test_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_test = test_data['ONCOGENIC'].to_numpy()

# Reshape the input features to be 4D for CNN input
x_train = x_train.reshape(x_train.shape[0], 13, 2, 1)
x_test = x_test.reshape(x_test.shape[0], 13, 2, 1)

# Preprocess the data by scaling it to [0, 1] and one-hot encoding the labels
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Define the CNN model architecture
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 1), activation='relu', input_shape=(13, 2, 1)),
        layers.MaxPooling2D((1, 1)),
        layers.Conv2D(64, (3, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(6, activation='softmax')
    ]
)

# Compile the model with appropriate loss, optimizer, and metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the model on the testing data
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])