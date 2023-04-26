import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the CSV dataset using pandas
dataset = pd.read_csv('path/to/dataset.csv')

# Split the dataset into training and testing data
train_data, test_data = train_test_split(dataset, test_size=0.2)

# Split the training and testing data into input features and labels
x_train = train_data.drop(['label'], axis=1).to_numpy()
y_train = train_data['label'].to_numpy()
x_test = test_data.drop(['label'], axis=1).to_numpy()
y_test = test_data['label'].to_numpy()

# Preprocess the data by scaling it to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the deep belief network model architecture
num_visible = 784
num_hidden1 = 500
num_hidden2 = 300
num_hidden3 = 100
num_classes = 10

model = Sequential()
model.add(Dense(num_hidden1, input_shape=(num_visible,), activation='relu'))
model.add(Dense(num_hidden2, activation='relu'))
model.add(Dense(num_hidden3, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with appropriate loss, optimizer, and metric
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the deep belief network model
num_epochs = 10
batch_size = 32
history = model.fit(x_train, keras.utils.to_categorical(y_train), batch_size=batch_size, epochs=num_epochs, validation_split=0.1)

# Evaluate the model on the testing data
score = model.evaluate(x_test, keras.utils.to_categorical(y_test), verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])