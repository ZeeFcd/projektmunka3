import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the CSV dataset using pandas
dataset = pd.read_csv('training.csv')

# Split the dataset into training and testing data
train_data, test_data = train_test_split(dataset, test_size=0.2)

# Split the training and testing data into input features and labels
x_train = train_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_train = train_data['ONCOGENIC'].to_numpy()
x_test = test_data.drop(['ONCOGENIC'], axis=1).to_numpy()
y_test = test_data['ONCOGENIC'].to_numpy()

# Preprocess the data by scaling it to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the Deep Boltzmann Machine model architecture
num_visible = 26
num_hidden = 122
num_iterations = 500

input_layer = Input(shape=(num_visible,))
hidden_layer = Dense(num_hidden, activation='sigmoid')(input_layer)
output_layer = Dense(num_visible, activation='sigmoid')(hidden_layer)

rbm = Model(inputs=input_layer, outputs=output_layer)
rbm.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the Deep Boltzmann Machine model
for i in range(num_iterations):
    rbm.fit(x_train, x_train, batch_size=32, epochs=1, verbose=0)
    print('Iteration:', i)

# Use the trained model to initialize the deep belief network
dbn_layers = [hidden_layer]
for i in range(1, len(rbm.layers)):
    dbn_layers.append(rbm.layers[i])

dbn = Model(inputs=input_layer, outputs=dbn_layers)
dbn.compile(optimizer=Adam(), loss='mean_squared_error')

# Remove the last layer and freeze the remaining layers for feature extraction
dbn.pop()
for layer in dbn.layers:
    layer.trainable = False

# Define the classifier model architecture using the frozen layers for feature extraction
classifier = Sequential()
classifier.add(dbn)
classifier.add(Flatten())
classifier.add(Dense(10, activation='softmax'))

# Compile the classifier model with appropriate loss, optimizer, and metric
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the classifier model using the frozen layers for feature extraction
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
classifier.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the classifier model on the testing data
score = classifier.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])