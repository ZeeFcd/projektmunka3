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

# Preprocess the data by scaling it to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the stacked autoencoder model architecture
input_shape = (26,)
encoding_dim1 = 122
encoding_dim2 = 13
encoding_dim3 = 2

input_layer = keras.Input(shape=input_shape)
encoder1 = layers.Dense(encoding_dim1, activation='relu')(input_layer)
encoder2 = layers.Dense(encoding_dim2, activation='relu')(encoder1)
encoder3 = layers.Dense(encoding_dim3, activation='relu')(encoder2)

decoder1 = layers.Dense(encoding_dim2, activation='relu')(encoder3)
decoder2 = layers.Dense(encoding_dim1, activation='relu')(decoder1)
output_layer = layers.Dense(26, activation='sigmoid')(decoder2)

model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model with appropriate loss, optimizer, and metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, x_train, batch_size=32, epochs=10, validation_split=0.1)

# Remove the decoder layers and use the encoder part as the feature extractor
encoder = keras.Model(inputs=input_layer, outputs=encoder3)
encoded_x_train = encoder.predict(x_train)
encoded_x_test = encoder.predict(x_test)

# Define the classifier model architecture using the encoded features
classifier = keras.Sequential(
    [
        layers.Dense(128, activation='relu', input_shape=(encoding_dim3,)),
        layers.Dense(6, activation='softmax')
    ]
)

# Compile the classifier model with appropriate loss, optimizer, and metric
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the classifier model using the encoded features
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
classifier.fit(encoded_x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

# Evaluate the classifier model on the testing data
score = classifier.evaluate(encoded_x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])