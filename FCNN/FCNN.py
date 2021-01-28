#! /usr/bin/python3

import time
import numpy as np
import pandas as pd
from numpy import load
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

x = load('../data/thicknesses.npy')
y = load('../data/scatter_CS.npy')

#x_test, x_valid, x_train = (x[:5000,:]-30)/40.0, (x[5000:10000,:]-30)/40.0, (x[10000:,:]-30)/40.0
x_test, x_valid, x_train = x[:5000,:], x[5000:10000,:], x[10000:,:]
y_test_tmp, y_valid_tmp, y_train_tmp = y[:5000,:], y[5000:10000,:], y[10000:,:]

# normalise input data
x_test = ( x_test - np.mean(x_test) )/np.std(x_test)
x_valid = ( x_valid - np.mean(x_valid) )/np.std(x_valid)
x_train = ( x_train - np.mean(x_train) )/np.std(x_train)

# reduces the number of data point by half
y_test = np.empty([5000,200])
y_valid = np.empty([5000,200])
y_train = np.empty([40000,200])
for i in range(200):
    y_test[:,i] = y_test_tmp[:,2*i]
    y_valid[:,i] = y_valid_tmp[:,2*i]
    y_train[:,i] = y_train_tmp[:,2*i]

# build network
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1)
model = keras.models.Sequential()
model.add(keras.layers.InputLayer([8]))
model.add(keras.layers.Dense(500, activation="relu", kernel_initializer=initializer))
model.add(keras.layers.Dense(500, activation="relu", kernel_initializer=initializer))
model.add(keras.layers.Dense(200))

loss = keras.losses.MeanAbsoluteError()
#loss = keras.losses.MeanSquaredError()
#loss = keras.losses.MeanAbsolutePercentageError()

#optimizer = keras.optimizers.RMSprop(lr=0.004, decay=0.99)
#optimizer = keras.optimizers.SGD(lr=0.002, momentum=0.9, nesterov=True)
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# compile network
model.compile(loss=loss, optimizer=optimizer, metrics="accuracy")

# train network
t = time.time()
history = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_data=(x_valid,y_valid))
t = time.time() - t
print("\nTime taken: ", t)

# plot loss against epoch
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.xlabel("Epoch")
plt.ylabel("Loss")

#a = np.array([[48,45,61,62,38,50,48,56]])
#Lambda  = np.linspace(400, 800, 200)
#predict = np.transpose(model.predict(a))
#plt.plot(Lambda,predict)
#plt.xlabel("Wavelength")
#plt.ylabel("Scattering cross section")
plt.show()
