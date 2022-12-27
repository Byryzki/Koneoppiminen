import tensorflow as tf
import numpy as np
import pickle
from keras import datasets, layers, models, optimizers, metrics, losses
import matplotlib.pyplot as plt

''' Convolution Neural Network with TensorFlow and CIFAR-10 '''

#Loading data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#For making a onehot vector of class names
'''
train_ll = []
test_ll = []

def OneHotClass(clsnro):
    vect = [0,0,0,0,0,0,0,0,0,0]
    vect[clsnro] = 1
    return vect

for sublist in train_labels:
    train_ll.append(OneHotClass(sublist[0]))

for sublist in test_labels:
    test_ll.append(OneHotClass(sublist[0]))
'''
#For checking the input data
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel(class_names[test_labels[i][0]])
plt.show()
'''

#Normalizing pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

#Simple NN
model = models.Sequential()

#spiced with couple of convolutional layers
model.add(layers.Conv2D(32, 5, strides=2, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, 5, strides=2, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, 5, strides=2, activation='relu', input_shape=(32, 32, 3)))

#Input layer
model.add(layers.Flatten())
model.add(layers.Dense(5, activation='sigmoid'))

#Output layer
model.add(layers.Dense(10))

model.summary()

optimizers.SGD(learning_rate=0.5)

#Decided to use integer values in labels instead of onehot vectors
#Visualizing as presented in Jupyter example

model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=30, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(f"final accuracy: {test_acc}")
