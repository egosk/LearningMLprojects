import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalize data
train_images = train_images/255.0
test_images = test_images/255.0

# 28 pixels x 28 = 784 neurons in input layer
# 10 output neurons since  10 classes
# hidden layer: maybe 15-20% of input size

model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28, 28)),  # flatten means [[1], [2], [3]] --> [1, 2, 3]
    keras.layers.Dense(128, activation="relu"),  # dense - each neuron connected to each in nxt layer
    keras.layers.Dense(10, activation="softmax")  # softmax -  probability of each output
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5) # how many times model will see each image

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("test_acc: ", test_acc)

predictions = model.predict(test_images)

# display 5 images and prediction for those
plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()

