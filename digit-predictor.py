# %%
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# %%
# Pixel value harulai [0, 1] ko range ma lyaune (normalization)
x_train = x_train / 255
x_test = x_test / 255

# %%
# Create keras sequential model
model = keras.Sequential()

# First layer of the neural network, this is where the image data goes.
# The image is 28*28 pixels, so we need to flatten the image into 784 values
# 784 nodes hune vayo yo layer ma
model.add(keras.layers.Flatten(input_shape=(28,28)))

# Second layer of the neural network, this layer is hidden and contains 128 nodes
model.add(keras.layers.Dense(128, activation="relu"))

# Last layer containing 10 nodes. 10 ota number 0, 1, 2, .. 9 vako le
# 10 ota final nodes
# classification ko lagi "softmax" use garne
model.add(keras.layers.Dense(10, activation="softmax"))

# %%
model.summary()

# %%
# Model lai compile hanne, ani train garna milxa
model.compile(loss=keras.losses.SparseCategoricalCrossentropy, optimizer="Adam", metrics=["accuracy"])

# %%
# Model lai train garne, epochs = num of times to train
# validation split = 0.2 vaneko purai data ma 20% chai 
# validation garna ko lagi chuttyayeko
history = model.fit(x_train, y_train, epochs=25, validation_split=0.2)

# %%
# Predict the test inputs
y_probabilities = model.predict(x_test)
y_predictions = y_probabilities.argmax(axis=1)
print("test y values:", y_test)
print("test y predictions:", y_predictions)

# %%
# Aba accuracy mesure garne model ko
# accuray improve garna layer badauna milyo, epoch value ajai dherai halna milyo
# tara overfitting huna sakxa
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predictions)

# %%
# Anaylze the training process

# Plot training and validation loss
plt.figure(figsize=(10, 5)) # figure wide banauxa
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy During Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# %%
# Test our model's output with 10 random test inputs
from random import randrange

for i in range(10):
    input_image = x_test[randrange(len(x_test))]
    plt.imshow(input_image, cmap="gray")
    probabilities = model.predict(input_image.reshape(1, 28, 28))
    prediction = probabilities.argmax(axis=1)[0] # one with highest probability
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show() 
