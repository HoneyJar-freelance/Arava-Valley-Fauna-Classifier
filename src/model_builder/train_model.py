##### IMPORT SECTION #####
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

#load the model and its classes

#runs the model and saves it as a History object


model.save('vgg16Run.h5') #saves the model as a readable file
print('Saved model to disk') #confirmation message

#The following code creates a graph of the accuracy of the model
plt.title('VGG-16 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

#The following code creates a graph of the loss of the model
plt.title('VGG-16 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Loss', 'Validation loss'])
plt.show()