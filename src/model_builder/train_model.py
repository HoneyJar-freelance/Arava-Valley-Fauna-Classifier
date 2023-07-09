##### IMPORT SECTION #####
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

#load the model and its classes

#runs the model and saves it as a History object
es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3) #stops training the network if validation loss doesn't improve after 3 epochs
hist = model.fit(x = trainData[0],                 #the training data
                 steps_per_epoch = None,           #None defaults to the number of batches
                 epochs = 15,                      #performed best in experimentation
                 callbacks = es1,                  #Tells the model to monitor es1; in this case, monitor val_loss with patience of 3
                 validation_data = trainData[1],   #the validation data
                 validation_steps = None,          #None defaults to the number of batches
                 verbose = 1)                      #1 shows progress bar. Helps gauge how much is done

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