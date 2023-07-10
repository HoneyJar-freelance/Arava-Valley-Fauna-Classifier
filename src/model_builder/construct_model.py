from keras.applications import VGG16
from keras import Sequential, models
from keras.layers import Flatten, Dense
from os import path, rename, remove
import json
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

def construct(dense_activation_0, dense_activation_1, optimizer, num_classes):
    '''
    Constructs a VGG16 CNN given a set of activation functions, an optimizer, and class size.
    dense_activation_0: str, representing an activation function
    dense_activation_1: str, representing an activation function
    num_classes: int; represents the number of classes available
    Return: keras.Sequential model instance
    '''

    VGG = VGG16(input_shape = (224, 224, 3), 
                include_top = False,
                weights = 'imagenet', 
                classes = num_classes)

    VGG.trainable = False #we dont want to train the first 13 layers, just the last 3.

    #model is flattened to feed into final dense layers
    #in final dense layer, the number of units must match how many possible classifications there are
    #softmax converts all results into a probability of it being that classification (val btw. 0-1)
    model = Sequential([VGG,                  
                        Flatten(),
                        Dense(units = 1024, activation = dense_activation_0),
                        Dense(units = 1024, activation = dense_activation_1),
                        Dense(units = num_classes, activation = 'softmax')])

    #compile the model
    #sparse_categorical_crossentropy: because we have a non-binary classification + want integer encoding, not 1-hot and we are encoding the predictions as integers
    model.compile(optimizer = optimizer,                       
                loss = 'sparse_categorical_crossentropy', 
                metrics = ['accuracy']) #uses MSE
    
    return model

def extract_classes(location):
    '''
    Extracts the classes from a file if it exists.
    location: filepath + name of file we are opening
    Returns: dict instance | 0
    ''' 
    try:
        with open(location, 'r') as classes:
            return json.load(classes)
    except:
        return 0


def get_new_classes(csvfile, current_classes=None):
    '''
    Extracts potential new classes from a csv file
    csvfile: filepath to csv file with images and labels
    current_classes: dict of currently registered classes. Is None if they do not exist
    returns: dict instance
    '''
    classes = {} #empty dict to use
    #if we have pre-existing classes, set classes to point to them
    if(current_classes is not None):
        classes = current_classes
    
    next_entry = len(classes) #len indexes starting at 1, not 0. Thus, next_entry = last value + 1
    for label in get_labels(csvfile):
        if classes.get(label) is None: #if the class doesnt exist
            classes[label] = next_entry #add it to the dictionary with the corresponding int encoding
            next_entry += 1
    
    return classes

def save_classes(classes, location):
    '''
    Saves a dictionary of classes to a json file, and backs up the old file
    classes: dict object of classes
    location: file path + file name to save the json file at. Will also be appended with '.OLD' to backup old file
    '''
    if(path.exists(f'{location}.OLD')): #checks to see if we have an old
        remove(f'{location}.OLD') #if so, remove it for the next one
    try:
        rename(location, f'{location}.OLD') #if json.old existed, its gone now.
    except: #Thus, the only error that can arise is that there is no current class file
        pass #we dont need to do anything
    finally: #always save
        with open(location, 'w') as fp:
            json.dump(obj=classes, fp=fp)

def save_model(model, location):
    pass

def get_labels(csvfile): #TODO: #10 add exception handling for if csvfile isnt a csv here!
   '''
   Extracts the labels column from a csv
   csvfile: filepath to csv file with images and labels
   Returns: list of labels (duplicates included)
   '''
   csvfile =  pd.read_csv(csvfile)
   new_classes = csvfile.loc[:,'Animal'].to_list()
   return new_classes

def load_model(model_name, classes_file):
    '''
    Loads the model and its classes as a dict.
    Returns: tuple(keras.models.Model, dict{classes})
    '''
    return (models.load_model(model_name), extract_classes(classes_file))

def train_model(model:models.Model, classes:dict, dataset:tf.data.Dataset, steps_per_epoch:int|None, epochs:int, validation_steps:int|None):
    es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3) #stops training the network if validation loss doesn't improve after 3 epochs
    
    hist = model.fit(x = dataset[0],                        #the training data
                     steps_per_epoch = steps_per_epoch,     #None defaults to the number of batches
                     epochs = epochs,                       #performed best in experimentation
                     callbacks = es1,                       #Tells the model to monitor es1; in this case, monitor val_loss with patience of 3
                     validation_data = dataset[1],          #the validation data
                     validation_steps = validation_steps,   #None defaults to the number of batches
                     verbose = 1)                           #1 shows progress bar. Helps gauge how much is done
    
    save_model()
    return hist

def visualize_performance(hist):
    '''
    Creates charts displaying the model's accuracy and loss metrics.
    hist: the result of running model.fit()
    '''
    #The following code creates a graph of the accuracy of the model
    plt.title('AVFC Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.show()

    #The following code creates a graph of the loss of the model
    plt.title('AVFC Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['Loss', 'Validation loss'])
    plt.show()