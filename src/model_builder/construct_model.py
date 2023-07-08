from keras.applications import VGG16
from keras import Sequential
from keras.layers import Flatten, Dense
from os import path, rename, remove
import json
import pandas as pd

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

def get_labels(csvfile): #TODO: #10 add exception handling for if csvfile isnt a csv here!
   '''
   Extracts the labels column from a csv
   csvfile: filepath to csv file with images and labels
   Returns: list of labels (duplicates included)
   '''
   csvfile =  pd.read_csv(csvfile)
   new_classes = csvfile.loc[:,'Animal'].to_list()
   return new_classes

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
