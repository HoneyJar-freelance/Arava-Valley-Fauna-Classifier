from keras.applications import VGG16
from keras import Sequential, models
from keras.layers import Flatten, Dense
from os import path, rename, remove
import json
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import logging

def construct(dense_activation_0, dense_activation_1, optimizer, num_classes):
    '''
    Constructs a VGG16 CNN given a set of activation functions, an optimizer, and class size.
    dense_activation_0: str, representing an activation function
    dense_activation_1: str, representing an activation function
    num_classes: int; represents the number of classes available
    Return: keras.Sequential model instance
    '''
    logging.info('construct() called.')
    
    logging.debug('Attempting to make VGG model')
    try:
        VGG = VGG16(input_shape = (224, 224, 3), 
                    include_top = False,
                    weights = 'imagenet', 
                    classes = num_classes)
        logging.info('Weights assigned.')

        VGG.trainable = False #we dont want to train the first 13 layers, just the last 3.
        logging.info('VGG.trainable: False')

        #model is flattened to feed into final dense layers
        #in final dense layer, the number of units must match how many possible classifications there are
        #softmax converts all results into a probability of it being that classification (val btw. 0-1)
        model = Sequential([VGG,                  
                            Flatten(),
                            Dense(units = 1024, activation = dense_activation_0),
                            Dense(units = 1024, activation = dense_activation_1),
                            Dense(units = num_classes, activation = 'softmax')])
        logging.info('Last three layers added successfully.')
        #compile the model
        #sparse_categorical_crossentropy: because we have a non-binary classification + want integer encoding, not 1-hot and we are encoding the predictions as integers
        model.compile(optimizer = optimizer,                       
                    loss = 'sparse_categorical_crossentropy', 
                    metrics = ['accuracy']) #uses MSE
        logging.info('Model compiled.')
        
        return model
    except:
        logging.exception('Unknown Error: Could not construct model.')
        return 0

def extract_classes(location):
    '''
    Extracts the classes from a file if it exists.
    location: filepath + name of file we are opening
    Returns: dict instance | 0
    ''' 
    logging.info('extract_classes() called')
    try:
        logging.debug('Trying to read classes file')
        with open(location, 'r') as classes:
            return json.load(classes)
    except:
        logging.debug('Classes file not found. Returning 0')
        return 0


def get_new_classes(csvfile, current_classes, class_file):
    '''
    Extracts potential new classes from a csv file
    csvfile: filepath to csv file with images and labels
    current_classes: dict of currently registered classes. Is None if they do not exist
    returns: dict instance
    '''
    logging.info('get_new_classes() called')
    classes = {} #empty dict to use
    #if we have pre-existing classes, set classes to point to them
    if(current_classes is not None):
        logging.debug('current_classes is not empty... prepping to add to it if necessary.')
        classes = current_classes
    logging.debug(f'classes: {"not empty" if classes else "empty"}')
    did_we_update = len(classes) #starting length
    next_entry = len(classes) #len indexes starting at 1, not 0. Thus, next_entry = last value + 1
    logging.debug(f'class dict next_entry:{next_entry}')
    for label in get_labels(csvfile):
        if classes.get(label) is None: #if the class doesnt exist
            logging.debug(f'found new class. it is: {label}')
            classes[label] = next_entry #add it to the dictionary with the corresponding int encoding
            next_entry += 1
    
    if(did_we_update != next_entry): #Then we added 1+ classes
        save_classes(classes, f'model_files\{class_file}')
    return classes

def save_classes(classes, location):
    '''
    Saves a dictionary of classes to a json file, and backs up the old file
    classes: dict object of classes
    location: file path + file name to save the json file at. Will also be appended with '.OLD' to backup old file
    '''
    logging.info('save_classes() called.')
    if(path.exists(f'{location}.OLD')): #checks to see if we have an old
        logging.debug('classes OLD file found. attempting to remove it')
        remove(f'{location}.OLD') #if so, remove it for the next one
    try:
        logging.debug('Trying to rename old class file as OLD')
        rename(location, f'{location}.OLD') #if json.old existed, its gone now.
    except:
        logging.exception('OldFileNotFoundError. This should not be an issue unless it exists')
        pass #we dont need to do anything
    finally: #always save
        logging.debug('Attempting to save new classes file. Warning: will overwrite if it exists but was not found in revious block.')
        with open(location, 'w') as fp:
            json.dump(obj=classes, fp=fp)
            logging.debug('file created.')

def save_model(model:tf.keras.models.Model, location:str):
    '''
    Saves the model as a .h5 file.
    model: tf.keras.models.Model instance
    location: file path + file name to save the model file at. Will also be appended with '.OLD' to backup old file
    '''
    logging.info('save_model() called.')
    if(path.exists(f'{location}.OLD')): #checks to see if we have an old
        logging.debug('model OLD file found. attempting to remove it')
        remove(f'{location}.OLD') #if so, remove it for the next one
    try:
        logging.debug('Trying to rename old model file as OLD')
        rename(location, f'{location}.OLD') #if .old existed, its gone now.
    except: 
        logging.exception('OldFileNotFoundError. This should not be an issue unless it exists')
        pass #we dont need to do anything
    finally: #always save
        logging.debug('Attempting to save new model file. Warning: will overwrite if it exists but was not found in revious block.')
        model.save(f'{location}')

def get_labels(csvfile): #TODO: #10 add exception handling for if csvfile isnt a csv here!
   '''
   Extracts the labels column from a csv
   csvfile: filepath to csv file with images and labels
   Returns: list of labels (duplicates included)
   '''
   logging.info('get_labels() called.')
   try: #csvfile might be None
    logging.debug('Trying to extract data from a csv file')
    csvfile =  pd.read_csv(csvfile)
    logging.debug('Success.')
    logging.debug('Trying to extract labels')
    labels = csvfile.loc[:,'Animal'].to_list()
    logging.debug('Success.')
    return labels
   except:
       logging.exception('Unknown exception thrown. Labels were not able to be extracted')
       return None

def load_model(model_name, classes_file):
    '''
    Loads the model and its classes as a dict.
    Returns: tuple(keras.models.Model, dict{classes})
    '''
    logging.info('load_model() called.')
    return (models.load_model(model_name), extract_classes(classes_file))

def train_model(model:models.Model, dataset:tf.data.Dataset, steps_per_epoch:int|None, epochs:int, batch_size:int, validation_steps:int|None, model_name:str):
    '''
    Trains a model on a given dataset with specified hyperparameters.

    Args:
    model: a tf.keras.models.Model
    dataset: a list[tf.data.Dataset] of training and validation data
    steps_per_epoch: int referring to how many gradient updates per epoch to do
    epochs: int; represents the number of epochs to run the model for
    batch_size: int; representing the number of images per batch
    validation_steps: int; represents how many backpropagations per run on the validation data. For best performance, this should be None.
    
    Returns: History object
    '''
    logging.info('train_model() called.')
    es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3) #stops training the network if validation loss doesn't improve after 3 epochs
    logging.debug(f'Early stopping set. es1:{es1}')
    try:
        logging.debug('Trying to train the model. Running fit()...')
        hist = model.fit(x = dataset[0],                        #the training data
                        steps_per_epoch = steps_per_epoch,     #None defaults to the number of batches
                        epochs = epochs,                       #performed best in experimentation
                        batch_size=batch_size,                 #memory allocation issues
                        callbacks = es1,                       #Tells the model to monitor es1; in this case, monitor val_loss with patience of 3
                        validation_data = dataset[1],          #the validation data
                        validation_steps = validation_steps,   #None defaults to the number of batches
                        verbose = 1)                           #1 shows progress bar. Helps gauge how much is done
        
        save_model(model, f'model_files\{model_name}')
        return hist
    except:
        logging.exception('CRITICAL!!! MODEL WAS UNABLE TO BE TRAINED: UNKNOWN EXCPETION: RETURNING 0')
        return 0

def visualize_performance(hist):
    '''
    Creates charts displaying the model's accuracy and loss metrics.
    hist: the result of running model.fit()
    '''
    logging.info('visualize_model() called.')
    logging.info('Saving model accuracy graph')
    #The following code creates a graph of the accuracy of the model
    plt.title('AVFC Model Accuracy (fMSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.show()

    logging.info('Saving model loss graph (fCE)')
    #The following code creates a graph of the loss of the model
    plt.title('AVFC Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['Loss', 'Validation loss'])
    plt.show()