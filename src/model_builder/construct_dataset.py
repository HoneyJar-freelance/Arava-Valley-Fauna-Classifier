from tensorflow import keras
from keras.utils import image_dataset_from_directory
from construct_model import get_labels, extract_classes

def get_data(link, classes_file, batch_size, val_split=None, csvfile=None):
    '''
    Creates a dataset of images to either be trained on or labeled
    link: directory/dropbox link to data
    classes_file: path to the classes file
    batch_size: batching for the data
    val_split: percentage of dataset to be saved for validation
    csvfile: path to csv file with labels
    '''
    labels = None
    if(csvfile is not None): #then we are generating predictions
        labels = get_labels(csvfile)
    try: #checks if link is a directory
        dataset = image_dataset_from_directory(directory=link,
                                               labels=labels,
                                               label_mode='int', #we want integer encoding
                                               class_names=extract_classes(classes_file),
                                               color_mode='grayscale', #standardizes the data first
                                               batch_size=batch_size,
                                               image_size=(224,224,3),
                                               shuffle=(True if labels else False), #if we are training, we want to shuffle
                                               seed= 19121954, #arbitrarily chosen. RIP Alan Turing
                                               validation_split=val_split, #split only if we are training
                                               subset='both')
        dataset = preprocess(dataset) #we aren't done preprocessing the data.
        return dataset
    except: #TODO: include dropbox link handling
        print('ERROR: NON-DIRECTORY PROVIDED. THE MODEL DOES NOT CURRENTLY SUPPORT DROPBOX LINKS')
        return 0
