from tensorflow import keras
import tensorflow as tf
from model_builder.construct_model import get_labels, extract_classes
from ReissLib.PickyPixels import image_verification as iv

def get_data(link:str, classes_file:str, batch_size:int|None, val_split=float|None, csvfile=str|None):
    '''
    Creates a dataset of images to either be trained on or labeled
    link: directory/dropbox link to data
    classes_file: path to the classes file
    batch_size: batching for the data
    val_split: percentage of dataset to be saved for validation
    csvfile: path to csv file with labels
    '''

    #prune any corrupted images to avoid any issues
    iv.detect_unopenable(link)

    labels = None
    if(csvfile is not None): #then we are generating predictions
        labels = get_labels(csvfile)
    try: #checks if link is a directory
        dataset = keras.utils.image_dataset_from_directory(directory=link,
                                               labels=labels,
                                               label_mode='int', #we want integer encoding
                                               class_names=extract_classes(classes_file),
                                               color_mode='grayscale', #ensures all images are grayscale, as few arent
                                               batch_size=batch_size,
                                               image_size=(224,224,3),
                                               shuffle=(True if labels else False), #if we are training, we want to shuffle
                                               seed= 19121954, #arbitrarily chosen. RIP Alan Turing
                                               validation_split=val_split, #split only if we are training
                                               subset='both')
        dataset = preprocess(dataset) #Need to convert all images to RGB for vgg16 weights. Also normalizes data
        return dataset
    except: #TODO: include dropbox link handling
        print('ERROR: NON-DIRECTORY PROVIDED. THE MODEL DOES NOT CURRENTLY SUPPORT DROPBOX LINKS')
        return 0

def preprocess(dataset:tf.data.Dataset|list[tf.data.Dataset]):
    '''
    Takes in a Dataset object, normalizes the images, and changes the color mode to RGB.
    Returns: Modified Dataset object
    '''
    if(type(dataset) == list): #we passed in both a training and validation subset, so process both
        dataset = [preprocess(dataset[0]), preprocess(dataset[1])] #both = [Dataset, Dataset]
        return dataset if (dataset[0] and dataset[1]) is not None else 0
    try:
        dataset = dataset.map(lambda x: tf.image.grayscale_to_rgb(x/255)) #divide by 255 to normalize, then rgb the images
        return dataset
    except:
        return None

def visualize_data(): #TODO: #14 implement this code
    pass
