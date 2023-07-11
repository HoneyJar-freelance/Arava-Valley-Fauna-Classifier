from tensorflow import keras
import tensorflow as tf
from model_builder.construct_model import get_labels, extract_classes
from ReissLib.PickyPixels import image_verification as iv
import logging

# Create a logging instance
logger = logging.getLogger('my_application')
logger.setLevel(logging.INFO) # you can set this to be DEBUG, INFO, ERROR

# Assign a file-handler to that instance
fh = logging.FileHandler("file_dir.log")
fh.setLevel(logging.INFO) # again, you can set this differently

# Format your logs (optional)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter) # This will set the format to the file handler

# Add the handler to your logging instance
logger.addHandler(fh)

def get_data(link:str, classes:dict, batch_size:int|None, val_split=float|None, csvfile=str|None):
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
    try:
        dataset = keras.utils.image_dataset_from_directory(directory=link,
                                                labels=labels,
                                                label_mode='int', #we want integer encoding
                                                color_mode='grayscale', #ensures all images are grayscale, as few arent
                                                batch_size=batch_size,
                                                image_size=(224,224),
                                                shuffle=(True if labels else False), #if we are training, we want to shuffle
                                                seed= 19121954, #arbitrarily chosen. RIP Alan Turing
                                                validation_split=val_split, #split only if we are training
                                                subset='both')
        dataset = preprocess(dataset) #Need to convert all images to RGB for vgg16 weights. Also normalizes data
    except ValueError as e:
        logger.exception(e)
        return 0
    except TypeError as e:
        logger.exception(e)
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
    except ValueError as e:
        logger.exception(e)
        return None
    except TypeError as e:
        logger.exception(e)
        return None
    except IOError as e:
        logger.exception(e)
        return None

def visualize_data(): #TODO: #14 implement this code
    pass
