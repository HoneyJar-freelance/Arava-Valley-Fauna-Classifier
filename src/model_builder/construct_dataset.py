from tensorflow import keras
import tensorflow as tf
from model_builder.construct_model import get_labels, extract_classes
from ReissLib.PickyPixels import image_verification as iv
import logging
from os.path import isdir

def get_data(link:str, batch_size, val_split, csvfile):
    '''
    Creates a dataset of images to either be trained on or labeled
    link: directory/dropbox link to data
    classes_file: path to the classes file
    batch_size: batching for the data
    val_split: percentage of dataset to be saved for validation
    csvfile: path to csv file with labels
    '''

    logging.info('get_data() called.')
    if(not isdir(link)):
        logging.warn(f'directory provided is not a valid directory. Value given:{link}. Returning 0')
        print('Warning: provided directory is not valid')
        return None

    #prune any corrupted images to avoid any issues
    logging.debug('attempting to prune corrupted images')
    print('Pruning corrupted images...')
    #iv.detect_unopenable(link) #TODO: #18 bug report: labels are not removed with its respective image BUG: breaks code

    labels = None
    if(csvfile is not None): #then we are generating predictions
        logging.info('Attempting to get labels from csv file')
        labels = get_labels(csvfile)
    try:
        logging.debug('Trying to construct a dataset')
        print('Constructing dataset...')
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
        logging.info(f'Dataset constructed: {dataset}')
        logging.debug('attempting to call preprocess and dataset')
        
        dataset = preprocess(dataset) #Need to convert all images to RGB for vgg16 weights. Also normalizes data
    except ValueError as e:
        logging.exception(e)
        return 0
    except TypeError as e:
        logging.exception(e)
        return 0
    except:
        logging.exception('Unknown exception thrown.')
        return 0

def preprocess(dataset):
    '''
    Takes in a Dataset object | list[Dataset], normalizes the images, and changes the color mode to RGB.
    Returns: Modified Dataset object
    '''
    logging.info('preprocess() called.')
    logging.info(f'Preprocessing dataset:{dataset}')
    if(isinstance(dataset, list)): #we passed in both a training and validation subset, so process both
        logging.info(f'dataset provided is a list, starting recursive call')
        dataset = [preprocess(dataset[0]), preprocess(dataset[1])] #both = [Dataset, Dataset]
        logging.info(f'dataset post-preprocessing: {dataset}')
        return dataset if (dataset[0] is not None and dataset[1] is not None) else 0
    try:
        logging.debug('Trying to apply map function to the dataset.')
        ''' shape of each BatchDataset instance
        Tensor("args_0:0", shape=(None, 224, 224, 1), dtype=float32) Tensor("args_1:0", shape=(None,), dtype=int32)
        Tensor("args_0:0", shape=(None, 224, 224, 1), dtype=float32) Tensor("args_1:0", shape=(None,), dtype=int32)
        '''
        #convert to rgb images, and normalize dataset.
        dataset.map(lambda image, label: rgb_and_normalize(image, label))
        logging.info(f'Dataset successfully mapped. dataset: {dataset}')
        return dataset
    except ValueError as e:
        logging.exception(msg=f'{e} -- dataset: {dataset}')
        return None
    except TypeError as e:
        logging.exception(msg=f'{e} -- dataset: {dataset}')
        return None
    except IOError as e:
        logging.exception(msg=f'{e} -- dataset: {dataset}')
        return None
    except:
        logging.exception(msg=f'Unknown exception thrown. -- dataset: {dataset}')
        return None

def visualize_data(): #TODO: #14 implement this code
    pass

def rgb_and_normalize(image, label):
    '''
    link in discord for credits
    '''
    return tf.cast(tf.image.grayscale_to_rgb(image), tf.float32)/255, label
