import tensorflow as tf
from tensorflow import keras
from keras.src.utils.image_utils import get_interpolation
from construct_model import get_labels, extract_classes
import logging
from os import walk
from time import sleep

def create_dataset(path:str, csvfile:str, classes:dict):
    '''
    Creates datasets for running a CNN model on.

    Args:
        path: filepath to directory with images.
        csvfile: path to csv file with labels of respective images.
        classes: dict of class names and their integer encoding.
    
    Rules:
        - csvfile must have the images and labels in the EXACT same order as the file paths. Default: alphabetical order.
        - Files in path must be valid image formats. Valid image formats are as follows:
            - jpg
            - png
            - gif (only first frame will be used)
            - bmp
    
    Returns:
        list[tf.data.Dataset, tf.data.Dataset] if csvfile is specified, else tf.data.Dataset.
    '''
    file_paths = []
    for root, dirs, files in walk(path):
        for file in files:
            valid_file_types = ('jpg', 'jpeg', 'png', 'gif', 'bmp')
            file_type = file.split('.')[-1].lower() #split at . , get last entry (file type), and lowercase it
            if(file_type in valid_file_types):
                file_paths.append(root + '\\' + file)
    
    labels = None
    if(csvfile is not None): #then we are generating predictions
        logging.debug('Attempting to get labels from csv file')
        labels = get_labels(csvfile)
        logging.debug('Success' if labels else 'Failure!!!!!!!!!!!!!!!!!!!!!!!')
    
    if(len(file_paths) == 0):
        raise ValueError('No valid files were found for dataset creation')
    
    if(labels is None):
        raise ValueError("Labels were not constructed.")

    path_ds = tf.data.Dataset.from_tensor_slices(file_paths) #dataset of file_paths
    path_ds = path_ds.flat_map(lambda file: load_image(file), num_parallel_calls=tf.data.AUTOTUNE) #allows images to be loaded on runtime, and applies relevant transformations

    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    label_ds = label_ds.map(lambda label: tf.one_hot(classes[label], len(classes))) #Required step by Tensorflow
    #TODO: #21 add support for multi-label classification (line 34, construct_dataset_new.py)

    train_ds, val_ds = prepare_datasets(path_ds, label_ds)

def prepare_datasets():
    pass

def load_image(path) -> tf.Tensor:
    '''
    Modified from keras.src.utils.image_dataset.load_image
    Loads an image from a file path and applies a series of transformations.

    Args:
        path: file path to image
    
    Returns: float tf.Tensor
    '''
    img = tf.io.read_file(path) #read the file
    img = tf.image.decode_image(img, channels=1, expand_animations=False) #grayscale it
    img = tf.image.grayscale_to_rgb(img) #Back to rgb
    img = img/255 #Normalization
    img = tf.image.resize(img, (224,224), method=get_interpolation('bilinear')) #Resize for transfer learning
    img.set_shape((224, 224, 3))
    return img