import tensorflow as tf
from tensorflow import keras
from keras.src.utils.image_utils import get_interpolation
from keras.src.utils.dataset_utils import get_training_or_validation_split
from construct_model import get_labels, extract_classes
import logging
from os import walk
from numpy import random

def create_dataset(path:str, csvfile:str, classes:dict, batch_size:int, val_split = None):
    '''
    Creates datasets for use in a CNN model.

    Args:
        path: filepath to directory with images.
        csvfile: path to csv file with labels of respective images.
        classes: dict of class names and their integer encoding.
        batch_size: int that determines how many images are in 1 batch.
        val_split: float|None determining if the data should be split.
    
    Rules:
        - csvfile must have the images and labels in the EXACT same order as the file paths. Default: alphabetical order.
        - Files in path must be valid image formats. Valid image formats are as follows:
            - jpg
            - png
            - gif (only first frame will be used)
            - bmp
        - If val_split is specified, csvfile must also be specified.
    
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
    
    if(labels is None and (val_split is None or val_split == 0)):
        raise ValueError('Both csvfile and val_split must be specified if either is specified.'
                         f'csvfile file given: {csvfile}'
                         f'val_split given: {val_split}')

    path_ds = tf.data.Dataset.from_tensor_slices(file_paths) #dataset of file_paths
    path_ds = path_ds.flat_map(lambda file: load_image(file), num_parallel_calls=tf.data.AUTOTUNE) #allows images to be loaded on runtime, and applies relevant transformations


    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    label_ds = label_ds.map(lambda label: tf.one_hot(classes[label], len(classes))) #Required step by Tensorflow
    #TODO: #21 add support for multi-label classification (line 34, construct_dataset_new.py)

    train_ds, val_ds = prepare_datasets(path_ds, label_ds, val_split, batch_size)

def prepare_datasets(path_ds:tf.data.Dataset, label_ds:tf.data.Dataset, val_split:float, batch_size:int) -> list[tf.data.Dataset]:
    
    #handle exceptions
    if(not batch_size):
        raise ValueError(f'Invalid value for batch_dataset. Should be int > 0, given {batch_size}')
    
    if(val_split): #Then we are training, thus shuffle data + split data
        train_imgs, train_labels = get_training_or_validation_split(path_ds, label_ds, val_split, 'training')
        val_imgs, val_labels = get_training_or_validation_split(path_ds, label_ds, val_split, 'validation')

        seed = random(1e-6)
        path_ds.shuffle(buffer_size= batch_size*8, seed=seed)

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