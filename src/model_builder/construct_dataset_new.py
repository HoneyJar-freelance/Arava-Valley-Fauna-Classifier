import tensorflow as tf
from tensorflow import keras
from keras.src.utils.image_utils import get_interpolation
from keras.src.utils.dataset_utils import get_training_or_validation_split
from model_builder.construct_model import get_labels
import logging
from os import walk
from numpy.random import randint

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
    if(csvfile is not None): #then we are training
        logging.debug('Attempting to get labels from csv file')
        labels = []
        for label in get_labels(csvfile):
            labels.append(classes[label]) #int encoding
        logging.debug('Success' if len(labels) > 0 else 'Failure!!!!!!!!!!!!!!!!!!!!!!!')
    
    if(len(file_paths) == 0):
        raise ValueError('No valid files were found for dataset creation')
    
    if(not bool(labels) and not bool(val_split)):
        raise ValueError('Both csvfile and val_split must be specified if either is specified.'
                         f'csvfile file given: {csvfile}'
                         f'val_split given: {val_split}')
    
    if(len(labels) != len(file_paths)):
        raise ValueError('Length mismatch on labels and file_paths:'
                         f'#labels: {len(labels)}'
                         f'#files : {len(file_paths)}')
    

    #TODO: #21 add support for multi-label classification (line 34, construct_dataset_new.py)

    train_ds, val_ds = associate_labels_with_data(file_paths, labels, val_split, batch_size, classes)
    #TODO: return

def associate_labels_with_data(file_paths:list[str], labels:list[int], val_split:float, batch_size:int, classes:dict) -> list[tf.data.Dataset]:
    '''
    Creates a dataset(s) used by the program for image classification.

    Args:
        path_ds: tf.data.Dataset that contains a list of file paths with mapped transformations
        labels_ds: tf.data.Dataset that contains a list of labels (int encoded) corresponding to path_ds
        val_split: The validation split (range of [0, 1)) of the data. != 0 if training, = 0 else
        batch_size: The number of images per batch. Excess is not dropped
    
        Rules:
            - len(path_ds) == len(labels_ds)
            - 0 <= val_split < 1
            - batch_size > 0
    
    Returns:
        list[tf.data.Dataset] if val_split is non-zero, else a singular tf.data.Dataset. The dataset is combined from path_ds and labels_ds.
    '''

    #handle exceptions
    if(not file_paths):
        raise ValueError('You must specify a dataset object for path_ds. None/Invalid given.')
    if(not batch_size or batch_size < 1):
        raise ValueError(f'Invalid value for batch_dataset. Should be int > 0, given {batch_size}')
    if(bool(labels) != bool(val_split)):
        raise ValueError('Invalid arguments for label_ds and val_split. Both must be present or absent.'
                         f'label_ds: {labels}     val_split: {val_split}')
    if(labels and (len(file_paths) != len(labels))):
        raise ValueError('Length mismatch on labels and file_paths:'
                         f'#labels: {len(labels)}'
                         f'#files : {len(file_paths)}')

    
    #determine if we are training or predicting
    if(labels): #Then we are training, thus shuffle data + split data
        if(val_split < 0 or val_split >= 1):
            raise ValueError(f'Invalid val_split given. Should be in range [0, 1). Given: {val_split}')
        logging.debug('attempting to split the data...')
        try:
            train_imgs, train_labels = get_training_or_validation_split(file_paths, labels, val_split, 'training')
            val_imgs, val_labels = get_training_or_validation_split(file_paths, labels, val_split, 'validation')
        except Exception as Arugment:
            logging.exception('ERROR: FAILED TO SPLIT DATA. MESSAGE: ')
            return 0
        logging.debug('Success.')

        logging.debug('Mapping functions to dataset...')
        #creates an formats the training datasets
        train_img_ds, train_labels_ds = prep_dataset(img_ds=tf.data.Dataset.from_tensor_slices(train_imgs), 
                                                     label_ds=tf.data.Dataset.from_tensor_slices(train_labels),
                                                     classes=classes)
        #creates and formats the validation datasets
        val_img_ds, val_labels_ds = prep_dataset(img_ds=tf.data.Dataset.from_tensor_slices(val_imgs), 
                                                     label_ds=tf.data.Dataset.from_tensor_slices(val_labels),
                                                     classes=classes)
        logging.debug('Success.')
        #combine image datasets with their respective labels
        logging.debug('Attempting to associate data with its labels')
        try:
            train_ds = tf.data.Dataset.zip((train_img_ds, train_labels_ds))
            val_ds = tf.data.Dataset.zip((val_img_ds, val_labels_ds))
        except Exception as Arguement:
            logging.exception('ERROR: FAILED TO COMBINE DATA WITH LABELS. Message: ')
            return 0
        logging.debug('Success.')

        #shuffle data
        seed = randint(1e6)
        train_ds = train_ds.shuffle(buffer_size= batch_size*8, seed=seed)
        
        #batch datasets
        train_ds = train_ds.batch(batch_size=batch_size)
        val_ds = val_ds.batch(batch_size=batch_size)

        return [train_ds, val_ds]
    
    else: #then we are generating predictions on the dataset
        return prep_dataset(img_ds=tf.data.Dataset.from_tensor_slices(file_paths), label_ds=None, classes=classes)

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

def prep_dataset(img_ds:tf.data.Dataset, label_ds:tf.data.Dataset, classes:dict) -> list[tf.data.Dataset]:
    '''
    Maps load_image onto each image file path, and one-hot encodes the label dataset if it exists.

    Args:
        img_ds: tf.data.Dataset containing file paths.
        label_ds: tf.data.Dataset containing the labels for each image.

    Returns:
        list[img_ds, label_ds]
    '''
    logging.info('prep_dataset called.')
    logging.debug(f'attempting to map the dataset. value before: {img_ds}')
    try:
        img_ds = img_ds.map(lambda file: load_image(file), num_parallel_calls=tf.data.AUTOTUNE) #allows images to be loaded on runtime, and applies relevant transformations
        logging.debug(f'Mapped. New value: {img_ds}')
    except Exception as Argument:
        logging.exception('ERROR: could not map to img_ds. Message: ')
        return 0
    if(label_ds is not None):
        logging.debug(f'label_ds provided. Attempting to one-hot encode...')
        try:
            label_ds = label_ds.map(lambda label: tf.one_hot(label, len(classes)), num_parallel_calls=tf.data.AUTOTUNE) #Required step by Tensorflow
            logging.debug(f'Mapped. label_ds value: {label_ds}')
        except Exception as Argument:
            logging.exception('ERROR: could not map to label_ds. Message: ')
            return 0
        return [img_ds, label_ds]
    else:
        return img_ds