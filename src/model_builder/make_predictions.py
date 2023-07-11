import sys
sys.path.append('src')
from model_builder import construct_dataset
from tensorflow import keras
import numpy as np
from os import walk
import logging

def predict(data_dir:str, classes:dict, batch_size:int, model:keras.models.Model):
    '''
    Generates predictions on a specified dataset.
    data_dir: directory path storing all the images to be processed
    classes: dict instance of classes
    batch_size: int representing how many images are in a batch
    model: CNN model generating the predictions
    '''
    logging.info('predict() called.')
    for root, dirs, files in walk(data_dir):
        for folder in dirs:
            logging.debug('attempting to filter out nonimage files from given directory')
            filenames = filter(filter_imgs, files) #files in current directory
            
            if(len(filenames)): #if there are image files
                logging.debug(f'{len(filenames)} images found.')
                logging.debug('Attempting to create dataset.')
                dataset = construct_dataset.get_data(link=data_dir, classes_file=classes, batch_size=batch_size)
                logging.debug(f'Dataset created. dataset:{dataset}')
                predictions = None

                try: #attempt to use multi-processing
                    logging.debug('Trying to use multiprocessing')
                    predictions = model.predict(x=dataset, batch_size=batch_size,verbose=1, use_multiprocessing=True)
                except:
                    logging.exception('Failed for unknown exception. Attempting to run without multi-processing...')
                    try:
                        predictions = model.predict(x=dataset, batch_size=batch_size,verbose=1) #if you cant, dont
                    except:
                        logging.exception('Unknown exception occurred. Reason was not multi-processing. Returning 0')
                        return 0

                logging.debug('Calling map_predictions_to_class()')
                predictions = map_predictions_to_class(classes, predictions) #decode predictions
                logging.debug('Attempting to create csv file with results')
                create_csv(filenames=filenames, relative_path=folder, root=root, predictions=predictions) #create csv file
    

def filter_imgs(filename:str):
    '''
    Filters out files that are not a valid image type.
    filename: str instance; name of a file
    
    Returns: boolean; true if valid, false if not
    '''

    #Not putting a logger here. TODO: Extensive testing jic
    filename = filename.lower() #standarizes mixed cases
    is_img = False

    if('.jpg' in filename):
        is_img = True

    elif('.jpeg' in filename):
        is_img = True
        
    elif('.png' in filename):
        is_img = True

    elif('.bmp' in filename):
        is_img = True

    elif('.gif' in filename):
        is_img = True
    
    return is_img

def map_predictions_to_class(classes:dict, predictions:np.ndarray):
    '''
    Decodes the integer encoding of each prediction matching the class dict.
    classes_file: file path to classes file
    predictions: numpy array(s) of predictions

    Returns: list(tuple) of decoded predictions. [(class, probability), ...]
    '''

    logging.info('map_predictions_to_class() called.')

    predictions_decoded = []
    for predict in predictions:
        logging.debug('Extracting encoded values')
        encoded_animal = np.argmax(predict) #gets the integer corresponding to the highest probability
        confidence_val = np.max(predict)    #gets the probability of said integer

        #decodes integer encoding
        logging.debug('Decoding values')
        decoded_animal = list(classes)[list(classes.values()).index(encoded_animal)]
        predictions_decoded.append((decoded_animal, confidence_val))
    
    return predictions_decoded

def create_csv(filenames:list[str], relative_path:str, root:str, predictions:list[tuple[int,float]]):
    '''
    Creates a CSV file to be imported into TimeLapse.
    filenames: list of image files in a directory of interest
    relative_path: folder the images are currently in
    root: root directory of the images, for the purposes of creating the csv
    predictions: list of prediction tuples. [(label, probability), ...]
    '''

    logging.info('create_csv() called.')
    try:
        logging.debug('Trying to write to a csv file...')
        with open(f'{root}\{relative_path}\labeled_data.csv','w') as fp:
            fp.write('File,RelativePath,Animal,Count\n')
            logging.info('Created header... Confirming that csv file is constructed.')

            if(len(predictions) == len(filenames)): #ensure proper alignment
                logging.debug('Number of predictions lines up with the number of files')
                for prediction, i in enumerate(predictions):
                    fp.write(f'{filenames[i]},{relative_path},{prediction[0]},1\n') #TODO: get actual count
            else:
                logging.error(f'The number of predictions did not align with the number of files. #predictions:{len(predictions)} #files:{len(filenames)}')
    except:
        logging.exception('CRITICAL ERROR! RESULTS WERE NOT SAVED PROPERLY. PLEASE RERUN THE MODEL AND/OR CONTACT THE DEVELOPERS!!!')
'''
TODO list:
1. get season from image metadata
2. get day/night/dawn/dusk from metadata
3. deprecate megadetector as it is slow and unreliable
'''