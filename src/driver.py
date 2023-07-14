import user_interface.UI as ui
from model_builder import construct_model, construct_dataset
from os.path import isfile
import logging
from datetime import datetime as dt

def dependency_files_exist():
    '''
    Checks to see if the model's .h5 exists and if the classes file exists
    Returns: Boolean True or False if both files exist
    '''
    return isfile(SAVED_MODEL_NAME) and isfile(SAVED_CLASSES_NAME)


if __name__ == '__main__':
    #creates a debugging tool
    timestamp = dt.now().strftime('%d-%m-%Y %H-%M-%S')
    logging.basicConfig(filename=f'logs\{timestamp}.log', level=logging.DEBUG, encoding='utf-8', datefmt='%d-%m-%Y %H:%M:%S %p', format='%(asctime)s %(levelname)s | %(message)s')
    #globals:
    SAVED_MODEL_NAME = 'AVFC_model.h5'
    SAVED_CLASSES_NAME = 'classes.json'

    #hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    VAL_SPLIT = 0.3
    STEPS_PER_EPOCH = None
    VALIDATION_STEPS = None
    OPTIMIZER = 'adagrad'
    DENSE_UNITS = 1024
    DENSE_ACTIVATION_0 = 'selu'
    DENSE_ACTIVATION_1 = 'relu'

    #attempt to load the model
    model = None
    classes = None
    try:
        logging.debug('Attempting to load a pre-existing model')
        model, classes = construct_model.load_model(SAVED_MODEL_NAME, SAVED_CLASSES_NAME)
        #Load UI and determine what needs to be done
        logging.debug('Attempting to load GUI')
        retrain_model, img_dir, csv_file = ui.loadUI() #gets a tuple of a (boolean, dir_path, filename)
        logging.info(f'tuple of info obtained: (retrain_model:{retrain_model}, img_dir:{img_dir}, csv_file:{csv_file})')
        logging.debug('Attempting to create dataset')
        dataset = construct_dataset.get_data(link=img_dir, classes=classes, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, csvfile=csv_file)

        #we need to retrain model.
        if(retrain_model):
            logging.debug('Attempting to retrain the model')
            classes = construct_model.get_new_classes(csv_file, classes) #update classes if we need to
            logging.info(f'Classes constructed. classes: {classes}')
            logging.debug('Attempting to train model. Here we go...')
            hist = construct_model.train_model(model=model, dataset=dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=NUM_EPOCHS, validation_steps=VALIDATION_STEPS)
            logging.info('Model successfully trained! Yipee!')
            construct_model.visualize_performance(hist) #For developer's sake.
        else: #we are making predictions
             logging.debug('Attempting to generate predictions')
             construct_model.predict()

    except:
        logging.warn(f"WARNING: ModelNotFoundError. Attempting to create a new one. If this is a mistake, check for {SAVED_MODEL_NAME}.OLD...")
        logging.info('Calling ui.load_dependency_not_found_prompt()')
        proceed = ui.load_dependency_not_found_prompt()
        if(proceed):
            img_dir, csv_file = proceed
            logging.info(f'User chose to proceed. Given values -> img_dir: {img_dir}  csv_file: {csv_file}')
            logging.debug('Attempting to get classes')
            classes = construct_model.get_new_classes(csv_file, None)
            print('Classes extracted...')
            model = construct_model.construct(DENSE_ACTIVATION_0, DENSE_ACTIVATION_1, OPTIMIZER, len(classes))
            print('Model assembled...')
            dataset = construct_dataset.get_data(link=img_dir,batch_size=BATCH_SIZE,val_split=VAL_SPLIT, csvfile=csv_file)
            if(not dataset):
                logging.error('Unknown issue occured. Dataset was not created properly. Terminating...')
            else:
                logging.debug('Beginning training...')
                print('Training and validation datasets obtained. Beginning training...\n==============================\n')
                construct_model.train_model(model=model,epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, dataset=dataset,steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS)