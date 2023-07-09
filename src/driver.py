import UI as ui
from model_builder import construct_model, construct_dataset
import src.jsonReading as jr
from os.path import isfile
from tensorflow import keras

CUTOFF = 0.3

choiceTuple = ui.loadUI() #gets a tuple of both the directory of interest, and a value (None, 0, 1) determining what to do
if(choiceTuple[1] != ""):
    match choiceTuple[0]:
        case 0:
            predictions = cnn.runModel(choiceTuple[1])
            count = jr.getCount(choiceTuple[1])
            cnn.makeCSV(choiceTuple[1], predictions, count, CUTOFF)
        case 1:
            print("currently not enabled. Sorry!")
            for i in range(700):
                beans = True

            #cnn.retrain(choiceTuple[1])
        case _:
            print('no choice made- everything is all good. closing application!')


def dependency_files_exist():
    '''
    Checks to see if the model's .h5 exists and if the classes file exists
    Returns: Boolean True or False if both files exist
    '''
    return isfile(SAVED_MODEL_NAME) and isfile(SAVED_CLASSES_NAME)


if __name__ == '__main__':
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
        model, classes = construct_model.load_model(SAVED_MODEL_NAME, SAVED_CLASSES_NAME)
    except:
        print(f"WARNING: ModelNotFoundError. Attempting to create a new one. If this is a mistake, check for {SAVED_MODEL_NAME}.OLD...")
        ui.load_dependency_not_found_prompt()



    #Load UI and determine what needs to be done
    retrain_model, img_dir, csv_file = ui.loadUI() #gets a tuple of a (boolean, dir_path, filename)
    

    
    if(not dependency_files_exist()): #model and/or class names cannot be found
        #train model
        images_dir, csv_file = ui.load_dependency_not_found_prompt() #TODO: implement; should inform user of error, and ask for training file
        if(csv_file):
            #TODO: call model.train(img_directory, csv_file)
            pass
        else:
            print('Debug: Either program was closed or an error occured')

    elif(retrain_model):
        #TODO: call model.train(img_directory, csv_file)
        pass
    elif(not retrain_model):
        #TODO: call model.run(img_directory)
        pass
    else:
        ui.give_error('Unable to process request.',f'retrain_model: {retrain_model}, img_directory: {img_dir}, csv_file: {csv_file}')
        #TODO: ui should notify user of an error