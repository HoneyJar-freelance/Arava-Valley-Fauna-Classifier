import user_interface.UI as ui
from model_builder import construct_model, construct_dataset
from os.path import isfile

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
        #Load UI and determine what needs to be done
        retrain_model, img_dir, csv_file = ui.loadUI() #gets a tuple of a (boolean, dir_path, filename)
        dataset = construct_dataset.get_data(link=img_dir, classes=classes, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, csvfile=csv_file)

        #we need to retrain model.
        if(retrain_model):
            classes = construct_model.get_new_classes(csv_file, classes) #update classes if we need to
            hist = construct_model.train_model(model=model, dataset=dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=NUM_EPOCHS, validation_steps=VALIDATION_STEPS)
            construct_model.visualize_performance(hist) #For developer's sake.
        else: #we are making predictions
             construct_model.predict()

    except:
        print(f"WARNING: ModelNotFoundError. Attempting to create a new one. If this is a mistake, check for {SAVED_MODEL_NAME}.OLD...")
        proceed = ui.load_dependency_not_found_prompt()
        if(proceed):
            img_dir, csv_file = proceed
            classes = construct_model.get_new_classes(csv_file, None)
            print('Classes extracted...')
            model = construct_model.construct(DENSE_ACTIVATION_0, DENSE_ACTIVATION_1, OPTIMIZER, len(classes))
            print('Model assembled...')
            dataset = construct_dataset.get_data(link=img_dir,batch_size=BATCH_SIZE,val_split=VAL_SPLIT, csvfile=csv_file)
            if(not dataset):
                print('ERROR: Something has gone wrong. Please try again.')
            else:
                print('Training and validation datasets obtained. Beginning training...\n==============================\n')
                construct_model.train_model(model=model,epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, dataset=dataset,steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS)