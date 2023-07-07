import tensorflow as tf
from tensorflow import keras
import pandas as pd

def construct_dataset(img_dir, csv_file, hyperparameters):
    '''
    Creates a tf.data.Dataset object containing all the images to train on and their labels, with a dynamic class.
    '''
    dataframe = pd.read_csv(csv_file)
    labels = dataframe["label"] #gets the individual labels of each image
    old_classes = fetch_classes() #loads pre-existing classes
    new_classes = dataframe['label'].unique() #just gets unique cases of new classes
    old_classes.join(new_classes)
    classes = old_classes.unique()

    dataset = keras.utils.image_dataset_from_directory(directory=img_dir,
                                             labels=labels,
                                             class_names=classes,
                                             color_mode='grayscale',
                                             batch_size=hyperparameters['BATCH_SIZE'],
                                             image_size=(224,224),
                                             shuffle=True,
                                             seed=19121954,
                                             validation_split=hyperparameters['VAL_SPLIT'],
                                             subset='both')
    
    dataset = tf.data.Dataset.apply(format_dataset())
    return dataset

#ImageNet only works with RGB.
#The following function converts grayscale images to RGB, and fixes the dataset
#dataset: a tf.data.dataset object, in this case, a dataset of images
#returns: a new tf.data.dataset object, in this case, the newly reformatted dataset
def format_dataset(dataset):
     #TODO: try tf.image.grayscale_to_rgb(dataset). it may work
     
     imgList = [] #list of all RGB images
     imgLabels = [] #list of labels assigned to each image
     #for every setOfBatches in dataset:
     #convert img to rgb, add it to imgList
     #add label to imgLabels
     print('=========== PREPROCESSING: ===========\n')
     batchCount = 1 #for debugging purposes 
     print('========\n', len(dataset), 'batches to process. Beginning ...')
     for setOfBatches in dataset:  #set of batches contains a batch of images and a batch of labels for each image
        for img in setOfBatches[0]: #setOfBatches[0] = batch of images
            img = tf.image.grayscale_to_rgb(img) #converts image to RGB format
            imgList.append(img) #adds to list 

            imageNum += 1
       
        for label in setOfBatches[1]: #setOfBatches[1] = labels
            imgLabels.append(label) #adds to list

        print('batch ', batchCount, 'completed. ', (round((batchCount/len(dataset) * 100), 2)), '%', ' finished.') #more debug text
        batchCount += 1 #even more debug stuff
     print("=========== PREPROCESSING COMPLETE ===========")

    #creates a new BatchDataset from imgList and imgLabels
     newTrainData = tf.data.Dataset.from_tensor_slices((imgList, imgLabels)).batch(batch_size = batchSize)
     print('new dataset created. tasks complete! \n===========')
     return newTrainData #returns the new dataset


def train_model(train_data, val_data, model=None):
    pass

def fetch_classes():
    return pd.read_json('classes.json')

def run_model():
    pass
