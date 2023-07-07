from keras.applications import VGG16
from keras import Sequential
from keras.layers import Flatten, Dense

def construct(dense_activation_0, dense_activation_1, optimizer, num_classes):
    '''
    Constructs a VGG16 CNN given a set of activation functions, an optimizer, and class size.
    dense_activation_0: str, representing an activation function
    dense_activation_1: str, representing an activation function
    num_classes: int; represents the number of classes available
    Return: keras.Sequential model instance
    '''

    VGG = VGG16(input_shape = (224, 224, 3), 
                include_top = False,
                weights = 'imagenet', 
                classes = num_classes)

    VGG.trainable = False #we dont want to train the first 13 layers, just the last 3.

    #model is flattened to feed into final dense layers
    #in final dense layer, the number of units must match how many possible classifications there are
    #softmax converts all results into a probability of it being that classification (val btw. 0-1)
    model = Sequential([VGG,                  
                        Flatten(),
                        Dense(units = 1024, activation = dense_activation_0),
                        Dense(units = 1024, activation = dense_activation_1),
                        Dense(units = num_classes, activation = 'softmax')])

    #compile the model
    #sparse_categorical_crossentropy: because we have a non-binary classification + want integer encoding, not 1-hot and we are encoding the predictions as integers
    model.compile(optimizer = optimizer,                       
                loss = 'sparse_categorical_crossentropy', 
                metrics = ['accuracy']) #uses MSE
    
    return model
