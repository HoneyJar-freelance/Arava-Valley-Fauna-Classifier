import csv
import tensorflow
import keras
import os

def create_table(name:str, header:str):
    '''
    Creates a csv file that records the performance of a model. The name should be tied to the architecture.

    name: name of the file (without extension)
    header: column titles for the csv
    '''
    with open(file=f'../tables/{name}.csv') as file:
        file.write(header)

def format_data(data:tuple|list):
    '''
    Takes in an iterable of information produced by the model, and returns a formatted string of data for a csv.

    data: iterable of data from a ML model
    returns: string of data formatted for csv
    '''
    pass

def record_data(file, data):
    if(not os.path.exists(f'../tables/{file}.csv')):
        create_table(name=file, header='epochs,batch_size,early_stop_at,train_acc,train_loss,val_acc,val_loss,figure_link')

    with open(f'../tables/{file}.csv', mode='a') as table:
        table.write(data)