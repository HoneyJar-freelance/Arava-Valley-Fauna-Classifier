import unittest
import sys
from datetime import datetime as dt
import logging
sys.path.append('src')
from model_builder import construct_dataset_new, construct_model
from tensorflow import keras
import keras
import tensorflow as tf

class ConstructDatasetTesting(unittest.TestCase):
    def test_load_image(self):
        fp = "C:\Users\jacob\Documents\GitHub\Arava-Valley-Fauna-Classifier\\tests\\testing_data\\test_0.jpeg"
        img = construct_dataset_new



    

if __name__ == '__main__':
    timestamp = dt.now().strftime('%d-%m-%Y %H-%M-%S')
    logging.basicConfig(filename=f'tests\\test logs\{timestamp}.log', level=logging.DEBUG, encoding='utf-8', datefmt='%d-%m-%Y %H:%M:%S %p', format='%(asctime)s %(levelname)s | %(message)s')
    
    unittest.main(verbosity=1) 

    