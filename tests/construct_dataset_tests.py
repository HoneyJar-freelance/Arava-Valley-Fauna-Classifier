import unittest
import sys
import json
import os
sys.path.append('src')
from model_builder import construct_dataset, construct_model
from tensorflow import keras
import tensorflow as tf

class ConstructDatasetTesting(unittest.TestCase):
    def test_get_data_non_valid_link(self):
        self.assertFalse(construct_dataset.get_data('apples',None,None,None))

    def test_preprocess_empty(self):
        self.assertFalse(construct_dataset.preprocess(None))
    
    def test_preprocess_not_dataset(self):
        self.assertFalse(construct_dataset.preprocess(1))
    
    def test_preprocess_valid_dataset(self):
        ds = keras.utils.image_dataset_from_directory('tests\\testing_data', labels=None, batch_size=8, color_mode='grayscale')
        self.assertTrue(construct_dataset.preprocess(ds))
    
    def test_preprocess_valid_dataset_verify_color_change(self):
        ds = keras.utils.image_dataset_from_directory('tests\\testing_data', labels=None, batch_size=8, color_mode='grayscale', image_size=(224,224))
        self.assertEqual((None, 224, 224, 3), tf.shape(ds))
    
    def test_preprocess_both_subsets(self):
        ds = keras.utils.image_dataset_from_directory('tests\\testing_data', labels=None, batch_size=8, seed=19121954, validation_split=0.3, color_mode='grayscale', subset='both')
        self.assertTrue(construct_dataset.preprocess(ds))

    def test_get_data_valid_info(self):
        ds = construct_dataset.get_data(link='tests\\testing_data', batch_size=32,val_split=None,csvfile='tests\\testing_data\\testing_labels.csv')
        self.assertTrue(ds)
    
    def test_get_data_batch_none(self):
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)

    