import unittest
import sys
import json
import os
sys.path.append('src')
from model_builder import construct_dataset
from tensorflow import keras

class ConstructDatasetTesting(unittest.TestCase):
    def test_get_data_non_valid_link(self):
        self.assertFalse(construct_dataset.get_data('apples',None,None))
    
    def test_get_data_valid_directory_training(self):
        directory = 'tests/testing_data/'
        self.assertIs()
    