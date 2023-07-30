import unittest
import sys
from datetime import datetime as dt
import logging
sys.path.append('src')
from model_builder import construct_dataset_new, construct_model
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt

class ConstructDatasetTesting(unittest.TestCase):
    def test_load_image(self):
        logging.info('test_load_image running...')
        fp = 'tests\\testing_data\\test_0.jpeg'
        logging.info('file path created.')
        img = construct_dataset_new.load_image(path=fp)
        logging.info('img loaded.')
        plt.imshow(img)
        plt.show()
        logging.info('test_load_image finished.')
        #Assuming the image is correctly grayscaled and normalized, this should appear clearly
    
    def test_prepare_datasets_invalid_batch_value_None(self):
        logging.info('test_prepare_datasets_invalid_batch_value_None called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        try:
            construct_dataset_new.prepare_datasets(test_ds_fp, test_ds_l, 0.3, None)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        logging.info('test_prepare_datasets_invalid_batch_value_None finished.')
    
    def test_prepare_datasets_invalid_batch_value_0(self):
        logging.info('test_prepare_datasets_invalid_batch_value_0 called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        try:
            construct_dataset_new.prepare_datasets(test_ds_fp, test_ds_l, 0.3, None)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        logging.info('test_prepare_datasets_invalid_batch_value_0 finished.')

    def test_prepare_datasets_invalid_no_val_split(self):
        logging.info('test_prepare_datasets_invalid_no_val_split called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.prepare_datasets(test_ds_fp, test_ds_l, None, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        logging.info('test_prepare_datasets_invalid_no_val_split finished.')
    
    def test_prepare_datasets_invalid_no_val_split_0(self):
        logging.info('test_prepare_datasets_invalid_no_val_split_0 called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.prepare_datasets(test_ds_fp, test_ds_l, 0, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        logging.info('test_prepare_datasets_invalid_no_val_split_0 finished.')
    
    def test_prepare_dataset_invalid_no_label(self):
        logging.info('test_prepare_datasets_invalid_no_label called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        
        try:
            construct_dataset_new.prepare_datasets(test_ds_fp, None, 0.5, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        logging.info('test_prepare_datasets_invalid_no_label finished.')
    
    def test_prepare_dataset_invalid_no_fpdataset(self):
        logging.info('test_prepare_datasets_invalid_no_label called.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_fp created.')
        
        try:
            construct_dataset_new.prepare_datasets(None, test_ds_l, 0.5, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        logging.info('test_prepare_datasets_invalid_no_label finished.')
    
    def test_prepare_dataset_valid_all(self):
        logging.info('test_prepare_dataset_valid_all called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.prepare_datasets(test_ds_fp, test_ds_l, 0.5, 32)
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        logging.info('test_prepare_dataset_valid_all finished.')
    

if __name__ == '__main__':
    timestamp = dt.now().strftime('%d-%m-%Y %H-%M-%S')
    logging.basicConfig(filename=f'tests\\test logs\{timestamp}.log', level=logging.DEBUG, encoding='utf-8', datefmt='%d-%m-%Y %H:%M:%S %p', format='%(asctime)s %(levelname)s | %(message)s')
    
    unittest.main(verbosity=1) 

    