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
    
    def test_prep_dataset_train(self):
        logging.info('test_prep_dataset_train called...')
        fp_ds = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug(f'fp_ds created: {fp_ds}')

        label_ds = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug(f'label_ds created: {label_ds}')

        ds = None
        try:
            ds = construct_dataset_new.prep_dataset(fp_ds, label_ds,{'0':0, '1':1})
            
        except Exception as Argument:
            logging.exception('ERROR: Test encountered error. What went wrong: ')
            self.assertTrue(False)
        
        correct_length = len(ds) == 2

        for i in iter(ds[0]):
                plt.imshow(i)
                plt.show()
        for i in iter(ds[1]):
            print(i)
        self.assertTrue((ds and correct_length), f'ds: {ds}  correct length: {correct_length}')

    def test_prep_dataset_predict(self):
        logging.info('test_prep_dataset_predict called...')
        fp_ds = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug(f'fp_ds created: {fp_ds}')

        ds = None
        try:
            ds = construct_dataset_new.prep_dataset(fp_ds, None,{'0':0, '1':1})
            
        except Exception as Argument:
            logging.exception('ERROR: Test encountered error. What went wrong: ')
            self.assertTrue(False)
        
        for i in iter(ds):
            plt.imshow(i)
            plt.show()

    def test_associate_labels_with_data_no_path_ds(self):
        logging.info('test_associate_labels_with_data_no_path_ds called.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        try:
            construct_dataset_new.associate_labels_with_data(None, test_ds_l, 0.3, None)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
    def test_associate_labels_with_data_invalid_batch_value_None(self):
        logging.info('test_associate_labels_with_data_invalid_batch_value_None called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 0.3, None)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_associate_labels_with_data_invalid_batch_value_neg(self):
        logging.info('test_associate_labels_with_data_invalid_batch_value_neg called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 0.3, -5)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
    def test_associate_labels_with_data_invalid_batch_value_float(self):
        logging.info('test_associate_labels_with_data_invalid_batch_value_float called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 0.3, 0.5)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
    def test_associate_labels_with_data_invalid_batch_value_0(self):
        logging.info('test_associate_labels_with_data_invalid_batch_value_0 called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 0.3, None)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_associate_labels_with_data_invalid_no_val_split(self):
        logging.info('test_associate_labels_with_data_invalid_no_val_split called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, None, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
    def test_associate_labels_with_data_invalid_no_val_split_0(self):
        logging.info('test_associate_labels_with_data_invalid_no_val_split_0 called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 0, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
    def test_associate_labels_with_data_invalid_no_label(self):
        logging.info('test_associate_labels_with_data_invalid_no_label called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, None, 0.5, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
    
    def test_associate_labels_with_data_length_mismatch(self):
        logging.info('test_associate_labels_with_data_length_mismatch called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
    
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 0.5, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_associate_labels_with_data_all_valid_no_split(self):
        logging.info('test_associate_labels_with_data_valid_single called.')
        ds = None
        try:
            ds = construct_dataset_new.associate_labels_with_data(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'],
                                                                   [0,1], 
                                                                   0.5, 
                                                                   32,
                                                                   {'0':0, '1':1})
        except Exception as Argument:
            logging.exception("Test failed. What went wrong: ")
            self.assertTrue(False, "Dataset failed to be constructed")
        
        logging.debug(f'bruh: {list(ds)}')
        self.assertTrue(len(list(ds)) == 2)
    
    def test_associate_labels_with_data_all_valid_split(self):
        logging.info('test_associate_labels_with_data_valid_all called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            ds = construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 0.4, 32)
            self.assertTrue(len(list(ds)) == 2)
        except:
            self.assertTrue(False)
    
    def test_associate_labels_with_data_neg_val_split(self):
        logging.info('test_associate_labels_with_data_neg_val_split called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, -0.5, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
    
    def test_associate_labels_with_data_val_split_1(self):
        logging.info('test_associate_labels_with_data_val_split_1 called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 1, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
    
    def test_associate_labels_with_data_val_split_large(self):
        logging.info('test_associate_labels_with_data_neg_val_split called.')
        test_ds_fp = tf.data.Dataset.from_tensor_slices(['tests/testing_data/test_0.jpeg', 'tests/testing_data/test_1.JPEG'])
        logging.debug('test_ds_fp created.')
        test_ds_l = tf.data.Dataset.from_tensor_slices([0,1])
        logging.debug('test_ds_l created.')
        
        try:
            construct_dataset_new.associate_labels_with_data(test_ds_fp, test_ds_l, 50, 32)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

if __name__ == '__main__':
    timestamp = dt.now().strftime('%d-%m-%Y %H-%M-%S')
    logging.basicConfig(filename=f'tests\\test logs\{timestamp}.log', level=logging.DEBUG, encoding='utf-8', datefmt='%d-%m-%Y %H:%M:%S %p', format='%(asctime)s %(levelname)s | %(message)s')
    
    suite = unittest.TestSuite()
    suite.addTest(ConstructDatasetTesting("test_associate_labels_with_data_all_valid_no_split"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    #unittest.main(verbosity=1) 

    