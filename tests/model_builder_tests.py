import unittest
import sys
import json
import logging
from datetime import datetime as dt
import os
sys.path.append('src')
from model_builder import construct_model

class ConstructModelTesting(unittest.TestCase):

    def test_construct(self): #Verify correct via json print
        logging.info("test_construct called.")
        test_act_0 = 'relu'
        test_act_1 = 'selu'
        test_classes = 2
        test_optimizer = 'adamax'
        test_model = construct_model.construct(dense_activation_0=test_act_0,dense_activation_1=test_act_1,num_classes=test_classes,optimizer=test_optimizer)
        try:
            with open('tests/model_json_dump.json','w') as jf:
                jf.writelines(json.dumps(test_model.get_config(),indent=4))
            logging.info("test_construct passed.")
        except:
            logging.info("test_construct failed.")
            self.assertTrue(False)
    
    def test_get_labels(self): #ensures a 1d list of animal labels is extracted
        logging.info("test_get_labels called.")
        answers = ['apple','nugget','matzo','potato','apple']
        self.assertListEqual(answers, construct_model.get_labels('tests/test_csv.csv'))
    
    def test_extract_classes_file_DNE(self): #ensures that 0 is returned when no file is found
        logging.info("test_extract_classes_file_DNE")
        self.assertFalse(construct_model.extract_classes('tests/test.json'))

    def test_extract_classes_file_exists(self):
        logging.info("test_extract_classes_file_exists called.")
        tmp_file = open('tests/test.json', 'x')
        tmp_file.write('{"beans":1}')
        tmp_file.close()
        is_dict = type(construct_model.extract_classes('tests/test.json')) == type(dict())
        os.remove('tests/test.json')
        self.assertTrue(is_dict)
    
    def test_extract_classes_file_gets_correct_dict(self):
        logging.info("test_extract_classes_file_gets_correct_dict called.")
        tmp_file = open('tests/test.json', 'x')
        tmp_file.write('{"beans":1}')
        tmp_file.close()
        is_correct_dict = construct_model.extract_classes('tests/test.json')['beans'] == 1
        os.remove('tests/test.json')
        self.assertTrue(is_correct_dict)
    
    def test_get_new_classes_no_prexisting(self):
        logging.info("test_get_new_classes_no_prexisting called.")
        new_classes = construct_model.get_new_classes('tests/test_csv.csv')
        correct_dict = {
            'apple':0,
            'nugget':1,
            'matzo':2,
            'potato':3
        }
        self.assertDictEqual(correct_dict, new_classes)

    def test_get_new_classes_add_classes(self):
        logging.info("test_get_new_classes_add_classes called.")
        test_classes = {'beans':0}
        new_classes = construct_model.get_new_classes('tests/test_csv.csv', test_classes)
        correct_dict = {
            'beans':0,
            'apple':1,
            'nugget':2,
            'matzo':3,
            'potato':4
        }
        self.assertDictEqual(correct_dict, new_classes)

    def test_save_classes_old_no_exist(self):
        logging.info("test_save_classes_old_no_exist called.")
        test_dict = {'beans':0}
        construct_model.save_classes(test_dict, 'tests/test.json')
        does_exist = os.path.exists('tests/test.json.OLD')
        try:
            os.remove('tests/test.json')
        except:
            pass
        self.assertFalse(does_exist)
    
    def test_save_classes_old_exist(self):
        logging.info("test_save_classes_old_exist called.")
        test_dict = {'beans':0}
        construct_model.save_classes(test_dict, 'tests/test.json')
        test_dict_2 = {'beans':0, 'apples':1}
        construct_model.save_classes(test_dict_2, 'tests/test.json')
        does_exist = os.path.exists('tests/test.json.OLD')
        does_exist_2 = os.path.exists('tests/test.json')
        try:
            os.remove('tests/test.json')
            os.remove('tests/test.json.OLD')
        except:
            pass
        self.assertTrue(does_exist and does_exist_2, f'old:{does_exist}     new:{does_exist_2}')



if __name__ == '__main__':
    timestamp = dt.now().strftime('%d-%m-%Y %H-%M-%S')
    logging.basicConfig(filename=f'tests\\test logs\{timestamp}.log', level=logging.DEBUG, encoding='utf-8', datefmt='%d-%m-%Y %H:%M:%S %p', format='%(asctime)s %(levelname)s | %(message)s')
    unittest.main(verbosity=2)