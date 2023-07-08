import unittest
import sys
import json
import os
sys.path.append('src')
from model_builder import construct_model

class ConstructModelTesting(unittest.TestCase):

    def test_construct(self): #Verify correct via json print
        test_act_0 = 'relu'
        test_act_1 = 'selu'
        test_classes = 2
        test_optimizer = 'adamax'
        test_model = construct_model.construct(dense_activation_0=test_act_0,dense_activation_1=test_act_1,num_classes=test_classes,optimizer=test_optimizer)

        with open('tests/model_json_dump.json','w') as jf:
            jf.writelines(json.dumps(test_model.get_config(),indent=4))
    
    def test_extract_classes_file_DNE(self): #ensures that 0 is returned when no file is found
        self.assertFalse(construct_model.extract_classes())

    def test_extract_classes_file_exists(self):
        tmp_file = open('src/model_files/classes.json', 'x') #WARNING: remove classes.json from here
        tmp_file.write('{"beans":1}')
        tmp_file.close()
        is_dict = type(construct_model.extract_classes()) == type(dict())
        os.remove('src/model_files/classes.json')
        self.assertTrue(is_dict)



if __name__ == '__main__':
    unittest.main(verbosity=2)