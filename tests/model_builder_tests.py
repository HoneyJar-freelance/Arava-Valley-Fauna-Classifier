import unittest
from src.model_builder import construct_model

class ConstructModelTesting(unittest.TestCase):

    def test_construct(self):
        test_act_0 = 'relu'
        test_act_1 = 'selu'
        test_classes = 2
        test_optimizer = 'adamax'
        test_model = construct_model.construct(dense_activation_0=test_act_0,dense_activation_1=test_act_1,num_classes=test_classes,optimizer=test_optimizer)

        