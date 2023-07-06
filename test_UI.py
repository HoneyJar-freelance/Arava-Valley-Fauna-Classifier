import unittest
import UI

class testUIFunctions(unittest.TestCase):
    def test_get_img_dir(self):
        img_dir = UI.get_img_dir()
        self.assertTrue(img_dir is str or img_dir is None)

    def test_get_img_dir_is_string(self):
        csv_file = UI.get_csv_file()
        self.assertTrue(csv_file is str or csv_file is None)
    
    def test_get_img_dir_is_csv(self):
        csv_file = UI.get_csv_file()
        self.assertTrue('.csv' in csv_file or csv_file is None)
    
    def test_load_dependency_not_found_prompt_valid_type(self):
        tup = UI.load_dependency_not_found_prompt()
        self.assertTrue(tup is tuple or tup is None)

    def test_load_dependency_not_found_prompt_has_strings(self):
        tup = UI.load_dependency_not_found_prompt()
        self.assertTrue((tup[0] is str and tup[1] is str) or tup is None)
    
    def test_give_error(self):
        msg = 'bingus'
        self.assertEqual(msg, UI.give_error(msg))

