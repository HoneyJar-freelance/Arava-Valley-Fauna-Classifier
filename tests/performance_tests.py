import sys
import os
import numpy as np
import performance_analysis.recorder as rd

def test_master():
    checks = [create_table_test_create_file(),
              create_table_test_header_added(),]
    
    print(f'{np.count_nonzero(checks)}/{len(checks)} tests passed.')
    
    return checks

def create_table_test_create_file():
    file = 'test'
    header=''
    recorder.create_table(file,header)
    if(not os.path.exists(f'../performance_analysis/tables/test.csv')):
        print('create_table_test_create_file() failed.')
        return False
    else:
        os.remove('../performance_analysis/tables/test.csv')
        return True

def create_table_test_header_added():
    file = 'test'
    header = 'beans'
    recorder.create_table(file,header)
    with open('../performance_analysis/tables/test.csv') as tester:
        isbeans = tester.readline()
        if(header != isbeans):
            print('create_table_test_header_added() failed.')
        
        os.remove('../performance_analysis/tables/test.csv')
        return header == isbeans
