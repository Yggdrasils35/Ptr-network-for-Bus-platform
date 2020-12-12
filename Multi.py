"""
Use multi processing to generate training data.
"""

import multiprocessing
import Data_Generator
import numpy as np


def create_one_group(i):
    data = Data_Generator.TSPDataset(500, 10)
    np.save('./DataSets/trainset' + str(i+1) + '.npy', data)
    print('trainset ' + str(i+1) + ' is done!')


if __name__ == '__main__':
    for i in range(4):
        p = multiprocessing.Process(target=create_one_group, args=(i,))
        p.start()
