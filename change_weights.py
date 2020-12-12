import numpy as np
import torch

if __name__ == '__main__':
    dataset = np.load('./DataSets/TestSet8.npy', allow_pickle=True)

    DataSet = []
    for data in dataset:
        solution = data['Solution']
        nodes = data['Points'][:, :2]
        weights = data['Points'][:, 2]
        weights = weights[:, np.newaxis] / 30
        data = torch.cat([nodes, weights], dim=1)
        data_dict = {'Points': data, 'Solution': solution}
        DataSet.append(data_dict)
    np.save('./DataSets/TestSet8.npy', DataSet)

