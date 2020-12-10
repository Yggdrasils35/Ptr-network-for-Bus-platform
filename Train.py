"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from PointerNet import PointerNet
from Data_Generator import TSPDataset

import matplotlib.pyplot as plt
import itertools
import math

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

test_flag = True

def get_solution(points):
    """
    Dynamic programing solution for TSP - O(2^n*n^2)
    https://gist.github.com/mlalevic/6222750

    :param points: List of (x, y) points
    :return: Optimal solution
    """

    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S-{0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S-{j}, k)][0] + all_distances[k][j], A[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B

    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[1]+[0]), res[0]


def main():
    parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

    # Data
    parser.add_argument('--train_size', default=2000, type=int, help='Training data size')
    parser.add_argument('--val_size', default=100, type=int, help='Validation data size')
    parser.add_argument('--test_size', default=100, type=int, help='Test data size')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    # Train
    parser.add_argument('--nof_epoch', default=50, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # GPU
    parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
    # TSP
    parser.add_argument('--nof_points', type=int, default=10, help='Number of points in TSP')
    # Network
    parser.add_argument('--embedding_size', type=int, default=256, help='Embedding size')
    parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--nof_lstms', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout value')
    parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

    params = parser.parse_args()

    if params.gpu and torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    model = PointerNet(params.embedding_size,
                       params.hiddens,
                       params.nof_lstms,
                       params.dropout,
                       params.bidir)

    print('Loading model...')
    model.load_state_dict(torch.load('parameter.pkl'))
    print('Loaded finished!')

    dataset = np.load('TrainSet.npy', allow_pickle=True)
    # dataset: list of points and solutions(sequences)

    dataloader = DataLoader(dataset,
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=0)

    if USE_CUDA:
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    CCE = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=params.lr)

    losses = []
    for epoch in range(params.nof_epoch):

        batch_loss = []
        iterator = tqdm(dataloader, unit='Batch')
        for i_batch, sample_batched in enumerate(iterator):
            # solution.append(sample_batched['Solution'])
            iterator.set_description('Epoch %i/%i' % (epoch + 1, params.nof_epoch))

            train_batch = Variable(sample_batched['Points'])
            target_batch = Variable(sample_batched['Solution'])

            if USE_CUDA:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()

            o, p = model(train_batch)

            o = o.contiguous().view(-1, o.size()[-1])

            target_batch = target_batch.view(-1)

            loss = CCE(o, target_batch)
            losses.append(loss.data)
            batch_loss.append(loss.data)

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            iterator.set_postfix(loss='{}'.format(loss.data))

        iterator.set_postfix(loss=torch.mean(torch.stack(batch_loss)))

    torch.save(model.state_dict(), 'parameter.pkl')


if __name__ == '__main__':
    main()
