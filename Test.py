import torch

import numpy as np
import argparse

from PointerNet import PointerNet
from Data_Generator import TSPDataset
from Data_Generator import sequence_generator
from Data_Generator import get_solution
from Data_Generator import get_cost

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

    # Data
    parser.add_argument('--train_size', default=2000, type=int, help='Training data size')
    parser.add_argument('--val_size', default=100, type=int, help='Validation data size')
    parser.add_argument('--test_size', default=100, type=int, help='Test data size')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    # Train
    parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # GPU
    parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
    # TSP
    parser.add_argument('--nof_points', type=int, default=10, help='Number of points in TSP')
    # Network
    parser.add_argument('--embedding_size', type=int, default=256, help='Embedding size')
    parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--nof_grus', type=int, default=3, help='Number of GRU layers')
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
                       params.nof_grus,
                       params.dropout,
                       params.bidir)

    print('Loading model...')
    model.load_state_dict(torch.load('./Parameters/parameterGRU.pkl'))
    print('Loaded finished!')

    test_num = 3
    test_point_num = 10
    point_array = np.random.rand(test_num, test_point_num, 2)
    nodes = np.random.rand(test_num, test_point_num, 2)
    weights = np.random.randint(1, 30, size=(test_num, test_point_num, 1), dtype=int)
    weights[:, 0, 0] = 0
    points_for_test = np.append(nodes, weights/30, axis=2)
    points = np.append(nodes, weights, axis=2)

    point_tensor = torch.tensor(points_for_test, dtype=torch.float)

    o, p = model(point_tensor)
    solutions = np.array(p)
    arr = np.array(list(range(1, test_point_num)))
    opt_list = []
    test_list = []
    random_list = []
    error_list = []

    sequences = []
    pre_sequences = sequence_generator()
    for sequence in pre_sequences:
        sequence = list(sequence)
        sequences.append(sequence)

    # show the result
    for i in range(test_num):
        point = points[i]
        depot = point[0, :2]
        depot = depot[np.newaxis, :]
        nodes = point[1:]
        solution = solutions[i]
        weights = points[:, 2]
        cost = get_cost(depot, nodes, solution)  # test solution

        solution_opt = np.pad(get_solution(depot, nodes, sequences), (1, 0))  # optimized solution
        cost_opt = get_cost(depot, nodes, solution_opt)
        
        solution_random = np.random.permutation(arr)
        solution_random = np.pad(solution_random, (1, 0))
        cost_random = get_cost(depot, nodes, solution_random)  # random solution

        error_opt = (cost - cost_opt) / cost_opt * 100
        error_list.append(error_opt)

        cost_list = [cost_random, cost_opt, cost]
        opt_list.append(cost_opt)
        test_list.append(cost)
        random_list.append(cost_random)

        print('Test{0}:'.format(i + 1))
        print(solution, 'cost is ', cost, '(Test solution)')
        print(solution_opt, 'cost is ', cost_opt, '(Optimized solution)')
        print(solution_random, 'cost is ', cost_random, '(Random solution)')
        print('The cost error is {0:.2f}%'.format(error_opt), '\n')

        plt.figure(i, (7, 7))

        plt.subplot(221)
        plt.title('Optimized solution')
        plt.scatter(point[:, 0], point[:, 1], s=weights)
        plt.plot(point[solution_opt][:, 0], point[solution_opt][:, 1], 'r')

        plt.subplot(222)
        plt.title('Test solution')
        plt.scatter(point[:, 0], point[:, 1], s=weights)
        plt.plot(point[solution][:, 0], point[solution][:, 1], 'b')

        plt.subplot(223)
        plt.title('Random solution')
        plt.scatter(point[:, 0], point[:, 1], s=weights)
        plt.plot(point[solution_random][:, 0], point[solution_random][:, 1], 'y')

        plt.subplot(224)
        plt.title('cost Comparison')
        plt.barh(range(3), cost_list, tick_label=['Rand', 'Opt', 'Test'], height=0.3)

    plt.figure(10, (7, 5))
    plt.title('Total result')
    x = range(1, test_num+1)
    plt.plot(x, opt_list, '.-', x, test_list, '.-', x, random_list, '.-')

    print(np.mean(error_list))
    plt.show()


if __name__ == '__main__':
    main()
