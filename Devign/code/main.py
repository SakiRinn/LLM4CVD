import argparse
import os
import os.path as osp
import pickle
import random
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import evaluate_metrics, train
from utils import tally_param, debug


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the test set.")
    args = parser.parse_args()

    set_seed(args.seed)

    model_dir = args.output_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if args.csv_path is None:
        args.csv_path = osp.join(args.output_dir, 'results.csv')

    if False and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_GGNNinput.json'),
                          valid_src=os.path.join(input_dir, 'valid_GGNNinput.json'),
                          test_src=os.path.join(input_dir, 'test_GGNNinput.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()

    if args.feature_size != dataset.feature_size:
        print('Warning!!! Dataset contains different feature vector than argument feature size.\n'
              f'Setting argument feature size to dataset feature size {dataset.feature_size}.', file=sys.stderr)
        args.feature_size = dataset.feature_size
        if args.feature_size > args.graph_embed_size:
            print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
                  f'Setting graph embedding size to argument feature size {args.feature_size}.', file=sys.stderr)
            args.graph_embed_size = args.feature_size

    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    if args.do_train:
        model = train(model=model, dataset=dataset, max_steps=1000000,
                      loss_function=loss_function, optimizer=optim, model_dir=model_dir,
                      dev_every=64, log_every=None, max_patience=100)
    if args.do_test:
        acc, pr, rc, f1, fpr = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                                dataset.get_next_test_batch, args.csv_path)
        debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f\t FPR: %0.2f'
              % (args.output_dir, acc, pr, rc, f1, fpr))
        debug('=' * 100)
