import argparse
import os
import torch
from utils.config import create_config
from utils.common_config import get_train_dataset, get_val_dataset, get_val_dataloader, get_model
import numpy as np
import ssl
from utils.utils import get_knn_indices

FLAGS = argparse.ArgumentParser(description='get knn indices.')
FLAGS.add_argument('--config_env', default='/home/ubuntu/DeepClustering/TIDIC/configs/env.yml', help='Location of path config file')
FLAGS.add_argument('--config_exp', default='/home/ubuntu/DeepClustering/TIDIC/configs/clustering/cifar10.yml', help='Location of experiments config file')
FLAGS.add_argument('--gpu', type=str, default='0,1')
FLAGS.add_argument('--topk', type=int, default=20)

args = FLAGS.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    p = create_config(args.config_env, args.config_exp, args.topk)
    print('dataset_name: ', p['train_db_name'])

    # Model
    model, preprocess = get_model(p)
    model = model.cuda()

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    train_dataset = get_train_dataset(p, preprocess, split='train', to_neighbors_dataset = False)
    train_dataloader = get_val_dataloader(p, train_dataset)
    val_dataset = get_val_dataset(p, preprocess)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d ' %(len(train_dataset)))
    print('Val samples %d ' %(len(val_dataset)))

    train_indices, train_accuracy= get_knn_indices(model, train_dataloader, args.topk)
    val_indices, val_accuracy = get_knn_indices(model, val_dataloader, args.topk)

    print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (p['num_neighbors'], 100 * train_accuracy))
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' % (5, 100 * val_accuracy))
    np.save(p['top{}_neighbors_train_path'.format(p['num_neighbors'])], train_indices)
    np.save(p['topk_neighbors_val_path'], val_indices)

if __name__ == "__main__":
    main()

