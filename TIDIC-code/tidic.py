import argparse
import os
import torch
import sys
from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_model,construct_semantic_space
from utils.evaluate_utils import get_predictions, hungarian_evaluate, kmeans, tidic_evaluate
from utils.train_utils import tidic_train
from utils.utils import Logger, get_features_eval
from datetime import datetime
from utils.utils import mkdir_if_missing

FLAGS = argparse.ArgumentParser(description='TIDIC Loss')
FLAGS.add_argument('--config_env', default='/home/ubuntu/DeepClustering/TIDIC/configs/env.yml', help='Location of path config file')
FLAGS.add_argument('--config_exp', default='/home/ubuntu/DeepClustering/TIDIC/configs/clustering/imagenet_dog.yml', help='Location of experiments config file')
FLAGS.add_argument('--gpu', type=str, default='1')
FLAGS.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
FLAGS.add_argument('--checkpoint', type=str, default='checkpoint.pth.tar') 

args = FLAGS.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    p = create_config(args.config_env, args.config_exp, args.topk, args.checkpoint)
    p['optimizer_image']['optimizer_kwargs']['lr'] = args.lr

    # Logger
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile_name = os.path.join(p['clustering_dir'], 'loggers', now+'.log')
    mkdir_if_missing(os.path.dirname(logfile_name))
    sys.stdout = Logger(filename=logfile_name, stream=sys.stdout)

    # Model
    model, preprocess = get_model(p)
    model = model.cuda()

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    train_dataset = get_train_dataset(p, preprocess, split='train', to_neighbors_dataset = True)
    val_dataset = get_val_dataset(p, preprocess, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

    # Get image centers
    if os.path.exists(os.path.join(p['clustering_dir'], 'kmeans_centers.pth')):
        image_centers, features = torch.load(os.path.join(p['clustering_dir'], 'kmeans_centers.pth'))

    else:
        dataloader = get_val_dataloader(p, val_dataset)
        features, targets = get_features_eval(dataloader, model)
        image_centers = kmeans(features, targets)
        image_centers = torch.from_numpy(image_centers).cuda()
        torch.save([image_centers, features], os.path.join(p['clustering_dir'], 'kmeans_centers.pth'))

    # Image optimizer
    if p['update_cluster_head_only']:
        for name, param in model.named_parameters():
            if 'cluster_head_i' in name :            # context vectors
                param.requires_grad = True
            else:  
                param.requires_grad = False
        head_i_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        image_optimizer = torch.optim.Adam(head_i_params, **p['optimizer_image']['optimizer_kwargs'])
        print('image_optimizer:', image_optimizer)

    # Construct semantic space
    print("Construct semantic space")
    text_dataloader = construct_semantic_space(p, image_centers, model, args)
    print("Construct over!")

    # Loss function
    from losses.losses import NNLoss, MaximalCodingRateReduction, Im_tex_LOSS
    criterion = NNLoss(args, p['num_classes'])
    criterion = criterion.cuda()
    # Max_coding Rate
    criterion_maxcoding_rate = MaximalCodingRateReduction(eps=0.1, gamma=1.0)
    criterion_maxcoding_rate = criterion_maxcoding_rate.cuda()
    # image-text infromation
    criterion_text = Im_tex_LOSS()
    criterion_text = criterion_text.cuda()

    # Checkpoint
    if  os.path.exists(p['clustering_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['clustering_checkpoint']), 'blue'))
        checkpoint = torch.load(p['clustering_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        image_optimizer.load_state_dict(checkpoint['image_optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_clustering_stats = None

    else:
        print(colored('No checkpoint file at {}'.format(p['clustering_checkpoint']), 'blue'))
        start_epoch = 0
        best_acc = 0
        best_clustering_stats = None

    # Computer image centers according to the confident samples
    from utils.compute_center import ComputeCenter
    cpt_center = ComputeCenter(num_cluster=p['num_classes'])

    early_stop_count = 0

    for epoch in range(start_epoch, p['epochs']):
        # Train
        print('Train ...')
        tidic_train(p, args, train_dataloader, text_dataloader, [image_centers, features],
                                                model, image_optimizer, criterion, criterion_maxcoding_rate, criterion_text,
                                                cpt_center, epoch+1, p['update_cluster_head_only'])

        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)
        clustering_stats = tidic_evaluate(predictions)
        lowest_loss_head = clustering_stats['lowest_loss_head']
        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=True, class_names=val_dataset.dataset.classes,
                                              confusion_matrix_file=os.path.join(p['clustering_dir'], 'confusion_matrix.png'))
        print('mlp_predict', clustering_stats)

        # Save the best clustering stats
        if clustering_stats['ACC'] > best_acc:
            print("best epoch",epoch)
            early_stop_count = 0
            best_acc = clustering_stats['ACC']
            best_clustering_stats = clustering_stats

        else:
            early_stop_count += 1
            if early_stop_count >= 10:
                break

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'image_optimizer': image_optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1,  'best_acc': best_acc},
                     p['clustering_checkpoint'])

    print('best_clustering_stats:')
    print(best_clustering_stats)

if __name__ == "__main__":
    main()

