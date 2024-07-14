import os
import math
import numpy as np
import torch
from utils.collate import collate_custom
import clip
from torch.utils import data
from utils.semantic_filters import image_centers_filter, uniqueness_filter
from utils.utils import get_wordnet_noun


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'ViT-L/14':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone, preprocess = clip.load("/home/ubuntu/DeepClustering/TIDIC/models/clip/ViT-L-14.pt", device=device)

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    from models.models import ClusteringModel
    model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])

    return model, preprocess

def get_train_dataset(p, transform, to_augmented_dataset=False,
                        to_neighbors_dataset=False, split=None):
    from data.cifar import CIFAR10
    dataset = CIFAR10(train=True, transform=transform, download=True)

    if to_augmented_dataset:
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset:
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['top{}_neighbors_train_path'.format(p['num_neighbors'])])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])

    return dataset


def get_val_dataset(p, transform=None, to_neighbors_dataset=False):

    from data.cifar import CIFAR10
    dataset = CIFAR10(train=False, transform=transform, download=True)

    if to_neighbors_dataset:
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5)

    return dataset

def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=True)

def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)

def construct_semantic_space(p, image_centers, model, args):

    # Get wordnet noun set
    filename = os.path.join(os.getcwd(), '/home/ubuntu/DeepClustering/TIDIC/data/noun.csv')
    nouns = get_wordnet_noun(filename)
    nouns_num = len(nouns)

    # Semantic dataset \mathcal{T}
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in nouns])

    # Dataloader
    text_targets = torch.zeros(len(text_inputs))
    text_indices = torch.arange(len(text_inputs))
    text_dataset = data.TensorDataset(text_inputs, text_targets, text_indices)
    text_dataset.filename = 'text'
    text_dataloader = get_val_dataloader(p, text_dataset)

    # Construct semantic space
    target_list1 = image_centers_filter(model, text_dataloader, image_centers) # image centers
    target_list2 = uniqueness_filter(model, text_dataloader)   # uniqueness
    target_list = torch.from_numpy(np.intersect1d(target_list1.cpu().numpy(), target_list2.cpu().numpy())).cuda()

    text_class_ids = torch.arange(len(target_list))      # ids
    text_indices = torch.arange(nouns_num)[target_list]  # indices
    text_dataset = data.TensorDataset(text_inputs[target_list], text_class_ids, text_indices)
    text_dataset.filename = 'text'
    text_dataloader = get_val_dataloader(p, text_dataset)

    return text_dataloader

def get_optimizer(p, model, cluster_head_only=False, prompt_only=False):
    if cluster_head_only:  # Only weights in the cluster head will be updated
        for name, param in model.named_parameters():
            if 'head_i' in name :            # context vectors
                param.requires_grad = True
            else:
                param.requires_grad = False

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert (len(params) == 2 * p['num_heads'])
    elif prompt_only:
        for name, param in model.named_parameters():
            if 'ctx' in name :            # context vectors
                param.requires_grad = True
            else:
                param.requires_grad = False

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        for name, param in model.named_parameters():
            if 'head_i' in name or 'ctx' in name :            # context vectors
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # params = model.parameters()

    if p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer

def run_scheduler(p, epoch, image_optimizer, steps, e_step):

    lr = p['optimizer_image']['optimizer_kwargs']['lr']
    # print("learning rate is ",lr)
    if p['optimizer_image']['scheduler'] == 'constant':
        lr = lr
    elif p['optimizer_image']['scheduler'] == 'cosine':   # the type of cosine may need to be changed
        eta_min = lr * (p['lr_decay_rate'] ** 3)
        e_steps = (epoch - 1) * steps + e_step
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * e_steps / (p['epochs'] * steps))) / 2

    for param_group in image_optimizer.param_groups:
        param_group['lr'] = lr
