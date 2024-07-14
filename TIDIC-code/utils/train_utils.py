import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
from utils.common_config import run_scheduler

def tidic_train(p, args, train_loader, text_loader, image_list,
                 model, image_optimizer, criterion, criterion_maxcoding_rate, criterion_text,
                 cpt_center, epoch, update_cluster_head_only=False):

    mcr2_losses = AverageMeter('MaxCodingRateduce loss', ':.4e')
    nn_consistency_losses = AverageMeter('image nearest neighbor consistency loss', ':.4e')
    it_consistency_losses = AverageMeter('image-semantic consistency loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [mcr2_losses, nn_consistency_losses, it_consistency_losses],
                             prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN

    image_centers, image_features = image_list
    image_centers_v1 = cpt_center.get_centers(image_features, model, args)
    text_centers  = cpt_center.search_sim_texts(args, image_centers_v1, text_loader, model)


    from losses.G_softmax import Gumble_Softmax
    G_softmax = Gumble_Softmax(tau=1)


    for i, batch in enumerate(train_loader):
        # Forward pass
        indices = batch['index'].cuda(non_blocking=True)
        n_indices = batch['n_index'].cuda(non_blocking=True)
        anchor_features = image_features[indices].cuda(non_blocking=True)
        neighbor_features = image_features[n_indices].cuda(non_blocking=True)

        # Network output
        if update_cluster_head_only:  # Only calculate gradient for backprop of cluster head
            anchor_subspace, anchor_outputs = model(anchor_features, forward_pass='head_i')
            neighbor_subspace, neighbor_outputs = model(neighbor_features, forward_pass='head_i')

        anchor_prob = G_softmax(anchor_outputs[0])
        loss_mcr2 = criterion_maxcoding_rate(anchor_subspace, anchor_prob, num_classes=10)

        mcr2_losses.update(np.mean(loss_mcr2))

        # Loss for every head_i
        nn_consistency_loss, it_consistency_loss = [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchor_outputs, neighbor_outputs):
            nn_consistency_loss_ = criterion(anchors_output_subhead,neighbors_output_subhead)
            it_consistency_loss_ = criterion_text(anchors_output_subhead, anchor_features, text_centers)

            nn_consistency_loss.append(nn_consistency_loss_)
            it_consistency_loss.append(it_consistency_loss_)

        nn_consistency_losses.update(np.mean([v.item() for v in nn_consistency_loss]))
        it_consistency_losses.update(np.mean([v.item() for v in it_consistency_loss]))

        nn_consistency_loss = torch.sum(torch.stack(nn_consistency_loss, dim=0))
        it_consistency_loss = torch.sum(torch.stack(it_consistency_loss, dim=0))

        loss = loss_mcr2 + 5.0 * nn_consistency_loss + 1.0 * it_consistency_loss
        image_optimizer.zero_grad()
        loss.backward()
        image_optimizer.step()

        run_scheduler(p, epoch, image_optimizer, len(train_loader), i)

        if i % 40 == 0:
            progress.display(i)
