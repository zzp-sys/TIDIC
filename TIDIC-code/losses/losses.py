import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

def entropy(x, input_as_probabilities):

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))

class NNLoss(nn.Module):
    def __init__(self, args):
        super(NNLoss, self).__init__()

    def forward(self, image_output, image_nb_output):

        # Softmax
        b, c = image_nb_output.size()
        image_prob = torch.softmax(image_output, dim=-1)
        image_nb_prob = torch.softmax(image_nb_output, dim=-1)

        similarity = torch.bmm(image_prob.view(b, 1, c), image_nb_prob.view(b, c, 1)).squeeze()

        # L_sim: Image nearest neighbor samples consistency learning loss
        nn_consistency_loss = (-torch.sum(torch.log(similarity), dim=0) / b) + 5.0 * entropy(torch.mean(image_prob, 0), input_as_probabilities=True)

        return nn_consistency_loss

class Im_tex_LOSS(nn.Module):

    def __init__(self):
        super(NNLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, image_output, image_feature, text_center):
        text_center = text_center.cuda()
        text_prob = torch.mm(image_feature, text_center.T).softmax(dim=-1)
        _, text_information = torch.max(text_prob, dim=1)

        # L_cl: Image and text consistency learning loss
        it_consistency_loss = self.CE(image_output, text_information)

        return it_consistency_loss

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)

        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        # This function support Y as label integer or membership probablity.
        if len(Y.shape) == 1:
            # if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label, 0, indx] = 1
        else:
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)

        mcr2_loss = - discrimn_loss + self.gamma * compress_loss

        return mcr2_loss
