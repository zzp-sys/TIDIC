import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
_tokenizer = _Tokenizer()


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = 768
        self.nclusters = nclusters
        self.nheads = nheads
        self.prompt_learner = None
        self.hidden_dim = 4096
        self.z_dim = 128
        self.cluster_head_i = nn.ModuleList([nn.Linear(self.hidden_dim, self.nclusters) for _ in range(self.nheads)])

        self.pre_feature = nn.Sequential(nn.Linear(self.backbone_dim, self.hidden_dim),
                                         nn.BatchNorm1d(self.hidden_dim),
                                         nn.ReLU(),
                                         )

        self.subspace = nn.Sequential(
            nn.Linear(self.hidden_dim, self.z_dim)
        )


    def forward(self, x, forward_pass='output_i'):
        if forward_pass == 'output_i':
            features = self.backbone.encode_image(x)
            features = features.float()
            pre_features = self.pre_feature(features)
            subspace_head = self.subspace(pre_features)
            out = [cluster_head_i(pre_features) for cluster_head_i in self.cluster_head_i]

            return subspace_head, out
        elif forward_pass == 'head_i':
            pre_features = self.pre_feature(x)
            subspace_head = self.subspace(pre_features)
            out = [cluster_head_i(pre_features) for cluster_head_i in self.cluster_head_i]

            return subspace_head, out
        elif forward_pass == 'backbone_i':
            out = self.backbone.encode_image(x)
            out = out.float()
        elif forward_pass == 'backbone_t':
            out = self.backbone.encode_text(x)
            out = out.float()
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out
