import torch
import torch.nn as nn
class Gumble_Softmax(nn.Module):
    def __init__(self,tau, straight_through=False):
        super().__init__()
        self.tau = tau
        self.straight_through = straight_through
    
    def forward(self,logits):
        logps = torch.log_softmax(logits,dim=1)
        gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
        logits = logps + gumble
        out = (logits/self.tau).softmax(dim=1)
        if not self.straight_through:
            return out
        else:
            out_binary = (logits*1e8).softmax(dim=1).detach()
            out_diff = (out_binary - out).detach()
            return out_diff + out