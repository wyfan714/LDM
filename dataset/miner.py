import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def l2_norm(x):
    if len(x.shape):# x.shape为1才会为true
        x = x.reshape((x.shape[0],-1))
    return F.normalize(x, p=2, dim=1)


def get_distance(x):
    _x = x.detach()
    sim = torch.matmul(_x, _x.t())
    dist = 2 - 2*sim
    dist += torch.eye(dist.shape[0]).to(dist.device)   # maybe dist += torch.eye(dist.shape[0]).to(dist.device)*1e-8
    dist = dist.sqrt()
    return dist





class   DistanceWeightedSampling(nn.Module):
    '''
    parameters
    ----------
    batch_k: int
        number of images per class

    Inputs:
        data: input tensor with shapeee (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx

    '''

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =True,  **kwargs):
        super(DistanceWeightedSampling,self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize

    def forward(self, x):
        k = self.batch_k
        n, d = x.shape
        distance = get_distance(x)
        distance = distance.clamp(min=self.cutoff)
        log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))

        if self.normalize:
            log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)

            weights = torch.exp(log_weights - torch.max(log_weights))

        if x.device != weights.device:
            weights = weights.to(x.device)

        mask = torch.ones_like(weights)
        for i in range(0,n,k):
            mask[i:i+k, i:i+k] = 0

        mask_uniform_probs = mask.double() *(1.0/(n-k))

        weights = weights*mask*((distance < self.nonzero_loss_cutoff).float()) + 1e-8
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / weights_sum

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.cpu().numpy()
        for i in range(n):
            block_idx = i // k

            if weights_sum[i] != 0:
                n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
            else:
                n_indices +=  np.random.choice(n, k-1, p=mask_uniform_probs[i]).tolist()
            for j in range(block_idx * k, (block_idx + 1)*k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return  a_indices, p_indices, n_indices




