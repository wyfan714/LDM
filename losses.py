import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import numpy as np
from pytorch_metric_learning import miners, losses, distances
from pytorch_metric_learning.losses import SoftTripleLoss
from dataset import miner


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class AMLoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alphap = 32, alphan = 32, delta = 0.1, lam = 1.0):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        """
        nb_classes:100
        sz_embed:512
    
        mrg :阈值
        .cuda()：把tensor放到GPU上
        """
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrgn = mrg
        self.alphap = alphap
        self.alphan = alphan
        self.mrg_list = mrg
        self.delta = delta
        self.lam = lam
        self.initialize_mrg(self.mrg_list, nb_classes)
        
    def initialize_mrg(self, mrg, nb_classes):
        self.mrg_list = torch.tensor([float(mrg)])
        self.mrg_list = torch.ones(nb_classes) * mrg
        self.mrg_list = torch.nn.Parameter(self.mrg_list)
       

    def forward(self, X, T):
        input_l = F.normalize(X, p=2, dim=1)
        proxy_l = F.normalize(self.proxies, p=2, dim=1)
        mrg_list = self.mrg_list
        cos = F.linear(input_l, proxy_l)  # Calcluate cosine similarity       
        cos_proxies = F.linear(proxy_l, proxy_l)  # Calculate proxy cosine similarity
        #cos = F.linear(l2_norm(X), l2_norm(P))  # Calculate cosine similarity
        #cos_proxies = F.linear(l2_norm(P), l2_norm(P))  # Calculate proxy cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)       
        N_one_hot = 1 - P_one_hot  
        with_neg_proxies = torch.nonzero(P_one_hot.sum(dim=0) == 0).squeeze(dim=1)  
        #self.mrg_list[with_neg_proxies].detach()
        norm_mrg_list = self.mrg_list.reshape(1, -1)
        #norm_mrg_list[with_neg_proxies].detach()
        pos_exp = torch.exp(-self.alphap  *  (cos - norm_mrg_list))       
        neg_exp = torch.exp(self.alphan  * (cos + self.mrgn))
        #neg_exp = torch.exp(32 * (cos - norm_mrg_list + 0.05))
       # print(norm_mrg_list)
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies
        cos_one_hot = torch.eye(self.nb_classes).cuda()
        cos_one_hot = 1 - cos_one_hot
        
        #proxy_sim = cos_proxies - norm_mrg_list.reshape(-1,1) + self.delta
       # proxy_sim = proxy_sim * cos_one_hot
        #proxy_sf = F.softmax(self.T * proxy_sim, dim = 1)
        
        sim_proxy_sum = torch.exp( (cos_proxies - mrg_list.reshape(-1,1) + self.delta))
        #sim_proxy_sum =  proxy_sf 
        #exp_cos = torch.exp(cos_proxies)
        #exp_cos = exp_cos * proxy_sf
        #sim_proxy_sum = exp_cos    
        sim_proxy_sum = (sim_proxy_sum * cos_one_hot).sum(dim=1)
        #print(sim_proxy_sum)
        #sim_proxy_sum = sim_proxy_sum[with_pos_proxies]
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies     
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes      
        #sim_proxy_term = torch.log(1 + sim_proxy_sum).sum() / num_valid_proxies

        sim_proxy_term = torch.log(1 + sim_proxy_sum).sum() / self.nb_classes 
        #sim_proxy_term = sim_proxy_sum.sum() / self.nb_classes
        #sim_proxy_term = sim_proxy_sum.sum() / num_valid_proxies
        loss = pos_term + neg_term   
        #print(loss)
        loss = loss + self.lam * sim_proxy_term  
        return loss

class sml(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed):
        torch.nn.Module.__init__(self)
        self.n_classes = nb_classes
        mrg = 0.1
        self.mu = torch.ones(nb_classes) * mrg
        self.nv = torch.ones(nb_classes) * mrg
        self.mu = torch.nn.Parameter(self.mu)
        self.nv = torch.nn.Parameter(self.nv)
        self.lam = 0.1
    def forward(self, X, labels):
        D = F.linear(X, X)
        batch_size = X.size(0)
        loss = list()
        for i in range(batch_size):
            #pos_pair = D[i][labels == labels[i]]
            #neg_pair = D[i][labels != labels[i]]
            pos_index = (labels == labels[i]).nonzero()
            neg_index = (labels != labels[i]).nonzero()
            for J in range(pos_index.size(0)):
                j = pos_index[J]
                if j == i:
                    continue
                for K in range(neg_index.size(0)):
                    k = neg_index[K]
                    loss1 = D[i][j] - D[i][k] + self.mu[labels[i]]
                   # loss2 = D[i][j] - D[j][k] + self.nv[labels[i]]
                    if loss1 > 0:
                        loss.append(loss1)
                    #if loss2 > 0:
                    #    loss.append(self.lam * loss2)
        loss1 = torch.sum(self.mu)
        loss2 = torch.sum(self.nv)
        loss.append(-1 * loss1 / self.n_classes)
        loss.append(-1 * loss2 / self.n_classes)
        return sum(loss)





class Proxywyf(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32,ps_mu=1.0, learn_mrg=True,ps_alpha=0.4):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.Tensor(nb_classes, sz_embed))
        nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
        self.n_classes = nb_classes
        self.input_dim = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.mrg_list = self.mrg
        self.learn_mrg = learn_mrg
        self.initialize_mrg(self.mrg_list, nb_classes)
        
    def initialize_mrg(self, mrg, nb_classes):
        self.mrg_list = torch.tensor([float(mrg)])
        if nb_classes:
            self.mrg_list = torch.ones(nb_classes) * self.mrg
        if self.learn_mrg:
            self.mrg_list = torch.nn.Parameter(self.mrg_list)

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxies, p=2, dim=1)
        if self.ps_mu > 0.0:
            input_l2, _, target  = proxy_synthesis(input_l2, proxy_l2, target, 
                                                         self.ps_alpha, self.ps_mu)
        cos = F.linear(input_l2, proxy_l2)  # Calcluate cosine similarity
        cos_proxies = F.linear(proxy_l2, proxy_l2)  # Calculate proxy cosine similarity
        
        #P_one_hot = F.one_hot(target, proxy_l2.shape[0])
        #N_one_hot = 1 - P_one_hot
        #with_neg_proxies = torch.nonzero(P_one_hot.sum(dim=0) == 0).squeeze(dim=1)
      
        # pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        # neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        #self.mrg_list[with_neg_proxies].detach()
        norm_mrg_list = self.mrg_list.reshape(1, -1)
        P_one_hot = binarize(T = target, nb_classes = self.n_classes)
        #P_one_hot = F.one_hot(target, proxy_l2.shape[0])
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - norm_mrg_list))
        #neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        neg_exp = torch.exp(self.alpha * (cos - norm_mrg_list - 0.15))
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        cos_one_hot = torch.eye(self.n_classes).cuda()
        cos_one_hot = 1 - cos_one_hot
        sim_proxy_sum = torch.exp(64 * (cos_proxies - norm_mrg_list ))
        sim_proxy_sum = (sim_proxy_sum * cos_one_hot).sum(dim=0)
        sim_proxy_sum = sim_proxy_sum[with_pos_proxies]
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        sim_proxy_term = torch.log(1 + sim_proxy_sum).sum() / proxy_l2.shape[0]
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() /  proxy_l2.shape[0]
        loss = pos_term + neg_term + sim_proxy_term    
        
        return loss


class PsPa(nn.Module):
    def __init__(self, input_dim, n_classes, alpha = 32, mrg = 0.1, ps_mu=0.0, ps_alpha=0.0):
        super(PsPa, self).__init__()
        self.proxies = torch.nn.Parameter(torch.Tensor(n_classes, input_dim))
        nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.mrg = mrg
        self.alpha = alpha
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxies, p=2, dim=1)
        
        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)
        cos = F.linear(input_l2, proxy_l2)  # Calcluate cosine similarity
       # P_one_hot = binarize(T = target, n_classes = self.n_classes)
        P_one_hot = F.one_hot(target, proxy_l2.shape[0])
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() /  proxy_l2.shape[0]
        loss = pos_term + neg_term     
        
        return loss

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.5, alpha=4):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = 32

    def forward(self, X, T):
        P = self.proxies
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calculate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        return loss


class CircleLoss(torch.nn.Module):
    r"""CircleLoss from
    `"Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    <https://arxiv.org/pdf/2002.10857>`_ paper.
    Parameters
    ----------
    m: float.
        Margin parameter for loss.
    gamma: int.
        Scale parameter for loss.
    Outputs:
        - **loss**: scalar.
    """
    def __init__(self, m, gamma):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.dp = 1 - m
        self.dn = m

    def forward(self, x, target):
        similarity_matrix = x @ x.T  # need gard here
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        negative_matrix = label_matrix.logical_not()
        positive_matrix = label_matrix.fill_diagonal_(False)

        sp = torch.where(positive_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))
        sn = torch.where(negative_matrix, similarity_matrix, torch.zeros_like(similarity_matrix))

        ap = torch.clamp_min(1 + self.m - sp.detach(), min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        logit_p = -self.gamma * ap * (sp - self.dp)
        logit_n = self.gamma * an * (sn - self.dn)

        logit_p = torch.where(positive_matrix, logit_p, torch.zeros_like(logit_p))
        logit_n = torch.where(negative_matrix, logit_n, torch.zeros_like(logit_n))

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss

class ProxyGML(nn.Module):
    def __init__(self, C, r, allNum, nums, N, weight_lambda, dim=512):
        super(ProxyGML, self).__init__()
        self.dim = dim
        self.C = C
        self.N = N
        self.r = r
        self.allNum = allNum
        self.nums = nums
        self.weight_lambda = weight_lambda
        # self.Proxies = torch.nn.Parameter(torch.randn(dim, C * N).cuda())# C*N表示所有类的proxy数目  C表示类数 N为一个类的proxy数目
        self.Proxies = torch.nn.Parameter(torch.randn(dim, allNum).cuda())
        # self.instance_label = torch.tensor(np.repeat(np.arange(C), N)).cuda()
        self.instance_label = self.convert(self.nums).cuda()
        print(self.instance_label)
        self.y_instacne_onehot = self.to_one_hot(self.instance_label, n_dims=self.C).cuda()  # 二值化
        # self.y_instacne_onehot = binarize2(self.instance_label,nums,allNum).cuda()
        print(self.y_instacne_onehot)
        torch.nn.init.kaiming_uniform_(self.Proxies, a=math.sqrt(5))
        self.index = 0
        print("#########")
        return

    def binarize2(T, nums, allNum):
        index = []
        cnt = 0
        nums = nums.int()
        lists = nums.numpy().tolist()
        for l in lists:
            cnt += l
            index.append(cnt)
        p_one_hot = torch.zeros(T.size(0), allNum)
        T = T.cpu().numpy()
        T = T.tolist()
        i = 0
        for t in T:
            p_one_hot[i][index[t] - lists[t]:index[t]] = 1
            i += 1
        return p_one_hot

    def convert(self, nums):
        newNums = []
        list = nums.numpy().tolist()
        i = 0
        for num in list:
            cnt = 0
            while cnt < num:
                newNums.append(i)
                cnt += 1
            i += 1
        newNums = torch.tensor(newNums)
        return newNums

    def to_one_hot(self, y, n_dims=None):
        ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
        y_tensor = y.type(torch.LongTensor).view(-1, 1)  # 换成竖的
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return y_one_hot

    def scale_mask_softmax(self, tensor, mask, softmax_dim, scale=1.0):
        # scale = 1.0 if self.opt.dataset != "online_products" else 20.0
        scale_mask_exp_tensor = torch.exp(tensor * scale) * mask.detach()
        scale_mask_softmax_tensor = scale_mask_exp_tensor / (
                1e-8 + torch.sum(scale_mask_exp_tensor, dim=softmax_dim)).unsqueeze(softmax_dim)  # 求比重
        return scale_mask_softmax_tensor

    def forward(self, input, target):
        self.index += 1
        centers = F.normalize(self.Proxies, p=2, dim=0)  # dim*proxies
        # constructing directed similarity graph
        similarity = input.matmul(centers)  # n*dim X dim*proxies=n*proxies?
        # relation-guided sub-graph construction
        positive_mask = torch.eq(target.view(-1, 1).cuda() - self.instance_label.view(1, -1),
                                 0.0).float().cuda()  # obtain positive mask
        # topk = math.ceil(self.r * self.C * self.N)  # 选取K个proxy
        topk = math.ceil(self.r * self.allNum.item())
        _, indices = torch.topk(similarity + 1000 * positive_mask, topk,
                                dim=1)  # "1000*" aims to rank faster 选取K个最相似的proxy
        mask = torch.zeros_like(similarity)
        mask = mask.scatter(1, indices, 1)  # 对indices二值化
        prob_a = mask * similarity  # n*proxies
        # revere label propagation (including classification process)
        logits = torch.matmul(prob_a, self.y_instacne_onehot)  # n*proxiesXproxies*C=n*C
        y_target_onehot = self.to_one_hot(target, n_dims=self.C).cuda()
        # y_target_onehot = binarize2(target,self.nums,self.allNum).cuda()
        logits_mask = 1 - torch.eq(logits, 0.0).float().cuda()  # 将logits二值化
        predict = self.scale_mask_softmax(logits, logits_mask, 1).cuda()
        # classification loss
        lossClassify = torch.mean(torch.sum(-y_target_onehot * torch.log(predict + 1e-20), dim=1))  # n*c 对行求和,然后求平均
        # regularization on proxies
        if self.weight_lambda > 0:
            simCenter = centers.t().matmul(centers)  # proxies*proxies
            centers_logits = torch.matmul(simCenter, self.y_instacne_onehot)  # proxies*C
            reg = F.cross_entropy(centers_logits, self.instance_label)
            return lossClassify + self.weight_lambda * reg, lossClassify
        else:
            return lossClassify, torch.tensor(0.0).cuda()


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes=self.nb_classes, embedding_size=self.sz_embed,
                                             softmax_scale=self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

class CirCleLoss(torch.nn.Module):
    def __init__(self, m, gamma):
        super(CirCleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss_func = losses.CircleLoss(m=self.m, gamma=self.gamma).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

 

class SoftTripleLoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(SoftTripleLoss, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.loss_func = losses.SoftTripleLoss(num_classes=self.nb_classes, embedding_size=self.sz_embed).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        """
        miner:样本挖掘
        """
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings=False)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MarginLoss(nn.Module):
    def __init__(self, num_classes, learn_beta):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.learn_beta = learn_beta
        self.miner = miner.DistanceWeightedSampling(batch_k=10)
        self.loss_func = losses.MarginLoss(num_classes=self.num_classes, learn_beta=self.learn_beta)

    def forward(self, embeddings, labels):
        pairs = self.miner(embeddings)
        loss = self.loss_func(embeddings, labels, pairs)
        return loss


# +*
class ProxyAnchorLoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        super(ProxyAnchorLoss, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.loss_func = losses.ProxyAnchorLoss(num_classes=self.nb_classes, embedding_size=self.sz_embed,
                                                margin=self.mrg, alpha=self.alpha).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
