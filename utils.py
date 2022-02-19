import numpy as np
import torch
import logging
import losses
import json
import os
from tqdm import tqdm
import torch.nn.functional as F
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    A[0] = torch.stack(A[0])
    A[1] = torch.stack(A[1])
    return [A[0], A[1], A[2]]

def calc_map_at_k(T, Y, k):
    map = 0
    for t,y in zip(T, Y):
        cnt = 0
        s = 0
        ap = 0
        for i in range(k):
            cnt = cnt + 1
            if t == y[i]:
                s += 1
                ap += s / cnt
        if s != 0:
            map += ap / s
    return map / len(T)

def evaluate_map(model, dataloader):
    X, T, path = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    K = 20
    Y = []
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()
    map = calc_map_at_k(T, Y, K)
    print("map@{}:{:.3f}".format(K, 100 * map))
    return map

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T ,path = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    dt_name = dataloader.dataset.__class__.__name__
 #   draw_tsne(X ,path, dt_name)
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    indice = cos_sim.topk(1+4)[1].cpu().numpy().tolist()
    #ps = path[cos_sim.topk(1 + 4)[1][:,1:]]
    #draw_retrieval(path, indice, Y, T, dt_name)
    # for i in range(len(t)):
    #     for j in t[i]:
    #         paths[i].append(j)
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    query_X, query_T, query_path = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T, gallery_path = predict_batchwise(model, gallery_dataloader)
    dt_name = query_dataloader.dataset.__class__.__name__
    # query_x 14218 512
    # gallery_x 12612 512
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []

    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1

        return match_counter / m

    # calculate recall @ 1, 2, 4, 8
    recall = []
    Y = gallery_T[cos_sim.topk(K)[1][:, :]]
    Y = Y.float().cpu()
    indice = cos_sim.topk(4)[1].cpu().numpy().tolist()
    # #ps = path[cos_sim.topk(1 + 4)[1][:,1:]]
    #draw_retrieval_inshop(query_path, gallery_path, indice, Y, query_T, dt_name)
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = calc_recall_at_k(query_T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    # for k in [1, 10, 20, 30, 40, 50]:
    #     r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
    #     recall.append(r_at_k)
    #     print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T , path = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    dt_name = dataloader.dataset.__class__.__name__
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    indice = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs, X)
            y = T[cos_sim.topk(1 + K)[1][:, 1:]]
            t = cos_sim.topk(1 + 4)[1]
            Y.append(y.float().cpu())
            indice.append(t.cpu())
            xs = []


    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs, X)
    y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    t = cos_sim.topk(1 + 4)[1]
    Y.append(y.float().cpu())
    indice.append(t.cpu())
    Y = torch.cat(Y, dim=0)
    indice = torch.cat(indice, dim=0)
    indice = indice.numpy().tolist()
    #draw_retrieval(path, indice, Y, T, dt_name)
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall

def plot_embedding(data, path):
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)

	random_arr = np.random.choice(len(path), len(path)//2 ,replace=False)
	fig = plt.figure(figsize=(300,300))
	ax = plt.subplot(111)
	plt.xticks([])
	plt.yticks([])
	ax.axis('off')
	#for i in tqdm(range(int(data.shape[0]/3))):
	for i in tqdm(range(random_arr.shape[0])):
		ax1 = fig.add_axes([data[random_arr[i], 0], data[random_arr[i], 1],0.01,0.01])
		img = Image.open(path[random_arr[i]])
		ax1.imshow(img)
		ax1.axis('off')

	return fig

def draw_tsne(X ,path, dt_name):
	print('Computing t-SNE embedding')

	data=X.cpu().numpy()
	if not os.path.exists(dt_name+'tsne.npz'):
		tsne = TSNE(n_components=2, init='pca', random_state=0)
		result = tsne.fit_transform(data)
		np.savez(dt_name+"tsne",result)
	else:
		npzfile=np.load(dt_name+'tsne.npz')
		result = npzfile['arr_0']

	fig = plot_embedding(result, path)
	fig.savefig(dt_name+"tsne.jpg")

def bbox_to_rect(bbox, color):
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=30)

def draw_retrieval(path, indice, Y, T, dt_name):
    random_arr = np.random.choice(len(path), 5, replace=False)
   # print(random_arr[0])
    fig = plt.figure(figsize=(100, 100))
    cnt = 1

    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    for i in tqdm(range(random_arr.shape[0])):
        for j in range(len(indice[random_arr[i]])):
            sub = plt.subplot(random_arr.shape[0], 5, cnt)
            cnt += 1
            sub.axis('off')
            img = Image.open(os.path.join(path[indice[random_arr[i]][j]]))
            img = img.resize((1000, 1000), Image.ANTIALIAS)
            # if j == 0:
            #     sub.set_title('Query')
            # if j == 2:
            #     sub.set_title('Top-4 Retrieval')
            plt.imshow(img)
            ax = sub.axis()
            if j == 0:
                sub.axes.add_patch(bbox_to_rect([ax[0], ax[2], ax[1], ax[3]], 'black'))
            if j > 0 and Y[random_arr[i]][j-1] == T[random_arr[i]]:
                sub.axes.add_patch(bbox_to_rect([ax[0], ax[2], ax[1], ax[3]], 'green'))
            if j > 0 and Y[random_arr[i]][j-1] != T[random_arr[i]]:
                sub.axes.add_patch(bbox_to_rect([ax[0], ax[2], ax[1], ax[3]], 'red'))
    plt.subplots_adjust(hspace=0, wspace=0.1)
    fig.savefig(dt_name+"retrieval.jpg")

def draw_retrieval_inshop(query_path, gallery_path, indice, Y, T, dt_name):
    random_arr = np.random.choice(len(query_path), 5, replace=False)
    fig = plt.figure(figsize=(100, 100))
    cnt = 1
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    for i in tqdm(range(random_arr.shape[0])):
        for j in range(len(indice[random_arr[i]])+1):
            sub = plt.subplot(5, 5, cnt)
            cnt += 1
            sub.axis('off')
            if j > 0:
                img = Image.open(os.path.join(gallery_path[indice[random_arr[i]][j-1]]))
            else:
                img = Image.open(os.path.join(query_path[random_arr[i]]))
            img = img.resize((1000, 1000), Image.ANTIALIAS)
            # if j == 0:
            #     sub.set_title('Query')
            # if j == 2:
            #     sub.set_title('Top-4 Retrieval')
            plt.imshow(img)
            ax = sub.axis()
            if j == 0:
                sub.axes.add_patch(bbox_to_rect([ax[0], ax[2], ax[1], ax[3]], 'black'))
            if j > 0 and Y[random_arr[i]][j-1] == T[random_arr[i]]:
                sub.axes.add_patch(bbox_to_rect([ax[0], ax[2], ax[1], ax[3]], 'green'))
            if j > 0 and Y[random_arr[i]][j-1] != T[random_arr[i]]:
                sub.axes.add_patch(bbox_to_rect([ax[0], ax[2], ax[1], ax[3]], 'red'))
    plt.subplots_adjust(hspace=0, wspace=0.1)
    fig.savefig(dt_name+"retrieval.jpg")

def evaluate_cos_Market(model,query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    num_query = len(query_dataloader.dataset.query)
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)

    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)

# get predictions by assigning nearest 8 neighbors with cosine

    distmat = euclidean_distance(query_X, gallery_X)
    q_pids = query_dataloader.dataset.pids
    q_pids = np.asarray(q_pids[:])
    g_pids = gallery_dataloader.dataset.pids
    g_pids = np.asarray(g_pids[:])
    q_camids = query_dataloader.dataset.camids
    q_camids = np.asarray(q_camids[:])
    g_camids = gallery_dataloader.dataset.camids
    g_camids = np.asarray(g_camids[:])
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    recall = []
    print("R@{} : {:.3f}".format(1, 100 * cmc[0]))
    recall.append(cmc[0])
    print("mAP@ : {:.3f}".format(100 * mAP))
    recall.append(mAP)
    return  recall

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def evaluate2(model,query_dataloader, gallery_dataloader):

    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)

    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    query = query_X.t().cpu().numpy()
    gallery = gallery_X.cpu().numpy()
    score = np.dot(gallery,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    q_camids = query_dataloader.dataset.camids
    q_camids = np.asarray(q_camids[:])
    g_camids = gallery_dataloader.dataset.camids
    g_camids = np.asarray(g_camids[:])
    query_index = np.argwhere(gallery_dataloader==query_dataloader)
    camera_index = np.argwhere(g_camids==q_camids)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gallery_dataloader==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    mAP,cmc = compute_mAP(index, good_index, junk_index)
    recall = []
    print("R@{} : {:.3f}".format(1, 100 * cmc[0]))
    recall.append(cmc[0])
    print("mAP@ : {:.3f}".format(100 * mAP))
    recall.append(mAP)
    return  recall


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
