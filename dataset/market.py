from .base import *
import os.path as osp
import torch
import PIL.Image
import glob
import re


class market1501(torch.utils.data.Dataset):
    dataset_dir = 'market1501'
    def __init__(self,root,mode,transform=None):
        self.mode = mode
        self.transform = transform
        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        for img_path, key, _ in train:
            self.train_im_paths.append(img_path)
            self.train_ys += [int(key)]

        for img_path, key, _ in query:
            self.query_im_paths.append(img_path)
            self.query_ys += [int(key)]

        for img_path, key, _ in gallery:
            self.gallery_im_paths.append(img_path)
            self.gallery_ys += [int(key)]

        if self.mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
            self.pids, self.camids = self.get_imagedata_info(self.train)
        elif self.mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
            self.pids, self.camids = self.get_imagedata_info(self.query)
        elif self.mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys
            self.pids,self.camids = self.get_imagedata_info(self.gallery)
            
    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        return len(self.ys)
            
    def __getitem__(self, index):
        
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]

        return im, target

    
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path,pid,camid))
        return dataset

    def get_imagedata_info(self, data):
        pids = []
        camids = []
        for _, pid, camid in data:
            pids += [pid]
            camids += [camid]
        return pids,camids
