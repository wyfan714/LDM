# from .base import *
import scipy.io
from torchvision.datasets.utils import list_dir
from torch.utils.data import Dataset
import os
from PIL import Image

class DogsDataset(Dataset):
    def __init__(self, root, mode, transform=None, cropped=False):
        self.root = root + '/dog'
        self.mode = mode
        self.cropped = cropped
        self.transform = transform
        self.ys = []
        if self.mode == 'train':
            self.classes = range(0, 120)
        elif self.mode == 'eval':
            self.classes = range(0, 120)

        # BaseDataset.__init__(self, self.root, self.mode, self.transform)
        split = self.load_split()
        self.images_folder = os.path.join(self.root, 'Images/Images')
        self.annotations_folder = os.path.join(self.root, 'Annotations/Annotation')
        self._breeds = list_dir(self.images_folder)
        if self.cropped:
            self._breed_annotations = [
                [(annotation, box, idx) for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])
            self._flat_breed_images = [(annotation + '.jpg', idx) for annotation, box, idx in
                                       self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]
            self._flat_breed_images = self._breed_images

        for _, y in self._flat_breed_images:
            if y in self.classes:
                self.ys.append(y)

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        # if self.target_transform:
        #     target_class = self.target_transform(target_class)

        return image, target_class

    def load_split(self):
        if self.mode == 'train':
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        elif self.mode == 'eval':
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']
        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)
                          ])
        return boxes
