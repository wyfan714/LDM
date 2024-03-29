from .base import *
import torchvision

#
#按照类别8:2的比例分配
#
class market(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/Market/pytorch'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            dir = "train_all"
            self.classes = range(0, 751)
        elif self.mode == 'eval':
            dir = "query"
            self.classes = range(0, 750)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root= os.path.join(self.root, dir)).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            self.ys += [y]
            self.I += [index]
            self.im_paths.append(os.path.join(self.root, i[0]))
            index += 1
