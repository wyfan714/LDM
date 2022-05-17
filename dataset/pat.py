from .base import *
import torchvision


class pat(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/PatternNet/images'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,18)
        elif self.mode == 'eval':
            self.classes = range(19,38)
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        im = torchvision.datasets.ImageFolder(root=self.root).imgs
        for i in im:
                # i[1]: label, i[0]: root
            y = i[1]
            fn = os.path.split(i[0])[1]
            if y in self.classes:
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(self.root, i[0]))
                index += 1
