from .base import *
import torchvision


class aid(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/AID'
        self.mode = mode
        self.transform = transform
        self.classes = range(0,30)
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        tt = torchvision.datasets.ImageFolder(root=self.root)
        sss =tt.imgs
        targ = tt.targets
        lis = [360, 310, 220, 400, 360, 260, 240, 350, 410, 300, 370, 250, 390, 280, 290, 340, 350, 390, 370, 420, 380, 260, 290, 410, 300, 300, 330, 290, 360, 420]
        # pre = 0
        # count = 0
        # for j in targ:
        #     if j==pre:
        #         count = count+1
        #     else:
        #         lis.append(count)
        #         pre=j
        #         count=1
        # lis.append(count)
        # print(lis)
        if self.mode == 'train':
            for i in sss:
                # i[1]: label, i[0]: root
                y = i[1]
                # fn image name like airport_1.jpg
                fn = os.path.split(i[0])[1]
                ssn = i[0][-9:-4]
                ssn = ssn.split('_')[1]
                if y in self.classes and int(ssn) <= lis[y]*0.5:
                    self.ys += [y]
                    self.I += [index]
                    self.im_paths.append(os.path.join(self.root, i[0]))
                    index += 1
            #print(self.ys)
        elif self.mode == 'eval':
            for i in sss:
                # i[1]: label, i[0]: root
                y = i[1]
                # fn needed for removing non-images starting with `._`
                fn = os.path.split(i[0])[1]
                #print(i[0])
                ssn = i[0][-9:-4]
                ssn = ssn.split('_')[1]
                #rint(ssn,lis[y])
                if y in self.classes and int(ssn) > lis[y]*0.5:
                    self.ys += [y]
                    self.I += [index]
                    self.im_paths.append(os.path.join(self.root, i[0]))
                    index += 1
            #print(self.ys.size)
