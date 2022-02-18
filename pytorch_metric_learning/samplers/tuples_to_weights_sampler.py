import torch
from torch.utils.data.sampler import Sampler
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..testers import BaseTester
import numpy as np
import logging


class TuplesToWeightsSampler(Sampler):
    def __init__(self, model, miner, dataset, subset_size=None, **tester_kwargs):
        self.model = model
        self.miner = miner
        self.dataset = dataset
        self.subset_size = subset_size
        self.tester = BaseTester(**tester_kwargs)
        self.device = self.tester.data_device
        self.weights = None

    def __len__(self):
        if self.subset_size:
            return self.subset_size
        return len(self.dataset)

    def __iter__(self):
        logging.info("Computing embeddings in {}".format(self.__class__.__name__))

        if self.subset_size:
            indices = c_f.safe_random_choice(
                np.arange(len(self.dataset)), size=self.subset_size
            )
            curr_dataset = torch.utils.data.Subset(self.dataset, indices)
        else:
            indices = torch.arange(len(self.dataset)).to(self.device)
            curr_dataset = self.dataset

        embeddings, labels = self.tester.get_all_embeddings(curr_dataset, self.model)
        embeddings = torch.from_numpy(embeddings).to(self.device)
        labels = torch.from_numpy(labels).to(self.device).squeeze(1)
        hard_tuples = self.miner(embeddings, labels)

        self.weights = torch.zeros(len(self.dataset)).to(self.device)
        self.weights[indices] = lmu.convert_to_weights(
            hard_tuples, labels, dtype=torch.float32
        )
        return iter(
            torch.utils.data.WeightedRandomSampler(
                self.weights, self.__len__(), replacement=True
            )
        )
