from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .dog import DogsDataset
from .import utils
from .base import BaseDataset
from .market import market1501

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'dog': DogsDataset
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
