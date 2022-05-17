from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .dog import DogsDataset
from .import utils
from .base import BaseDataset
from .ucmd import ucmd
from .pat import pat
from .aid import aid
from .market import market

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'ucmd': ucmd,
    'aid': aid,
    'pat': pat,
    'market': market,
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
