import numpy, scipy

import pyscf
from pyscf import lib

class TensorHyperConraction(lib.StreamObject):
    tol = 1e-4
    def __init__(self, mol):
        self.mol    = mol