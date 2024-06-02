import pyscf
from pyscf import gto as gto_mol
from pyscf.pbc import gto as gto_pbc

import thc
import thc.mol.least_square

import thc.pbc.isdf
import thc.pbc.least_square

# Some wrappers for the molecular and periodic systems
def LeastSquareFitting(m: pyscf.gto.MoleBase, kpts=None):
    """Create an least square fitting tensor hyper-conraction object for the given molecule or cell.
    
    Args:
        m: The molecule or cell for which to create the interpolative separable density fitting object.
    """
    assert isinstance(m, pyscf.gto.MoleBase)
    if isinstance(m, gto_mol.Mole):
        assert kpts is None
        return thc.mol.least_square.LeastSquareFitting(m)
    
    elif isinstance(m, gto_pbc.Cell):
        if kpts is not None:
            return thc.pbc.least_square.WithKPoint(m, kpts)
        else:
            return thc.pbc.least_square.LeastSquareFitting(m)
    
    else:
        raise RuntimeError(f"Unsupported type for LS-THC: {type(m)}")
    
LS = LeastSquareFitting

def InterpolativeSeparableDensityFitting(m: pyscf.gto.MoleBase, *args, **kwargs):    
    assert isinstance(m, pyscf.gto.MoleBase)
    
    if isinstance(m, gto_pbc.Cell):
        return thc.pbc.isdf.InterpolativeSeparableDensityFitting(m, *args, **kwargs)
    
    else:
        raise RuntimeError(f"Unsupported type for ISDF: {type(m)}")
    
ISDF = FFTISDF = InterpolativeSeparableDensityFitting

def TensorHyperConraction(m: pyscf.gto.MoleBase, *args, method="isdf", **kwargs):
    """Create a tensor hyper-conraction object for the given molecule or cell.
    
    Args:
        m: The molecule or cell for which to create the tensor hyper-conraction object.
    """

    if method.lower() == "isdf":
        return InterpolativeSeparableDensityFitting(m, *args, **kwargs)
    
    elif method.lower() == "ls":
        return LeastSquareFitting(m, *args, **kwargs)

    else:
        raise RuntimeError(f"Unsupported method for THC: {method}")
    
THC = TensorHyperConraction
