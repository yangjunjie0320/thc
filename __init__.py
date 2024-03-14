import pyscf
from pyscf import gto as mol
from pyscf.pbc import gto as pbc

import thc
import thc.mol, thc.pbc

# Some wrappers for the molecular and periodic systems

def InterpolatingPoints(m: pyscf.gto.MoleBase, *args, **kwargs):
    """Create an interpolating points object for the given molecule or cell.
    
    Args:
        m: The molecule or cell for which to create the interpolating points.
    """

    assert isinstance(m, pyscf.gto.MoleBase)

    if isinstance(m, mol.Mole):
        return thc.mol.InterpolatingPoints(m, *args, **kwargs)
    
    elif isinstance(m, pbc.Cell):
        raise NotImplementedError("Interpolating points for periodic systems is not yet implemented.")
        # return thc.pbc.InterpolatingPoints(m, *args, **kwargs)
    
    else:
        raise TypeError(f"Unknown type {type(m)}")
    
Grids = InterpolatingPoints

def InterpolativeSeparableDensityFitting(m: pyscf.gto.MoleBase, *args, **kwargs):
    """Create an interpolative separable density fitting object for the given molecule or cell.
    
    Args:
        m: The molecule or cell for which to create the interpolative separable density fitting object.
    """

    assert isinstance(m, pyscf.gto.MoleBase)

    if isinstance(m, mol.Mole):
        return thc.mol.InterpolativeSeparableDensityFitting(m, *args, **kwargs)
    
    elif isinstance(m, pbc.Cell):
        raise NotImplementedError("Interpolative separable density fitting for periodic systems is not yet implemented.")
        # return thc.pbc.InterpolativeSeparableDensityFitting(m, *args, **kwargs)
    
    else:
        raise TypeError(f"Unknown type {type(m)}")
    
ISDF = InterpolativeSeparableDensityFitting

def TensorHyperConraction(m: pyscf.gto.MoleBase, *args, method="isdf", **kwargs):
    """Create a tensor hyper-conraction object for the given molecule or cell.
    
    Args:
        m: The molecule or cell for which to create the tensor hyper-conraction object.
    """

    if method == "isdf":
        return InterpolativeSeparableDensityFitting(m, *args, **kwargs)

    else:
        raise NotImplementedError(f"Method {method} is not yet implemented.")
    
THC = TensorHyperConraction
