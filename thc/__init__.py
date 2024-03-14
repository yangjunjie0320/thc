import pyscf
from pyscf import gto as gto_mol
from pyscf.pbc import gto as gto_pbc

import thc
import thc.mol.isdf, thc.pbc.isdf

# Some wrappers for the molecular and periodic systems
def InterpolativeSeparableDensityFitting(m: pyscf.gto.MoleBase, *args, **kwargs):
    """Create an interpolative separable density fitting object for the given molecule or cell.
    
    Args:
        m: The molecule or cell for which to create the interpolative separable density fitting object.
    """

    assert isinstance(m, pyscf.gto.MoleBase)

    if isinstance(m, gto_mol.Mole):
        return thc.mol.isdf.InterpolativeSeparableDensityFitting(m, *args, **kwargs)
    
    elif isinstance(m, gto_pbc.Cell):
        return thc.pbc.isdf.InterpolativeSeparableDensityFitting(m, *args, **kwargs)
    
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
