import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

import thc
from thc.pbc.least_square import LeastSquareFitting
from thc.pbc.gen_grids import Grids

class WithKPoint(LeastSquareFitting):
    def __init__(self, cell, kpts=None):
        self.cell = self.mol = cell
        self.with_df = pbc.df.GDF(cell, kpts)
        self.grids = InterpolatingPoints(cell)
        self.grids.level = 0
        self.max_memory = cell.max_memory

    def eval_gto(self, coords, weights, kpt=None, kpts=None):
        print("eval_gto")
        print("coords", coords.shape)
        print("weights", weights.shape)
        phi = self.cell.pbc_eval_gto("GTOval", coords, kpt=kpt, kpts=kpts)
        phi *= (numpy.abs(weights) ** 0.5)[:, None]
        phi = numpy.array(phi)
        print("phi", phi.shape)
        return numpy.einsum("kxm,x->kxm", phi, numpy.sqrt(numpy.abs(weights)))

if __name__ == '__main__':
    import pyscf
    from pyscf import pbc
    c = pyscf.pbc.gto.Cell()
    c.atom = '''C     0.0000  0.0000  0.0000
                C     1.6851  1.6851  1.6851'''
    c.basis = '321g'
    c.a = '''0.0000, 3.3701, 3.3701
             3.3701, 0.0000, 3.3701
             3.3701, 3.3701, 0.0000'''
    c.unit = 'bohr'
    c.build()

    kmesh = [2, 2, 2]
    kpts  = c.make_kpts(kmesh)

    import thc
    thc = thc.LS(c, kpts=kpts)
    # thc.with_df = pyscf.pbc.df.rsdf.RSGDF(c, kpts=kpts)
    thc.with_df.verbose = 6
    thc.verbose = 6

    # thc.grids.c_isdf = 20
    # thc.max_memory = 2000
    # thc.build()

    # grids = pbc.dft.gen_grid.UniformGrids(c)
    # grids.atom_grid = (10, 86)
    # coord = thc.grids.coords
    # weigh = thc.grids.weights 
    from pyscf.pbc.dft.gen_grid import get_becke_grids
    from pyscf.pbc.dft.gen_grid import get_uniform_grids

    mesh = [10, 10, 10]
    coord = get_uniform_grids(c, mesh=mesh)
    weigh = c.vol / numpy.prod(mesh) * numpy.ones(numpy.prod(mesh))

    phi_k = thc.eval_gto(coord, weigh, kpts=kpts)
    phi_k = numpy.array(phi_k)

    ovlp_k_ref = thc.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)
    ovlp_k_sol = numpy.einsum("kxm,kxn->kmn", phi_k.conj(), phi_k)

    err = numpy.max(numpy.abs(ovlp_k_ref - ovlp_k_sol))
    print("npts = %d, err = %g" % (phi_k.shape[0], err))
