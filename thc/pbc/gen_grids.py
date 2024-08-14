import ctypes, numpy, scipy
import scipy.linalg

import pyscf
from pyscf.lib import logger
from pyscf.pbc.dft import numint

from pyscf import dft, lib
import pyscf.dft.gen_grid

from pyscf.dft.gen_grid import nwchem_prune, gen_atomic_grids
from pyscf.pbc.gto import eval_gto as pbc_eval_gto
libpbc = lib.load_library('libpbc')

import thc
from thc.mol.gen_grids import InterpolatingPointsMixin

class BeckeGrids(InterpolatingPointsMixin, pyscf.pbc.dft.gen_grid.BeckeGrids):
    def __init__(self, cell):
        self.mol = self.cell = cell
        pyscf.pbc.dft.gen_grid.BeckeGrids.__init__(self, cell)

    def build(self, *args, **kwargs):
        pyscf.pbc.dft.gen_grid.BeckeGrids.build(self, *args, **kwargs)
        return thc.mol.gen_grids.InterpolatingPointsMixin.build(self, *args, **kwargs)

    def _divide(self, coord):
        cell = self.cell
        from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
        xa = self.mol.atom_coords()
        xl = translation_vectors_for_kmesh(cell, [3, 3, 3], wrap_around=True)

        xg = coord

        nl = xl.shape[0]
        na = xa.shape[0]
        ng = xg.shape[0]
        assert na < ng

        dx = xl[:, None, None, :] + xa[None, :, None, :] - xg[None, None, :, :]
        d = numpy.linalg.norm(dx, axis=3)
        d = numpy.min(d, axis=0)
        assert d.shape == (na, ng)

        ind = numpy.argmin(d, axis=0)
        return [numpy.where(ind == ia)[0] for ia in range(na)]

    def _eval_gto(self, coord, weigh):
        phi = self.cell.pbc_eval_gto("GTOval", coord, kpt=None, kpts=None)
        return phi * (numpy.abs(weigh) ** 0.5)[:, None]
    
class UniformGrids(BeckeGrids, pyscf.pbc.dft.gen_grid.UniformGrids):
    def __init__(self, cell):
        self.mol = self.cell = cell
        pyscf.pbc.dft.gen_grid.UniformGrids.__init__(self, cell)

    def build(self, *args, **kwargs):
        log = logger.Logger(self.stdout, self.verbose)
        self.coords = pyscf.pbc.gto.get_uniform_grids(self.cell, self.mesh, wrap_around=False)
        thc.mol.gen_grids.InterpolatingPointsMixin.build(self, *args, **kwargs)
        return self

if __name__ == "__main__":
    c   = pyscf.pbc.gto.Cell()
    c.a = numpy.eye(3) * 3.5668
    c.atom = '''C     0.0000  0.0000  0.0000
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.0000
                C     2.6751  2.6751  0.8917
                C     1.7834  0.0000  1.7834
                C     2.6751  0.8917  2.6751
                C     0.0000  1.7834  1.7834
                C     0.8917  2.6751  2.6751'''
    c.basis  = 'gth-szv'
    c.pseudo = 'gth-pade'
    c.verbose = 0
    c.unit = 'aa'
    c.build()

    grid = UniformGrids(c)
    grid.mesh = [20, 20, 20]
    grid.verbose = 4
    grid.c_isdf  = None
    grid.kernel()

    grid = BeckeGrids(c)
    grid.level = 0
    grid.verbose = 4
    grid.c_isdf  = None
    grid.kernel()