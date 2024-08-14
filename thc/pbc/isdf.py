import pyscf.pbc
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

import thc
from thc.pbc.gen_grids import BeckeGrids
from thc.pbc.least_square import LeastSquareFitting

class InterpolativeSeparableDensityFitting(LeastSquareFitting):
    def __init__(self, cell):
        self.cell = self.mol = cell
        self.grids = BeckeGrids(cell)
        self.grids.level = 0
        self.max_memory = cell.max_memory
    
    def build(self):
        log = logger.Logger(self.stdout, self.verbose)
        self.grids.verbose = self.verbose
        
        cell = self.cell
        grids = self.grids

        if grids.coords is None:
            grids.dump_flags()
            grids.build()

        self.dump_flags()

        cput0 = (logger.process_clock(), logger.perf_counter())

        mesh = cell.mesh
        coord = cell.get_uniform_grids(mesh, wrap_around=False)
        weights = numpy.ones(coord.shape[0])

FFTISDF = ISDF = InterpolativeSeparableDensityFitting

if __name__ == '__main__':
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

    import thc
    thc = thc.ISDF(c)
    thc.verbose = 6
    thc.grids.c_isdf = 40
    thc.max_memory = 2000
    thc.build()

    vv = thc.coul
    xx = thc.xipt

    import pyscf.pbc.df

    df_obj = pyscf.pbc.df.FFTDF(c)
    df_obj.mesh = c.mesh
    df_obj.build()

    eri_ref = df_obj.get_eri(compact=False)

    eri_sol = numpy.einsum("IJ,Im,In,Jk,Jl->mnkl", vv, xx, xx, xx, xx, optimize=True)
    eri_sol = eri_sol.reshape(*eri_ref.shape)

    print(eri_ref[:10, :10])
    print(eri_sol[:10, :10])

    err = numpy.abs(eri_sol - eri_ref).max()
    print(err)