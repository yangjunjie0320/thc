import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

import thc
from thc.pbc.gen_grids import BeckeGrids

class LeastSquareFitting(thc.mol.LeastSquareFitting):
    def __init__(self, cell):
        self.cell = self.mol = cell
        self.with_df = pbc.df.GDF(cell)
        self.grids = BeckeGrids(cell)
        self.grids.level = 0
        self.max_memory = cell.max_memory

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)

        log.info('\n******** %s ********', self.__class__)
        for k, v in self.__dict__.items():
            if not isinstance(v, (int, float, str)) or k == "verbose": continue
            log.info('%s = %s', k, v)
        log.info('')

    def eval_gto(self, coords, weights):
        phi = self.cell.pbc_eval_gto("GTOval", coords)
        phi *= (numpy.abs(weights) ** 0.5)[:, None]
        return phi

LS = LeastSquareFitting

if __name__ == '__main__':
    c = pyscf.pbc.gto.Cell()
    c.atom = '''C     0.0000  0.0000  0.0000
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.0000
                C     2.6751  2.6751  0.8917
                C     1.7834  0.0000  1.7834
                C     2.6751  0.8917  2.6751
                C     0.0000  1.7834  1.7834
                C     0.8917  2.6751  2.6751'''
    c.basis = '321g'
    c.a = numpy.eye(3) * 3.5668
    c.unit = 'aa'
    c.build()

    import thc
    thc = thc.LS(c)
    thc.verbose = 6
    thc.grids = BeckeGrids(c)
    thc.grids.level = 1
    thc.grids.c_isdf = None
    thc.max_memory = 2000
    thc.build()

    coul = thc.coul
    xipt = thc.xipt

    from pyscf.lib import unpack_tril
    a0 = a1 = 0 # slice for auxilary basis
    for cderi in thc.with_df.loop(blksize=200):
        a1 = a0 + cderi.shape[0]

        df_chol_sol = numpy.einsum("QI,Im,In->Qmn", coul[a0:a1], xipt, xipt, optimize=True)
        df_chol_ref = unpack_tril(cderi)

        err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
        err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol)
        print("err[%4d:%4d] Max: %6.4e, Mean: %6.4e" % (a0, a1, err1, err2))
        a0 = a1

        df_chol_sol = None
        df_chol_ref = None
        
