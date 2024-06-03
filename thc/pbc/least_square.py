import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

import thc
from thc.pbc.gen_grids import InterpolatingPoints

class LeastSquareFitting(thc.mol.LeastSquareFitting):
    def __init__(self, cell):
        self.cell = self.mol = cell
        self.with_df = pbc.df.GDF(cell)
        self.grids = InterpolatingPoints(cell)
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

from thc.pbc.k_least_square import WithKPoint
LeastSquareFittingWithKPoint = WithKPoint

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
    thc.with_df = pyscf.pbc.df.rsdf.RSGDF(c)
    thc.with_df.verbose = 6
    thc.verbose = 6
    thc.grids.c_isdf = 20
    thc.max_memory = 2000
    thc.build()

    vv = thc.coul
    xx = thc.xipt

    from pyscf.lib import pack_tril, unpack_tril
    df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xx, xx, optimize=True)

    df_chol_ref = numpy.zeros_like(df_chol_sol)
    a0 = a1 = 0 # slice for auxilary basis
    for cderi in thc.with_df.loop(blksize=20):
        a1 = a0 + cderi.shape[0]
        df_chol_ref[a0:a1] = unpack_tril(cderi)
        a0 = a1

    err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
    err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol) / df_chol_ref.size
    print("Method = %s, Error = % 6.4e % 6.4e" % ("cholesky", err1, err2))
