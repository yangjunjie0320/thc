import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import df
from pyscf import lib
from pyscf.dft import numint
from pyscf.lib import logger

from thc.mol.gen_grids import Grids

class TensorHyperConractionMixin(lib.StreamObject):
    max_memory = 1000
    verbose = 5

    coul = None
    xipt = None

    tol = 1e-16

class LeastSquareFitting(TensorHyperConractionMixin):
    def __init__(self, mol):
        self.mol = mol
        self.with_df = df.DF(mol)
        self.grids = Grids(mol)
        self.grids.level = 0
        self.max_memory = mol.max_memory

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)

        log.info('\n******** %s ********', self.__class__)
        for k, v in self.__dict__.items():
            if not isinstance(v, (int, float, str)) or k == "verbose": continue
            log.info('%s = %s', k, v)
        log.info('')

    def eval_gto(self, coords, weights):
        phi  = numint.eval_ao(self.mol, coords)
        phi *= (numpy.abs(weights) ** 0.5)[:, None]
        return phi

    def build(self):
        log = logger.Logger(self.stdout, self.verbose)
        self.with_df.verbose = self.verbose
        self.grids.verbose = self.verbose

        with_df = self.with_df
        grids   = self.grids

        if self.grids.coords is None:
            log.info('\n******** %s ********', self.grids.__class__)
            self.grids.dump_flags()
            self.grids.build()

        if self.with_df._cderi is None:
            log.info('\n')
            self.with_df.build()

        self.dump_flags()

        cput0 = (logger.process_clock(), logger.perf_counter())

        phi = self.eval_gto(grids.coords, grids.weights)
        zeta = lib.dot(phi, phi.T) ** 2
        ng, nao = phi.shape

        chol, perm, rank = lib.scipy_helper.pivoted_cholesky(zeta, tol=self.tol, lower=False)
        nip = rank
        
        perm = perm[:nip]
        chol = chol[:nip, :nip]
        err  = abs(chol[nip - 1, nip - 1])
        log.info("Pivoted Cholesky rank: %d / %d, err = %6.4e", rank, ng, err)

        xipt = phi[perm]
        cput1 = logger.timer(self, "interpolating vectors", *cput0)

        # Build the coulomb kernel
        # rhs = numpy.einsum("Qmn,Im,In->QI", cderi, xip, xip)
        naux = with_df.get_naoaux()
        rhs = numpy.zeros((naux, nip))

        # TODO: Is this correct?
        blksize = int(self.max_memory * 1e6 * 0.5 / (8 * nao ** 2))
        blksize = max(4, blksize)

        a0 = a1 = 0 # slice for auxilary basis
        for cderi in with_df.loop(blksize=blksize):
            a1 = a0 + cderi.shape[0]
            for i0, i1 in lib.prange(0, nip, blksize): # slice for interpolating vectors
                # TODO: sum over only the significant shell pairs
                cput = (logger.process_clock(), logger.perf_counter())

                ind = numpy.arange(nao)
                x2  = xipt[i0:i1, :, numpy.newaxis] * xipt[i0:i1, numpy.newaxis, :]
                x2  = lib.pack_tril(x2 + x2.transpose(0, 2, 1))
                x2[:, ind * (ind + 1) // 2 + ind] *= 0.5
                rhs[a0:a1, i0:i1] += lib.dot(cderi, x2.T)
                logger.timer(self, "RHS [%4d:%4d, %4d:%4d]" % (a0, a1, i0, i1), *cput)
                x2 = None
            a0 = a1

        cput1 = logger.timer(self, "RHS", *cput1)

        ww = scipy.linalg.solve_triangular(chol.T, rhs.T, lower=True).T
        coul = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T
        
        cput1 = logger.timer(self, "solving linear equations", *cput1)
        logger.timer(self, "LS-THC", *cput0)

        self.coul = coul
        self.xipt = xipt
        return coul, xipt

LS = LeastSquareFitting

if __name__ == '__main__':
    m = pyscf.gto.M(
        atom="""
        O   -6.0082242   -6.2662586   -4.5802338
        H   -6.5424905   -5.6103439   -4.0656431
        H   -5.6336920   -6.8738199   -3.8938766
        O   -5.0373186   -5.2694388   -1.2581917
        H   -4.2054186   -5.5931034   -0.8297803
        H   -4.7347234   -4.8522045   -2.1034720
        """, basis="ccpvdz", verbose=0
    )

    import thc
    thc = thc.LS(m)
    thc.grids.level  = 3
    thc.verbose      = 6
    thc.grids.c_isdf = None
    thc.grids.tol    = 1e-20
    thc.max_memory   = 500
    thc.build()

    coul = thc.coul
    xipt = thc.xipt

    from pyscf.lib import unpack_tril
    a0 = a1 = 0 # slice for auxilary basis
    for cderi in thc.with_df.loop(blksize=50):
        a1 = a0 + cderi.shape[0]

        df_chol_sol = numpy.einsum("QI,Im,In->Qmn", coul[a0:a1], xipt, xipt, optimize=True)
        df_chol_ref = unpack_tril(cderi)

        err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
        err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol)
        print("err[%4d:%4d] Max: %6.4e, Mean: %6.4e" % (a0, a1, err1, err2))
        a0 = a1

        df_chol_sol = None
        df_chol_ref = None

