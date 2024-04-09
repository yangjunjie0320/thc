import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import df
from pyscf import lib
from pyscf.dft import numint
from pyscf.lib import logger

from thc.mol.gen_grids import InterpolatingPoints

# def cholesky(x, tol=1e-8, log=None):
#     ngrid, nao = x.shape
    

#     from scipy.linalg.lapack import dpstrf
#     chol, perm, rank, info = dpstrf(x4, tol=tol)

#     nisp = rank
#     if log is not None:
#         log.info("Cholesky: rank = %d / %d" % (rank, ngrid))

#     perm = (numpy.array(perm) - 1)[:nisp]

#     tril = numpy.tril_indices(nisp, k=-1)
#     chol = chol[:nisp, :nisp]
#     chol[tril] *= 0.0
#     return chol, x[perm]

class TensorHyperConractionMixin(lib.StreamObject):
    tol = 1e-4
    max_memory = 1000
    verbose = 5

    vv = None
    xip = None

class LeastSquareFitting(TensorHyperConractionMixin):
    def __init__(self, mol):
        self.mol = mol
        self.with_df = df.DF(mol)
        self.grids = InterpolatingPoints(mol)
        self.grids.level = 0

        self.tol = 1e-8
        self.max_memory = mol.max_memory

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)

        log.info('\n******** %s ********', self.__class__)
        for k, v in self.__dict__.items():
            if not isinstance(v, (int, float, str)) or k == "verbose": continue
            log.info('%s = %s', k, v)
        log.info('')

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

        naux = with_df.get_naoaux()
        phi  = numint.eval_ao(self.mol, grids.coords)
        phi *= (numpy.abs(grids.weights) ** 0.5)[:, None]

        zeta = lib.dot(phi, phi.T) ** 2
        chol, perm, rank = lib.scipy_helper.pivoted_cholesky(zeta, tol=self.tol)
        log.info("Pivoted Cholesky rank: %d / %d", rank, phi.shape[0])

        mask = perm # [:rank]
        xx = phi[mask]
        nip, nao = xx.shape
        cput1 = logger.timer(self, "interpolating vectors", *cput0)

        # Build the coulomb kernel
        # rhs = numpy.einsum("Qmn,Im,In->QI", cderi, xip, xip)
        rhs = numpy.zeros((naux, nip))
        blksize = int(self.max_memory * 1e6 * 0.5 / (8 * nao ** 2))
        blksize = max(4, blksize)

        a0 = a1 = 0 # slice for auxilary basis
        for cderi in with_df.loop(blksize=blksize):
            a1 = a0 + cderi.shape[0]
            cderi = lib.unpack_tril(cderi)
            rhs[a0:a1] = numpy.einsum("Qmn,Im,In->QI", cderi, xx, xx, optimize=True)

            # for i0, i1 in lib.prange(0, nip, blksize): # slice for interpolating vectors
            #     # TODO: sum over only the significant shell pairs
            #     cput = (logger.process_clock(), logger.perf_counter())

            #     ind = numpy.arange(nao)
            #     # x2  = xx[i0:i1, :, numpy.newaxis] * xx[i0:i1, numpy.newaxis, :]
            #     # x2  = lib.pack_tril(x2 + x2.transpose(0, 2, 1))
            #     # x2[:, ind * (ind + 1) // 2 + ind] *= 0.5
            #     rhs[a0:a1, i0:i1] += numpy.einsum("Qmn,Im,In->QI", cderi, xx[i0:i1], xx[i0:i1], optimize=True)
            #     logger.timer(self, "RHS [%4d:%4d, %4d:%4d]" % (a0, a1, i0, i1), *cput)
                # x2 = None

            a0 = a1

        cput1 = logger.timer(self, "RHS", *cput1)

        x4 = lib.dot(xx, xx.T) ** 2
        coul, res, rank, s = scipy.linalg.lstsq(x4.T, rhs.T)
        coul = coul.T

        cput1 = logger.timer(self, "solving linear equations", *cput1)
        log.info(
            "Least squares: res = %6.4e, rank = %4d / %4d, cond = %6.4e",
            numpy.linalg.norm(res) / nip, rank, nip, s[rank]
            )
        logger.timer(self, "LS-THC", *cput0)

        self.coul = coul
        self.vipt = xx
        return coul, xx

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
    thc.verbose = 6
    thc.tol = 1e-16
    thc.grids.level  = 0
    thc.grids.c_isdf = 20
    thc.with_df.auxbasis = "weigend"
    thc.max_memory = 2000
    thc.build()

    vv = thc.coul
    xx = thc.vipt

    from pyscf.lib import unpack_tril
    df_chol_ref = unpack_tril(thc.with_df._cderi)
    df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xx, xx, optimize=True)

    err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
    err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol) / m.natm
    print("Method = %s, Error = % 6.4e % 6.4e" % ("cholesky", err1, err2))
