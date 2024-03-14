import numpy, scipy
import scipy.linalg

import pyscf
from pyscf.pbc import df
from pyscf import lib
from pyscf.dft import numint
from pyscf.lib import logger

import thc
from thc.mol.isdf import cholesky
from thc.pbc.gen_grids import InterpolatingPoints

class InterpolativeSeparableDensityFitting(thc.mol.isdf.InterpolativeSeparableDensityFitting):
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
        rho  = numpy.einsum("Im,In->Imn", phi, phi)
        # rho  = lib.pack_tril(rho)
        # chol, xip = cholesky(phi, tol=1e-4, log=log)
        # nip, nao  = xip.shape
        # cput1 = logger.timer(self, "interpolating vectors", *cput0)

        # # Build the coulomb kernel
        # # coul = numpy.einsum("Qmn,Im,In->QI", cderi, xip, xip)
        # coul = numpy.zeros((naux, nip))
        # blksize = int(self.max_memory * 1e6 * 0.5 / (8 * nao ** 2))
        # blksize = max(4, blksize)

        # a0 = a1 = 0 # slice for auxilary basis
        # for cderi in with_df.loop(blksize=blksize):
        #     a1 = a0 + cderi.shape[0]

        #     for i0, i1 in lib.prange(0, nip, blksize): # slice for interpolating vectors
        #         cput = (logger.process_clock(), logger.perf_counter())
        #         xx = lib.pack_tril(numpy.einsum("Im,In->Imn", xip[i0:i1], xip[i0:i1]))
        #         coul[a0:a1, i0:i1] += (xx.dot(cderi.T)).T * 2.0
        #         logger.timer(self, "coulomb kernel [%4d:%4d, %4d:%4d]" % (a0, a1, i0, i1), *cput)
        #         xx = None

        #     a0 = a1

        # cput1 = logger.timer(self, "coulomb kernel", *cput1)

        # ww = scipy.linalg.solve_triangular(chol.T, coul.T, lower=True).T
        # vv = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T
        # cput1 = logger.timer(self, "solving linear equations", *cput1)

        nao = phi.shape[1]
        df_cderi = numpy.zeros((naux, nao, nao))
        q0 = 0
        for cderi in with_df.loop():
            q1 = q0 + cderi.shape[0]
            df_cderi[q0:q1] = lib.unpack_tril(cderi)
            q0 = q1

        coul = numpy.einsum("Qmn,Inm->QI", df_cderi, rho)

        rhorho = numpy.einsum("Imn,Jmn->IJ", rho, rho)

        u, s, vh = scipy.linalg.svd(rhorho)
        m = numpy.sum(s > 1e-10)
        print("SVD: %d / %d" % (m, len(s)))

        rhorho_pinv = numpy.dot(vh[:m].T, numpy.dot(numpy.diag(1.0 / s[:m]), u.T[:m]))
        vv = coul.dot(rhorho_pinv)
        xip = phi

        self.vv = vv
        self.xip = xip
        return vv, xip

ISDF = InterpolativeSeparableDensityFitting

if __name__ == '__main__':
    c = pyscf.pbc.gto.Cell()
    c.atom = """
        O   -6.0082242   -6.2662586   -4.5802338
        H   -6.5424905   -5.6103439   -4.0656431
        H   -5.6336920   -6.8738199   -3.8938766
        O   -5.0373186   -5.2694388   -1.2581917
        H   -4.2054186   -5.5931034   -0.8297803
        H   -4.7347234   -4.8522045   -2.1034720
        """
    c.basis = '321g'
    c.a = numpy.eye(3) * 10.0
    c.build()

    for c_isdf in [5, 10, 20, 40, 80]:
        import thc
        t = thc.ISDF(c)
        t.verbose = 0
        t.grids.c_isdf = c_isdf
        # t.grids = pyscf.pbc.dft.gen_grid.BeckeGrids(c)
        # t.grids.level = 0
        # t.grids.verbose = 10
        # t.grids.build()
        t.grids.c_isdf = c_isdf
        t.with_df._cderi_to_save = pyscf.__config__.TMPDIR + "/cderi.h5"
        t.max_memory = 2000
        t.build()

        vv = t.vv
        xip_ao = t.xip

        df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xip_ao, xip_ao)
        df_chol_ref = numpy.zeros_like(df_chol_sol)
        q0 = 0
        for cderi in t.with_df.loop():
            q1 = q0 + cderi.shape[0]
            df_chol_ref[q0:q1] = lib.unpack_tril(cderi)
            q0 = q1

        err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
        err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol) / c.natm
        print("Method = %s, Error = % 6.4e % 6.4e" % ("cholesky", err1, err2))