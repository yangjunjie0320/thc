import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

from thc.pbc.gen_grids import InterpolatingPoints
from thc.mol.least_square import cholesky, TensorHyperConractionMixin

class LeastSquareFitting(TensorHyperConractionMixin):
    def __init__(self, mol):
        self.cell = self.mol = mol
        self.with_df = pbc.df.GDF(mol)
        self.grids = InterpolatingPoints(mol)
        self.grids.level = 0

        self.tol = 1e-16
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
        import sys
        phi  = numint.eval_ao(self.mol, grids.coords)
        # numpy.savetxt(sys.stdout, phi[:10, :10], fmt="% 12.8f", delimiter=",")
        phi *= (numpy.abs(grids.weights) ** 0.5)[:, None]
        # numpy.savetxt(sys.stdout, phi[:10, :10], fmt="% 12.8f", delimiter=",")
        # chol, xx = cholesky(phi, tol=self.tol, log=log)

        xx = phi
        print(xx)

        x4 = lib.dot(xx, xx.T) ** 2
        assert numpy.allclose(x4, x4.T)

        u, s, vh = scipy.linalg.svd(x4)
        numpy.savetxt(sys.stdout, x4, fmt="% 6.4e", delimiter=",", header="s")

        x4inv = numpy.linalg.pinv(x4)
        print("x4 = ", x4)
        print("x4inv = ", x4inv)
        print(x4 @ x4inv)
        assert 1 == 2

        import sys
        numpy.savetxt(sys.stdout, x4[:10, :10], fmt="% 12.8f", delimiter=",", header="x4")
        mask = s > 1e-8
        print("Number of significant singular values: %d / %d" % (numpy.sum(mask), len(mask)))
        assert numpy.allclose(u @ numpy.diag(s) @ vh, x4)
        # x4inv = vh[mask].T @ numpy.diag(1.0 / s[mask]) @ u[mask, :].T
        x4inv = scipy.linalg.pinvh(x4)

        x4_dot_x4inv = numpy.diag(s) @ vh @ vh[mask].T @ numpy.diag(1.0 / s[mask])
        assert numpy.allclose(vh @ vh.T, numpy.eye(vh.shape[0]))
        assert numpy.allclose(u @ u.T, numpy.eye(u.shape[0]))
        assert numpy.allclose(x4_dot_x4inv, numpy.eye(x4_dot_x4inv.shape[0])[:, :3])
       

        numpy.savetxt(sys.stdout, x4_dot_x4inv[:20, :20], fmt="% 12.8f", delimiter=",", header="x4 @ x4inv")

        numpy.savetxt(sys.stdout, u[:10, :10], fmt="% 12.8f", delimiter=",", header="u")

        print((u @ x4_dot_x4inv).shape, (u[:3]).shape)
        x4_dot_x4inv = u @ x4_dot_x4inv @ (u[:, :3]).T
        
        numpy.savetxt(sys.stdout, u[:10, :10], fmt="% 12.8f", delimiter=",", header="u")
        numpy.savetxt(sys.stdout, x4_dot_x4inv[:10, :10], fmt="% 12.8f", delimiter=",", header="x4 @ x4inv")
        assert 1 == 2

        # x4_dot_x4inv = x4inv @ x4
        # numpy.savetxt(sys.stdout, x4_dot_x4inv[:10, :10], fmt="% 12.8f", delimiter=",", header="x4 @ x4inv")
        # assert 1 == 2

        # nip, nao  = xx.shape
        # cput1 = logger.timer(self, "interpolating vectors", *cput0)

        # # Build the coulomb kernel
        # # coul = numpy.einsum("Qmn,Im,In->QI", cderi, xip, xip)
        # # coul = numpy.zeros((naux, nip))
        # cderi_full = numpy.zeros((naux, nao, nao))
        # blksize = int(self.max_memory * 1e6 * 0.5 / (8 * nao ** 2))
        # blksize = max(4, blksize)

        # a0 = a1 = 0 # slice for auxilary basis
        # for cderi in with_df.loop(blksize=blksize):
        #     a1 = a0 + cderi.shape[0]

        #     for i0, i1 in lib.prange(0, nip, blksize): # slice for interpolating vectors
        #         # TODO: sum over only the significant shell pairs
        #         # cput = (logger.process_clock(), logger.perf_counter())

        #         # ind = numpy.arange(nao)
        #         # x2  = xx[i0:i1, :, numpy.newaxis] * xx[i0:i1, numpy.newaxis, :]
        #         # x2  = lib.pack_tril(x2 + x2.transpose(0, 2, 1))
        #         # x2[:, ind * (ind + 1) // 2 + ind] *= 0.5
        #         # coul[a0:a1, i0:i1] += pyscf.lib.dot(cderi, x2.T)
        #         # logger.timer(self, "coulomb kernel [%4d:%4d, %4d:%4d]" % (a0, a1, i0, i1), *cput)
        #         # x2 = None
                
        #         cderi_full[a0:a1] = lib.unpack_tril(cderi)
        #         print(lib.unpack_tril(cderi).dtype)

        #     a0 = a1

        # cput1 = logger.timer(self, "coulomb kernel", *cput1)
        # print(cderi_full.shape)
        # coul = numpy.einsum("Qmn,Im,In->QI", cderi_full, xx, xx, optimize=True)
        # vv = coul @ x4inv
        # vv = scipy.linalg.solve_triangular(chol, coul.T, lower=False).T
        # assert 1 == 2

        # ww = scipy.linalg.solve_triangular(chol.T, coul.T, lower=True).T
        # vv = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T
        # cput1 = logger.timer(self, "solving linear equations", *cput1)
        # logger.timer(self, "LS-THC", *cput0)

        # self.vv = vv
        # self.xx = xx
        # return vv, xx

LS = LeastSquareFitting

if __name__ == '__main__':
    c = pyscf.pbc.gto.Cell()
    c.atom = """
    He 2.000000 2.000000 2.000000
    He 2.000000 2.000000 4.000000
    """
    c.a = numpy.diag([4, 4, 6])
    c.basis = "sto3g"
    c.verbose = 0
    c.build()

    import thc
    thc = thc.LS(c)
    thc.verbose = 6
    thc.tol = 1e-10
    thc.grids.c_isdf = 2

    thc.with_df.build()
    thc.with_df._cderi = "/Users/yangjunjie/Downloads/cderi.h5"
    thc.max_memory = 2000
    thc.build()

    # vv = thc.vv
    # xx = thc.xx

    # from pyscf.lib import unpack_tril
    # df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xx, xx, optimize=True)

    # df_chol_ref = numpy.zeros_like(df_chol_sol)
    # a0 = a1 = 0 # slice for auxilary basis
    # for cderi in thc.with_df.loop(blksize=20):
    #     a1 = a0 + cderi.shape[0]
    #     df_chol_ref[a0:a1] = unpack_tril(cderi)

    # err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
    # err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol) / c.natm
    # print("Method = %s, Error = % 6.4e % 6.4e" % ("cholesky", err1, err2))
