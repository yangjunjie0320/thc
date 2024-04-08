import pyscf.pbc
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

        ovlp_diag = numpy.diag(self.cell.pbc_intor("int1e_ovlp"))

        naux = with_df.get_naoaux()
        phi = self.cell.pbc_eval_gto("GTOval", grids.coords)
        phi *= (numpy.abs(grids.weights) ** 0.5)[:, None]
        phi *= 1 / numpy.sqrt(ovlp_diag[None, :])

        chol, xx = cholesky(phi, tol=1e-8, log=log)
        nip, nao  = xx.shape
        cput1 = logger.timer(self, "interpolating vectors", *cput0)

        rho_ref = numpy.einsum("gm,gn->gmn", phi, phi)


        zeta = lib.dot(phi, phi.T)
        print(numpy.abs(zeta).min(), numpy.abs(zeta).max())

        wsqrt = numpy.abs(grids.weights) ** 0.5
        print(wsqrt.min(), wsqrt.max())
        assert 1 == 2
 
        from pyscf.lib.scipy_helper import pivoted_cholesky
        chol, perm, rank = pivoted_cholesky(zeta, lower=False, tol=1e-12)
        mask = perm # [:400]

        xx = phi[mask]
        nip = xx.shape[0]
        chol = chol[:nip, :nip]

        df_chol_ref = numpy.zeros((naux, nao, nao))
        a0 = a1 = 0 # slice for auxilary basis
        from pyscf.lib import pack_tril, unpack_tril
        for cderi in with_df.loop(blksize=20):
            a1 = a0 + cderi.shape[0]
            df_chol_ref[a0:a1] = unpack_tril(cderi)

 
        # u, s, vh = scipy.linalg.svd(x2)
        # numpy.savetxt(self.cell.stdout, s, fmt="% 6.4e", header="Singular values", delimiter=",")
        # assert 1 == 2

        rhs = pack_tril(df_chol_ref)
        # print("rhs = ", rhs.shape)
        # print("x2 = ", x2.shape)
        # a = scipy.linalg.lstsq(x2.T, rhs.T)[0]
        # print("a = ", a.shape)
        # err = numpy.linalg.norm(rhs.T - x2.T @ a)
        # print("Error = % 6.4e" % err)

        x2 = numpy.einsum("Im,In->Imn", xx, xx)
        x2 = lib.pack_tril(x2)
        x2inv = scipy.linalg.pinv(x2)
    
        a = (rhs @ x2inv)
        err = numpy.linalg.norm(rhs - a @ x2)
        print("Error = % 6.4e" % err)

        # zz = zeta[mask][:, mask]
        # zzinv, rank = scipy.linalg.pinv(zz, return_rank=True)
        # theta = zzinv @ zeta[mask]

        # rho_sol = numpy.einsum("Ig,Im,In->gmn", theta, xx, xx)

        # err = numpy.linalg.norm(rho_ref - rho_sol)
        # print("Error = % 6.4e" % err)

        # coul = numpy.zeros((naux, nip))
        # blksize = int(self.max_memory * 1e6 * 0.5 / (8 * nao ** 2))
        # blksize = max(4, blksize)

        # a0 = a1 = 0 # slice for auxilary basis
        # for cderi in with_df.loop(blksize=blksize):
        #     a1 = a0 + cderi.shape[0]

        #     for i0, i1 in lib.prange(0, nip, blksize): # slice for interpolating vectors
        #         # TODO: sum over only the significant shell pairs
        #         cput = (logger.process_clock(), logger.perf_counter())

        #         ind = numpy.arange(nao)
        #         x2  = xx[i0:i1, :, numpy.newaxis] * xx[i0:i1, numpy.newaxis, :]
        #         x2  = lib.pack_tril(x2 + x2.transpose(0, 2, 1))
        #         x2[:, ind * (ind + 1) // 2 + ind] *= 0.5
        #         coul[a0:a1, i0:i1] += pyscf.lib.dot(cderi, x2.T)
        #         logger.timer(self, "coulomb kernel [%4d:%4d, %4d:%4d]" % (a0, a1, i0, i1), *cput)
        #         x2 = None

        #     a0 = a1

        # cput1 = logger.timer(self, "coulomb kernel", *cput1)

        # ww = scipy.linalg.solve_triangular(chol.T, coul.T, lower=True).T
        # vv = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T
        # cput1 = logger.timer(self, "solving linear equations", *cput1)
        # logger.timer(self, "LS-THC", *cput0)

        self.vv = a
        self.xx = xx
        return self.vv, self.xx

LS = LeastSquareFitting

if __name__ == '__main__':
    c = pyscf.pbc.gto.Cell()
    c.atom = '''C     0.      0.      0.    
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.    
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751'''
    c.basis = 'gth-szv'
    c.pseudo = 'gth-pade'
    c.a = numpy.eye(3) * 3.5668
    c.unit = 'aa'
    c.build()

    import thc
    thc = thc.LS(c)
    thc.verbose = 6
    thc.tol = 1e-10
    thc.grids.c_isdf = 10
    thc.max_memory = 2000

    # thc.grids.coords = c.gen_uniform_grids([10, ] * 3)
    # thc.grids.weights = (lambda ng: numpy.ones(ng) * c.vol/ng)(thc.grids.coords.shape[0])
    thc.with_df._cderi = "/Users/yangjunjie/Downloads/gdf.h5"
    thc.build()

    vv = thc.vv
    xx = thc.xx

    print("vv = ", vv.shape)
    print("xx = ", xx.shape)

    from pyscf.lib import pack_tril, unpack_tril
    df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xx, xx, optimize=True)

    df_chol_ref = numpy.zeros_like(df_chol_sol)
    a0 = a1 = 0 # slice for auxilary basis
    for cderi in thc.with_df.loop(blksize=20):
        a1 = a0 + cderi.shape[0]
        df_chol_ref[a0:a1] = unpack_tril(cderi)

    df_chol_ref = pack_tril(df_chol_ref)
    # u, s, vh = scipy.linalg.svd(df_chol_ref)
    # numpy.savetxt(c.stdout, s, fmt="% 6.4e", header="Singular values", delimiter=",")

    df_chol_sol = pack_tril(df_chol_sol)

    # print("df_chol_ref = ", df_chol_ref.shape)
    # numpy.savetxt(c.stdout, df_chol_ref[:20, :10], fmt="% 8.4e", header="Reference Cholesky factor", delimiter=",")

    # print("df_chol_sol = ", df_chol_sol.shape)
    # numpy.savetxt(c.stdout, df_chol_sol[:20, :10], fmt="% 8.4e", header="Solution Cholesky factor", delimiter=",")


    err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
    err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol) / c.natm
    print("Method = %s, Error = % 6.4e % 6.4e" % ("cholesky", err1, err2))
