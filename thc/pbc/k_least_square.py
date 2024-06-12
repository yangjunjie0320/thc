import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

import thc
from thc.pbc.least_square import LeastSquareFitting
from thc.pbc.gen_grids import BeckeGrids

class WithKPoints(LeastSquareFitting):
    def __init__(self, cell, kpts=None):
        assert kpts is not None
        self.cell = self.mol = cell
        self.with_df = pbc.df.GDF(cell, kpts=kpts)
        self.grids = BeckeGrids(cell)
        self.grids.level = 0
        self.max_memory = cell.max_memory

    def eval_gto(self, coords, weights, kpt=None, kpts=None):
        phi = self.cell.pbc_eval_gto("GTOval", coords, kpt=kpt, kpts=kpts)
        phi = numpy.array(phi)
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

        if with_df._cderi is None:
            log.info('\n')
            with_df.build()

        self.dump_flags()
        cput0 = (logger.process_clock(), logger.perf_counter())

        nq = nk = len(self.with_df.kpts)
        phi0 = self.eval_gto(grids.coords, grids.weights, kpt=self.with_df.kpts[0])
        zeta0 = lib.dot(phi0, phi0.conj().T) ** 2
        ng, nao = phi0.shape
        chol, perm, rank = lib.scipy_helper.pivoted_cholesky(zeta0, tol=self.tol, lower=False)
        nip = rank

        perm = perm[:nip]
        chol = chol[:nip, :nip]
        err  = abs(chol[nip - 1, nip - 1])
        log.info("Pivoted Cholesky rank: %d / %d, err = %6.4e", rank, ng, err)

        xipt_k = self.eval_gto(grids.coords[perm], grids.weights[perm], kpts=self.with_df.kpts)
        assert xipt_k.shape == (nk, nip, nao)

        zeta_q = numpy.zeros((nq, nip, nip), dtype=zeta0.dtype)

        for k1, vk1 in enumerate(self.with_df.kpts):
            for k2, vk2 in enumerate(self.with_df.kpts):
                q = kcons[k1, k2]
                vq = self.with_df.kpts[q]

                zeta1 = lib.dot(xipt_k[k1], xipt_k[k1].conj().T)
                zeta2 = lib.dot(xipt_k[k2], xipt_k[k2].conj().T)
                zeta_q[q] = zeta1 * zeta2.conj()


        # for ik, kpt in enumerate(self.with_df.kpts):
        #     phi_k  = self.eval_gto(grids.coords, grids.weights, kpt=kpt)
        #     zeta_k = lib.dot(phi_k, phi_k.conj().T)
        #     zeta_k *= zeta_k.conj()
        #     ng, nao = phi_k.shape

        #     if perm is None:
        #         chol, perm, rank = lib.scipy_helper.pivoted_cholesky(zeta_k, tol=self.tol, lower=False)
        #         nip = rank
            
        #         perm = perm[:nip]
        #         chol = chol[:nip, :nip]
        #         err  = abs(chol[nip - 1, nip - 1])
        #         log.info("Pivoted Cholesky rank: %d / %d, err = %6.4e", rank, ng, err)

        #     xipt = phi_k[perm]
        #     cput1 = logger.timer(self, "interpolating vectors", *cput0)

        # # # Build the coulomb kernel
        # # # rhs = numpy.einsum("Qmn,Im,In->QI", cderi, xip, xip)
        #     naux = with_df.get_naoaux()
        #     rhs = numpy.zeros((naux, nip))

        # # # TODO: Is this correct?
        #     blksize = int(self.max_memory * 1e6 * 0.5 / (8 * nao ** 2))
        #     blksize = max(4, blksize)

        #     a0 = a1 = 0 # slice for auxilary basis
        #     for cderi in with_df.loop(blksize=blksize):
        #         a1 = a0 + cderi.shape[0]
        #         for i0, i1 in lib.prange(0, nip, blksize): # slice for interpolating vectors
        #             # TODO: sum over only the significant shell pairs
        #             cput = (logger.process_clock(), logger.perf_counter())

        #             ind = numpy.arange(nao)
        #             x2  = xipt[i0:i1, :, numpy.newaxis] * xipt[i0:i1, numpy.newaxis, :]
        #             x2  = lib.pack_tril(x2 + x2.transpose(0, 2, 1))
        #             x2[:, ind * (ind + 1) // 2 + ind] *= 0.5
        #             rhs[a0:a1, i0:i1] += lib.dot(cderi, x2.T)
        #             logger.timer(self, "RHS [%4d:%4d, %4d:%4d]" % (a0, a1, i0, i1), *cput)
        #             x2 = None
        #         a0 = a1

        # cput1 = logger.timer(self, "RHS", *cput1)

        # ww = scipy.linalg.solve_triangular(chol.T, rhs.T, lower=True).T
        # coul = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T
        
        # cput1 = logger.timer(self, "solving linear equations", *cput1)
        # logger.timer(self, "LS-THC", *cput0)

        # self.coul = coul
        # self.xipt = xipt
        # return coul, xipt

if __name__ == '__main__':
    import pyscf
    from pyscf import pbc
    c = pyscf.pbc.gto.Cell()
    c.atom = '''C     0.0000  0.0000  0.0000
                C     1.6851  1.6851  1.6851'''
    c.basis = 'sto3g'
    c.a = '''0.0000, 3.3701, 3.3701
             3.3701, 0.0000, 3.3701
             3.3701, 3.3701, 0.0000'''
    c.unit = 'bohr'
    c.build()

    kmesh = [4, 4, 4]
    kpts  = c.make_kpts(kmesh)

    thc = WithKPoints(c, kpts=kpts)
    thc.with_df._cderi = "/Users/yangjunjie/Downloads/gdf.h5"
    thc.verbose = 6

    thc.grids.c_isdf = 20
    thc.max_memory = 2000
    thc.build()

    # grids = pbc.dft.gen_grid.UniformGrids(c)
    # grids.atom_grid = (10, 86)
    # coord = thc.grids.coords
    # weigh = thc.grids.weights 
    # from pyscf.pbc.dft.gen_grid import get_becke_grids
    # from pyscf.pbc.dft.gen_grid import get_uniform_grids

    # mesh = [10, 10, 10]
    # coord = get_uniform_grids(c, mesh=mesh)
    # weigh = c.vol / numpy.prod(mesh) * numpy.ones(numpy.prod(mesh))

    # phi_k = thc.eval_gto(coord, weigh, kpts=kpts)
    # phi_k = numpy.array(phi_k)

    # ovlp_k_ref = thc.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)
    # ovlp_k_sol = numpy.einsum("kxm,kxn->kmn", phi_k.conj(), phi_k)

    # err = numpy.max(numpy.abs(ovlp_k_ref - ovlp_k_sol))
    # print("npts = %d, err = %g" % (phi_k.shape[0], err))
