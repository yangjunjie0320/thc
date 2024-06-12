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
        
        from pyscf.pbc.lib.kpts_helper import get_kconserv
        vk = self.with_df.kpts
        kconserv3 = get_kconserv(self.cell, vk)
        kconserv2 = kconserv3[:, :, 0].T
        nk = vk.shape[0]
        nq = len(numpy.unique(kconserv2))
        assert nk == nq

        phi0 = self.eval_gto(grids.coords, grids.weights, kpt=self.with_df.kpts[0])
        ng, nao = phi0.shape
        naux = with_df.get_naoaux()

        zeta0 = lib.dot(phi0, phi0.conj().T) ** 2
        zeta0 = zeta0.real
        chol, perm, rank = lib.scipy_helper.pivoted_cholesky(zeta0, tol=self.tol, lower=False)
        nip = rank

        perm = perm[:nip]
        chol = chol[:nip, :nip]
        err  = abs(chol[nip - 1, nip - 1])
        log.info("Pivoted Cholesky rank: %d / %d, err = %6.4e", rank, ng, err)

        xipt_k = self.eval_gto(grids.coords[perm], grids.weights[perm], kpts=vk, kpt=None)
        assert xipt_k.shape == (nk, nip, nao)

        coul_q = numpy.zeros((nq, naux, nip), dtype=xipt_k.dtype)
        for q in range(nq):
            vq = vk[q]

            zq = numpy.zeros((nip, nip),  dtype=xipt_k.dtype)
            jq = numpy.zeros((naux, nip), dtype=xipt_k.dtype)

            for k1, vk1 in enumerate(vk):
                k2 = kconserv2[q, k1]
                vk2 = vk[k2]

                z1 = lib.dot(xipt_k[k1], xipt_k[k1].conj().T)
                z2 = lib.dot(xipt_k[k2], xipt_k[k2].conj().T)
                zq += z1 * z2.conj()

                a0 = a1 = 0
                for cderi_real, cderi_imag, sign in with_df.sr_loop([vk1, vk2], blksize=20, compact=False):
                    a1 = a0 + cderi_real.shape[0]
                    cderi = cderi_real + cderi_imag * 1j
                    cderi = cderi[:jq[a0:a1].shape[0]].reshape(-1, nao, nao)
                    jq[a0:a1] += numpy.einsum("Qmn,Im,In->QI", cderi, xipt_k[k1].conj(), xipt_k[k2], optimize=True)

                    # print("k1 = %d, k2 = %d, q = %d, [%4d:%4d]" % (k1, k2, q, a0, a1))
                    a0 = a1

            coul_q[q] = numpy.dot(jq, scipy.linalg.pinv(zq))
            print(coul_q[q].shape)

        for k1, vk1 in enumerate(vk):
            for k2, vk2 in enumerate(vk):
                q = kconserv2[k1, k2]

                a0 = a1 = 0
                for cderi_real, cderi_imag, sign in with_df.sr_loop([vk1, vk2], blksize=20, compact=False):
                    a1 = a0 + cderi_real.shape[0]
                    cderi_ref = cderi_real + cderi_imag * 1j
                    cderi_ref = cderi_ref.reshape(-1, nao, nao)

                    print(coul_q[q, a0:a1].shape)
                    cderi_sol = numpy.einsum("QI,Im,In->Qmn", coul_q[q, a0:a1], xipt_k[k1], xipt_k[k2], optimize=True)
                    print(cderi_sol.shape)

                    err1 = numpy.max(numpy.abs(cderi_ref - cderi_sol))
                    err2 = numpy.linalg.norm(cderi_ref - cderi_sol)

                    print("k1 = %d, k2 = %d, q = %d, [%4d:%4d] Max: %6.4e, Mean: %6.4e" % (k1, k2, q, a0, a1, err1, err2))

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

    kmesh = [2, 2, 2]
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
