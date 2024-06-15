import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

import thc
from thc.pbc.least_square import LeastSquareFitting
from thc.pbc.gen_grids import BeckeGrids, UniformGrids

class WithKPoints(LeastSquareFitting):
    def __init__(self, cell, kpts=None):
        assert kpts is not None
        self.cell = self.mol = cell
        self.with_df = pbc.df.GDF(cell, kpts=kpts)
        self.grids = BeckeGrids(cell)
        self.grids.level = 0
        self.max_memory = cell.max_memory

    def eval_gto(self, coords, weights, kpt=None, kpts=None):
        nao = self.cell.nao_nr()
        ng = len(weights)

        if kpt is None and kpts is not None:
            nk = len(kpts)
            phi = self.cell.pbc_eval_gto("GTOval", coords, kpt=None, kpts=kpts)
            phi = numpy.array(phi)
            assert phi.shape == (nk, ng, nao), phi.shape
            return numpy.einsum("kxm,x->kxm", phi, numpy.abs(weights) ** 0.5)
        
        elif kpt is not None and kpts is None:
            phi = self.cell.pbc_eval_gto("GTOval", coords, kpt=kpt, kpts=None)
            phi = numpy.array(phi)
            assert phi.shape == (ng, nao)
            return numpy.einsum("xm,x->xm", phi, numpy.abs(weights) ** 0.5)
        
        else:
            raise RuntimeError("kpt and kpts cannot be None simultaneously")
    
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
        assert numpy.linalg.norm(phi0.imag) < 1e-10
        phi0 = phi0.real

        ng, nao = phi0.shape
        naux = with_df.get_naoaux()

        # It's a question why we need even more points 
        # than the full rank. The reason might be different
        # k-points have different important grids.
        # We shoud try to use more interpolating points
        # to get better accuracy. Anyway, we should check it.
        # Looks like the best strategy is to c_isdf as the proirity
        # and then use the tol to control the accuracy.

        # We should 1. further 
        # zeta0 = lib.dot(phi0, phi0.T) ** 2
        # chol, perm, rank = lib.scipy_helper.pivoted_cholesky(zeta0, tol=1e-16, lower=False)
        # nip = ng # rank

        # perm = perm[:nip]
        # chol = chol[:nip, :nip]
        # err  = abs(chol[nip - 1, nip - 1])
        
        phi_k = self.eval_gto(grids.coords, grids.weights, kpts=vk, kpt=None)
        nk, ng, nao = phi_k.shape
        
        z_q = numpy.zeros((nq, ng, ng), dtype=phi_k.real.dtype)
        for k1, vk1 in enumerate(vk):
            for k2, vk2 in enumerate(vk):
                q = kconserv2[k1, k2]
                z1 = numpy.dot(phi_k[k1], phi_k[k1].conj().T)
                z2 = numpy.dot(phi_k[k2], phi_k[k2].conj().T)
                z12 = z1.conj() * z2
                z_q[q] += z12.real

        mask = []
        chols = []
        for q in range(nq):
            chol, perm, rank = lib.scipy_helper.pivoted_cholesky(z_q[q], tol=1e-16, lower=False)
            err = abs(chol[rank - 1, rank - 1])
            mask += list(perm[:rank])
            chols.append(chol)

            log.info("Pivoted Cholesky rank: %d / %d, err = %6.4e", rank, ng, err)

        mask = numpy.sort(numpy.unique(mask))
        nip = len(mask)
        
        zeta_q = z_q[:, mask][:, :, mask]

        # after we get everything, we select the
        # correct interpolating points and then
        # solve the least square fitting problem.
        # chols_new = []
        # for q in range(nq):
        #     chol = chols[q]
        #     chols_new.append(chol[mask][:, mask])

        xipt_k = phi_k[:, mask, :]
        # zeta_q = numpy.zeros((nq, nip, nip), dtype=xipt_k.dtype)
        coul_q = numpy.zeros((nq, naux, nip), dtype=xipt_k.dtype)
        jq = numpy.zeros((nq, naux, nip), dtype=xipt_k.dtype)

        for k1, vk1 in enumerate(vk):
            for k2, vk2 in enumerate(vk):
                q = kconserv2[k1, k2]
                # z1 = numpy.dot(xipt_k[k1], xipt_k[k1].conj().T)
                # z2 = numpy.dot(xipt_k[k2], xipt_k[k2].conj().T)
                # zeta_q[q] += z1 * z2.conj()

                a0 = a1 = 0
                for cderi_real, cderi_imag, sign in with_df.sr_loop([vk1, vk2], blksize=200, compact=False):
                    a1 = a0 + cderi_real.shape[0]
                    b1 = jq[q, a0:a1].shape[0]

                    cderi = cderi_real + cderi_imag * 1j
                    cderi = cderi[:b1].reshape(b1, nao, nao)

                    jq[q, a0:a1] += numpy.einsum("Qmn,Im,In->QI", cderi, xipt_k[k1].conj(), xipt_k[k2], optimize=True)

        for q in range(nq):
            u, s, vh = scipy.linalg.svd(zeta_q[q])
            mask = s > 1e-14
            rank = mask.sum()

            print("s = \n", s)

            err = s[rank - 1]
            log.info("q = %d, rank = %d / %d, err = %6.4e", q, rank, nip, err)
            zinv = u[:, mask] @ numpy.diag(1 / s[mask]) @ vh[mask]
            coul_q[q] = numpy.dot(jq[q], zinv)

        for k1, vk1 in enumerate(vk):
            for k2, vk2 in enumerate(vk):
                # if not (k1 == 0 and k2 == 0):
                #     break
                
                q = kconserv2[k1, k2]
                jq = coul_q[q]
                # assert jq.imag.max() < 1e-10
                xk1 = xipt_k[k1]
                xk2 = xipt_k[k2]
                assert xk1.imag.max() < 1e-10
                assert xk2.imag.max() < 1e-10

                xk1 = xk1.real
                xk2 = xk2.real

                a0 = a1 = 0
                for cderi_real, cderi_imag, sign in with_df.sr_loop([vk1, vk2], blksize=150, compact=False):
                    a1 = a0 + cderi_real.shape[0]
                    cderi_ref = cderi_real + cderi_imag * 1j
                    cderi_ref = cderi_ref[:jq[a0:a1].shape[0]].reshape(-1, nao * nao)
                    cderi_sol = numpy.einsum("QI,Im,In->Qmn", jq[a0:a1], xk1, xk2.conj(), optimize=True)
                    cderi_sol = cderi_sol.reshape(-1, nao * nao)

                    err1 = numpy.max(numpy.abs(cderi_ref - cderi_sol))
                    err2 = numpy.linalg.norm(cderi_ref - cderi_sol)

                    print("k1 = %d, k2 = %d, q = %d, [%4d:%4d] Max: %6.4e, Mean: %6.4e" % (k1, k2, q, a0, a1, err1, err2))
                    a0 = a1

if __name__ == '__main__':
    import pyscf
    from pyscf import pbc
    c = pyscf.pbc.gto.Cell()
    c.atom  = 'He 2.0000 2.0000 2.0000; He 2.0000 2.0000 4.0000'
    c.basis = '321g'
    c.a = numpy.diag([4.0000, 4.0000, 6.0000])
    c.unit = 'bohr'
    c.build()

    kmesh = [2, 2, 2]
    kpts  = c.make_kpts(kmesh)
    # print(kpts)

    thc = WithKPoints(c, kpts=kpts)
    thc.verbose = 10

    thc.grids = BeckeGrids(c)
    thc.grids.verbose = 20
    thc.grids.c_isdf = None
    thc.grids.tol = None
    thc.grids.level = 0
    thc.build()
    assert 1 == 2

    coord0 = numpy.array(thc.grids.coords)
    weigh0 = numpy.array(thc.grids.weights)
    ng = len(weigh0)
    nk = len(kpts)
    nao = c.nao_nr()

    phik = thc.eval_gto(coord0, weigh0, kpts=kpts)
    assert phik.shape == (nk, ng, nao)

    # ovlp_k_ref = c.pbc_intor("int1e_ovlp", kpts=kpts)
    # ovlp_k_ref = numpy.array(ovlp_k_ref)
    # assert ovlp_k_ref.shape == (nk, nao, nao)

    # rho_k1_k2 = numpy.einsum("kxm,lxn->klmnx", phik.conj(), phik)
    # assert rho_k1_k2.shape == (nk, nk, nao, nao, ng)
    # ovlp_k_sol = numpy.einsum("kkmnx->kmn", rho_k1_k2)
    # assert ovlp_k_sol.shape == (nk, nao, nao)
    # err = numpy.linalg.norm(ovlp_k_ref - ovlp_k_sol)
    # assert err < 1e-4, err

    kconserv3 = pyscf.pbc.lib.kpts_helper.get_kconserv(c, kpts)
    kconserv2 = kconserv3[:, :, 0].T
    nq = len(numpy.unique(kconserv2))
    vk = kpts

    assert nk == nq
    zeta = numpy.zeros((nq, ng, ng), dtype=phik.dtype)
    for k1, vk1 in enumerate(vk):
        for k2, vk2 in enumerate(vk):
            q = kconserv2[k1, k2]
            z1 = numpy.dot(phik[k1], phik[k1].conj().T)
            z2 = numpy.dot(phik[k2], phik[k2].conj().T)
            zeta[q] += z1 * z2


    for q in range(nq):
        u, s, vh = scipy.linalg.svd(zeta[q])
        mask = s > 1e-16
        rank = mask.sum()
        err = s[rank - 1]
        print("q = %d, rank = %d / %d, err = %6.4e" % (q, rank, ng, err))

    