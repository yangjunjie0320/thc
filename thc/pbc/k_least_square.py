import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

from pyscf.lib.scipy_helper import pivoted_cholesky_python as pivoted_cholesky

import thc
from thc.pbc.least_square import LeastSquareFitting
from thc.pbc.gen_grids import BeckeGrids, UniformGrids

def get_cderi(thc_obj=None, k1_and_k2=None):
    k1, k2 = k1_and_k2
    vk1 = thc_obj.with_df.kpts[k1]
    vk2 = thc_obj.with_df.kpts[k2]

    naux = thc_obj.with_df.get_naoaux()
    nao = thc_obj.cell.nao_nr()

    cderi_real = numpy.zeros((naux, nao * nao))
    cderi_imag = numpy.zeros((naux, nao * nao))

    a0 = a1 = 0
    for _cderi_real, _cderi_imag, sign in thc_obj.with_df.sr_loop([vk1, vk2], blksize=200, compact=False):
        a1 = a0 + cderi_real.shape[0]
        b1 = a0 + _cderi_real[a0:a1].shape[0]

        cderi_real[a0:a1] = _cderi_real[a0:b1]
        cderi_imag[a0:a1] = _cderi_imag[a0:b1]
        a0 = a1

    return (cderi_real + cderi_imag * 1j).reshape(naux, nao, nao)

# for a given combination of k-points, we can compute the LS-THC
def ls_thc_for_kpts(thc_obj=None, k1_and_k2=None):
    k1, k2 = k1_and_k2
    vk1 = kpts[k1]
    vk2 = kpts[k2]

    from pyscf.pbc.lib.kpts_helper import get_kconserv
    vk = thc_obj.with_df.kpts
    nk = vk.shape[0]

    kconserv3 = get_kconserv(thc_obj.cell, vk)
    kconserv2 = kconserv3[:, :, 0].T

    assert thc_obj is not None
    coord = thc_obj.grids.coords
    weigh = thc_obj.grids.weights

    phik1 = thc_obj.eval_gto(coord, weigh, kpt=vk1)
    phik2 = thc_obj.eval_gto(coord, weigh, kpt=vk2)

    zeta1 = (phik1.conj() @ phik1.T) * (phik2 @ phik2.conj().T)
    zeta2 = numpy.einsum("Im,In,Jm,Jn->IJ", phik1.conj(), phik2, phik1, phik2.conj(), optimize=True)
    assert numpy.allclose(zeta1, zeta2)

    zeta = zeta1
    assert numpy.allclose(zeta, zeta.conj().T)

    cderi_ref = get_cderi(thc_obj, k1_and_k2)

    u, s, vh = scipy.linalg.svd(zeta)
    mask = s > 1e-16
    u = u[:, mask]
    s = s[mask]
    vh = vh[mask]

    y = u @ numpy.diag(s) @ vh
    assert numpy.linalg.norm(y - zeta) < 1e-10

    # x = vh.T.conj() @ numpy.diag(1 / s) @ u.conj().T
    x = scipy.linalg.pinv(zeta)

    rhs = numpy.einsum("Qmn,Im,In->QI", cderi_ref, phik1, phik2.conj(), optimize=True)
    z = rhs @ x

    cderi_sol = numpy.einsum("QI,Im,In->Qmn", z, phik1.conj(), phik2, optimize=True)

    err1 = numpy.max(numpy.abs(cderi_ref - cderi_sol))
    err2 = numpy.linalg.norm(cderi_ref - cderi_sol)

    print("k1 = %d, k2 = %d, Max: %6.4e, Mean: %6.4e" % (k1, k2, err1, err2))
    return zeta, z

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
        nk = vk.shape[0]

        kconserv3 = get_kconserv(self.cell, vk)
        kconserv2 = kconserv3[:, :, 0].T
        # kconserv2 = numpy.arange(nk * nk).reshape(nk, nk)
        nq = len(numpy.unique(kconserv2))
        assert nk == nq

        phi_k = self.eval_gto(grids.coords, grids.weights, kpts=vk, kpt=None)
        nk, ng, nao = phi_k.shape
        naux = with_df.get_naoaux()
        
        z_q = numpy.zeros((nq, ng, ng), dtype=phi_k.dtype)
        
        for k1, vk1 in enumerate(vk):
            for k2, vk2 in enumerate(vk):
                q = kconserv2[k1, k2]
                phik1 = phi_k[k1]
                phik2 = phi_k[k2]

                assert phik1.shape == (ng, nao)
                assert phik2.shape == (ng, nao)

                z_q[q] += numpy.einsum("Jm,Jn,Im,In->JI", phik1.conj(), phik2, phik1, phik2.conj(), optimize=True)

        ww = numpy.zeros(ng)
        for q in range(nq):
            assert numpy.allclose(z_q[q], z_q[q].conj().T)

            chol, perm, rank = pivoted_cholesky(z_q[q], tol=1e-16, lower=False)
            ww[perm[:rank]] += abs(numpy.diag(chol)[:rank])

            err = abs(chol[rank - 1, rank - 1])
            log.info("Pivoted Cholesky rank: %d / %d, err = %6.4e", rank, ng, err)
            
        m = (ww > 1e-16)
        nip = m.sum()
        zeta_q = z_q[:, m][:, :, m]
        xipt_k = phi_k[:, m, :]

        assert zeta_q.shape == (nq, nip, nip)
        assert xipt_k.shape == (nk, nip, nao)
        
        rhs = numpy.zeros((nq, naux, nip), dtype=xipt_k.dtype)

        for k1, vk1 in enumerate(vk):
            for k2, vk2 in enumerate(vk):
                q = kconserv2[k1, k2]
                a0 = a1 = 0

                xk1 = xipt_k[k1]
                xk2 = xipt_k[k2]
                for cderi_real, cderi_imag, sign in with_df.sr_loop([vk1, vk2], blksize=200, compact=False):
                    a1 = a0 + cderi_real.shape[0]
                    b1 = rhs[q, a0:a1].shape[0]

                    cderi = cderi_real + cderi_imag * 1j
                    cderi = cderi[:b1].reshape(b1, nao, nao)

                    rhs[q, a0:a1] += numpy.einsum("Qmn,Im,In->QI", cderi, xk1, xk2.conj(), optimize=True)

                    a0 = a1

        coul_q = numpy.zeros((nq, naux, nip), dtype=xipt_k.dtype)
        for q in range(nq):
            u, s, vh = scipy.linalg.svd(zeta_q[q])
            zeta_ = u @ numpy.diag(s) @ vh
            err = numpy.linalg.norm(zeta_ - zeta_q[q])
            assert err < 1e-10

            m = s > 1e-14
            rank = m.sum()

            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank]

            err = s[rank - 1]
            log.info("q = %d, rank = %d / %d, err = %6.4e", q, rank, nip, err)
            zinv = vh.T.conj() @ numpy.diag(1 / s) @ u.conj().T
            coul_q[q] = rhs[q] @ zinv

        for k1, vk1 in enumerate(vk):
            for k2, vk2 in enumerate(vk):
                for k3, vk3 in enumerate(vk):
                    k4 = kconserv3[k1, k2, k3]
                    vk4 = vk[k4]

                    xk1 = xipt_k[k1]
                    xk2 = xipt_k[k2]
                    xk3 = xipt_k[k3]
                    xk4 = xipt_k[k4]

                    q12 = kconserv2[k1, k2]
                    q34 = kconserv2[k3, k4]

                    jq12 = coul_q[q12]
                    jq34 = coul_q[q34]

                    if not q12 == 0:
                        continue

                    eri_ref = with_df.get_eri([vk1, vk2, vk3, vk4], compact=False)
                    eri_ref = eri_ref.reshape(nao, nao, nao, nao)

                    eri_sol = numpy.einsum(
                        "QI,QJ,Im,In,Jk,Jl->mnkl", 
                        jq12, jq34, 
                        xk1.conj(), xk2, 
                        xk3, xk4.conj(), 
                        optimize=True
                    )

                    err1 = numpy.max(numpy.abs(eri_ref - eri_sol))
                    err2 = numpy.linalg.norm(eri_ref - eri_sol)

                    print("k1 = %d, k2 = %d, k3 = %d, k4 = %d, q12 = %d, q34 = %d, Max: %6.4e, Mean: %6.4e" % (k1, k2, k3, k4, q12, q34, err1, err2))
                    assert err1 < 1e-4
                    assert err2 < 1e-4
                # q = kconserv2[k1, k2]
                # j = coul_q[q]
                # xk1 = xipt_k[k1]
                # xk2 = xipt_k[k2]

                # xk1 = xk1.real
                # xk2 = xk2.real

                # a0 = a1 = 0
                # for cderi_real, cderi_imag, sign in with_df.sr_loop([vk1, vk2], blksize=150, compact=False):
                #     a1 = a0 + cderi_real.shape[0]
                #     cderi_ref = cderi_real + cderi_imag * 1j
                #     cderi_ref = cderi_ref[:j[a0:a1].shape[0]].reshape(-1, nao * nao)
                #     cderi_sol = numpy.einsum("QI,Im,In->Qmn", j[a0:a1], xk1.conj(), xk2, optimize=True)
                #     cderi_sol = cderi_sol.reshape(-1, nao * nao)

                #     from numpy import savetxt
                #     savetxt(self.cell.stdout, (cderi_ref.real)[:10, :10], fmt="% 6.4e", delimiter=", ", header="\ncderi_ref")
                #     savetxt(self.cell.stdout, (cderi_sol.real)[:10, :10], fmt="% 6.4e", delimiter=", ", header="\ncderi_sol")

                #     err1 = numpy.max(numpy.abs(cderi_ref - cderi_sol))
                #     err2 = numpy.linalg.norm(cderi_ref - cderi_sol)

                #     print("k1 = %d, k2 = %d, q = %d, [%4d:%4d] Max: %6.4e, Mean: %6.4e" % (k1, k2, q, a0, a1, err1, err2))
                #     a0 = a1

                #     assert err1 < 1e-4

if __name__ == '__main__':
    import pyscf
    from pyscf import pbc
    c = pyscf.pbc.gto.Cell()
    c.atom  = 'He 2.0000 2.0000 2.0000; He 2.0000 2.0000 6.0000'
    c.basis = '321g'
    c.a = numpy.diag([4.0000, 4.0000, 8.0000])
    c.unit = 'bohr'
    c.build()

    kmesh = [2, 2, 4]
    kpts  = c.make_kpts(kmesh)

    thc = WithKPoints(c, kpts=kpts)
    thc.verbose = 10

    thc.with_df._cderi_to_save = "/Users/yangjunjie/Downloads/cderi-224.h5"
    thc.with_df.build()

    thc.grids = UniformGrids(c)
    thc.grids.verbose = 20
    thc.grids.c_isdf = None
    thc.grids.tol    = None
    thc.grids.mesh   = [4,] * 3
    thc.grids.build()

    vk = thc.with_df.kpts
    nk = vk.shape[0]
    nq = nk

    kconserv3 = pyscf.pbc.lib.kpts_helper.get_kconserv(c, vk)
    kconserv2 = kconserv3[:, :, 0].T

    q = 1
    vq = vk[q]

    zeta0 = None
    z0 = None
    for k1, vk1 in enumerate(kpts):
        k2 = kconserv2[q, k1]
        vk2 = kpts[k2]

        if not (k1, k2) in [(0, 3), (1, 0)]:
            continue
        
        print("\nk1 = %d, k2 = %d" % (k1, k2))
        print("vk1 = ", vk1)
        print("vk2 = ", vk2)
        print("vk1 - vk2 = ", vk1 - vk2)
        print("vq  = ", vq)
        # print(numpy.einsum("x,Lx->L", vk1 - vk2 + vq, c.lattice_vectors()))

        # first question is how to recover z1 from different k1 - k2
        zeta1, z1 = ls_thc_for_kpts(thc, k1_and_k2=(k1, k2))

        if z0 is None:
            z0 = z1
            zeta0 = zeta1
            from numpy import savetxt

            # print("\nz0 real")
            # savetxt(c.stdout, z0.real[:10, :10], fmt="% 6.4e", delimiter=", ")

            # print("\nz0 imag")
            # savetxt(c.stdout, z0.imag[:10, :10], fmt="% 6.4e", delimiter=", ")

            print("\nzeta0 real")
            savetxt(c.stdout, zeta0.real[:10, :10], fmt="% 6.4e", delimiter=", ")

            print("\nzeta0 imag")
            savetxt(c.stdout, zeta0.imag[:10, :10], fmt="% 6.4e", delimiter=", ")

        else:
            err = numpy.abs(z1 - z0).max()
            print("err = %6.4e" % err)

            if err > 1e-4:
                # print("\nz1 real")
                # savetxt(c.stdout, z1.real[:10, :10], fmt="% 6.4e", delimiter=", ")

                # print("\nz1 imag")
                # savetxt(c.stdout, z1.imag[:10, :10], fmt="% 6.4e", delimiter=", ")

                print("\nzeta0 real")
                savetxt(c.stdout, zeta1.real[:10, :10], fmt="% 6.4e", delimiter=", ")

                print("\nzeta0 imag")
                savetxt(c.stdout, zeta1.imag[:10, :10], fmt="% 6.4e", delimiter=", ")

                assert 1 == 2