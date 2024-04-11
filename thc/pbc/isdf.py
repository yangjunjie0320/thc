import pyscf.pbc
import numpy, scipy
import scipy.linalg

import pyscf
from pyscf import pbc
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.lib import logger

import thc
from thc.mol.least_square import TensorHyperConractionMixin
from thc.pbc.gen_grids import InterpolatingPoints


class InterpolativeSeparableDensityFitting(TensorHyperConractionMixin):
    def __init__(self, mol):
        self.cell = self.mol = mol
        self.mesh = self.cell.mesh

        self.grids = InterpolatingPoints(mol)
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
        phi = self.cell.pbc_eval_gto("GTOval", coords)
        phi *= (numpy.abs(weights) ** 0.5)[:, None]
        return phi
    
    def build(self, kpts=None):
        assert kpts is None

        log = logger.Logger(self.stdout, self.verbose)
        self.grids.verbose = self.verbose

        if self.grids.coords is None:
            log.info('\n******** %s ********', self.grids.__class__)
            self.grids.dump_flags()
            self.grids.build()
        grids = self.grids

        self.dump_flags()

        mesh = self.mesh
        ng = numpy.prod(mesh)

        coord = self.cell.gen_uniform_grids(mesh)
        weigh = self.cell.vol / ng * numpy.ones(ng)

        # calculate the distance among coords and grids.coords
        # dist = numpy.sum((coord[:, numpy.newaxis, :] - grids.coords[numpy.newaxis, :, :]) ** 2, axis=2)
        # mask = numpy.argmin(dist, axis=0)
        # mask = numpy.unique(mask)
        mask = numpy.arange(ng)

        phi = self.eval_gto(coord, weigh)
        zeta = lib.dot(phi[mask], phi[mask].T) ** 2
        chol, perm, rank = lib.scipy_helper.pivoted_cholesky(zeta.real, tol=self.tol, lower=False)
        nip = rank
        
        perm = perm[:nip]
        chol = chol[:nip, :nip]
        err  = abs(chol[nip - 1, nip - 1])
        log.info("Pivoted Cholesky rank: %d / %d, err = %6.4e", rank, len(mask), err)

        mask = mask[perm]

        xx = phi[mask]
        z = lib.dot(xx, phi.T) ** 2
        cc, rank = scipy.linalg.pinv(z[:, mask], return_rank=True)
        theta = cc @ z

        y = numpy.fft.fftn(theta.reshape(-1, *mesh), axes=(1, 2, 3)).reshape(-1, ng)

        from pyscf.pbc.tools import get_coulG
        g2inv = get_coulG(self.cell, mesh=mesh)
        g2inv *= self.cell.vol / ng ** 2
        coul = numpy.einsum('ug,g,vg->uv', y.conj(), g2inv, y, optimize=True)

        self.coul = coul
        self.vipt = xx
        return coul, xx



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
    c.mesh = [20, 20, 20]
    c.build()

    import thc
    thc = thc.ISDF(c)
    thc.verbose = 6
    thc.grids.c_isdf = 40
    thc.max_memory = 2000
    thc.build()

    vv = thc.coul
    xx = thc.vipt

    import pyscf.pbc.df

    df_obj = pyscf.pbc.df.FFTDF(c)
    df_obj.mesh = c.mesh
    df_obj.build()

    eri_ref = df_obj.get_eri(compact=False)

    eri_sol = numpy.einsum("IJ,Im,In,Jk,Jl->mnkl", vv, xx, xx, xx, xx, optimize=True)
    eri_sol = eri_sol.reshape(*eri_ref.shape)

    print(eri_ref[:10, :10])
    print(eri_sol[:10, :10])

    err = numpy.abs(eri_sol - eri_ref).max()
    print(err)