import numpy, scipy
import scipy.linalg
from scipy.sparse import dok_array

import pyscf
from pyscf import df
from pyscf import lib
from pyscf.dft import numint
from pyscf.lib import logger

from thc.gen_grids import InterpolatingPoints

def build_rho(phi=None, tol=1e-8):
    ng, nao = phi.shape
    rho = numpy.einsum("Im,In->Imn", phi, phi)
    rho = lib.pack_tril(rho)

    # This part remains similar; identifying non-zero elements
    mask = numpy.where(numpy.abs(phi) > numpy.sqrt(tol))

    for ig, mu in zip(*mask):
        rho_g_mu = phi[ig, mu] * phi[ig, :(mu+1)]
        rho_g_mu[-1] *= 0.5

        # TODO: Vectorize or optimize this loop
        munu = mu * (mu + 1) // 2 + numpy.arange(mu+1)
        ix   = numpy.abs(rho_g_mu) > tol
        rho[ig, munu[ix]] = rho_g_mu[ix]

    return rho

def cholesky(phi, tol=1e-8, log=None):
    ngrid, nao = phi.shape
    rho = numpy.einsum("Im,In->Imn", phi, phi)
    rho = lib.pack_tril(rho)

    ss  = rho.dot(rho.T)
    ss += ss.T

    from scipy.linalg.lapack import dpstrf
    chol, perm, rank, info = dpstrf(ss, tol=tol) 

    nisp = rank
    if log is not None:
        log.info("Cholesky: rank = %d / %d" % (rank, ngrid))

    perm = (numpy.array(perm) - 1)[:nisp]

    tril = numpy.tril_indices(nisp, k=-1)
    chol = chol[:nisp, :nisp]
    chol[tril] *= 0.0
    visp = phi[perm]
    return chol, visp


class TensorHyperConractionMixin(lib.StreamObject):
    tol = 1e-4
    max_memory = 1000
    verbose = 5

class InterpolativeSeparableDensityFitting(TensorHyperConractionMixin):
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
        chol, visp = cholesky(phi, tol=1e-12, log=log)
        cput1 = logger.timer(self, "interpolating vectors", *cput0)

        nisp, nao  = visp.shape
        risp = numpy.einsum("Im,In->Imn", visp, visp)
        risp = lib.pack_tril(risp)
        mem_rho = visp.nbytes * 1e-6
        log.info("Memory usage for rho = % 6d MB" % mem_rho)
        cput1 = logger.timer(self, "interpolating density", *cput0)

        # Build the coulomb kernel
        coul = numpy.zeros((naux, nisp))
        blksize = int(self.max_memory * 1e6 * 0.9 / (8 * risp.size))
        blksize = min(naux, blksize)
        blksize = max(4, blksize)
        
        p0 = 0
        for cderi in with_df.loop(blksize=blksize):
            cput0 = (logger.process_clock(), logger.perf_counter())
            p1 = p0 + cderi.shape[0]
            coul[p0:p1] += (risp.dot(cderi.T)).T * 2.0

            logger.timer(self, "coulomb kernel [% 4d:% 4d]" % (p0, p1), *cput0)
            p0 += cderi.shape[0]

        cput1 = logger.timer(self, "coulomb kernel", *cput1)

        ww = scipy.linalg.solve_triangular(chol.T, coul.T, lower=True).T
        vv = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T  
        cput1 = logger.timer(self, "solving linear equations", *cput1)

        return vv, visp

    def kernel(self, *args, **kwargs):
        return super().kernel(*args, **kwargs)
    
ISDF = InterpolativeSeparableDensityFitting

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

    thc = ISDF(m)
    thc.verbose = 6
    thc.grids.level  = 1
    thc.grids.c_isdf = 100
    thc.with_df.auxbasis = "ccpvdz-jkfit"
    thc.max_memory = 100
    vv, visp = thc.build()

    from pyscf.lib import unpack_tril
    df_chol_ref = unpack_tril(thc.with_df._cderi)
    df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, visp, visp)
    
    err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
    err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol) / m.natm
    print("Method = %s, Error = % 6.4e % 6.4e" % ("cholesky", err1, err2))
