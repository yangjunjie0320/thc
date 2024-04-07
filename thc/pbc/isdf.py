import numpy, scipy
import scipy.linalg

import pyscf
from pyscf.pbc import df
from pyscf import lib
from pyscf.dft import numint
from pyscf.lib import logger

import thc
from thc.mol.least_square import cholesky
from thc.pbc.gen_grids import InterpolatingPoints

from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band
from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

def _get_rhoR(mydf, dm_kpts, hermi=1):
    '''Electron density in real space grids.
    '''

    kpts      = numpy.zeros((1,3))
    kpts_band = None

    ### step 1 , evaluate ao_values on the grid
    cell = mydf.cell
    grids = mydf.grids
    coords = numpy.asarray(grids.coords).reshape(-1,3)
    mesh = grids.mesh
    ngrids = numpy.prod(mesh)
    assert ngrids == coords.shape[0]

    ### step 2, evaluate the density on the grid as the weight for kmean
    ### TODO: make it linear scaling

    dm_kpts = numpy.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, _ = dms.shape[:3]
    assert nset == 1
    assert nkpts == 1  # only gamma point for now

    kpts_band = _format_kpts_band(kpts_band, kpts)

    # density in grid space   $\rho(G)=\int_\Omega \rho(R) e^{-iGr} dr$
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    # the above comment is from pyscf/pbc/dft/multigrid_pair.py
    # $\rho(R) = 1/\Omega \int_BZ \rho(G) e^{iGr} dG$ ???
    rhoR = pyscf.tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.flatten()
    assert rhoR.size == ngrids

    return rhoR

def isdf(mydf, dm_kpts, hermi=1, naux=None, c=5, max_iter=100, kpts=numpy.zeros((1,3)), kpts_band=None, verbose=None):
    '''
    Args:
        mydf                : the DF object
        dm_kpts (numpy.ndarray): (nset, nkpts, nao, nao) density matrix in k-space
        hermi (int)         : int, optional
                              If :math:`hermi=1`, the task list is built only for
                              the upper triangle of the matrix. Default is 0.
        naux (int)          : number of auxiliary basis functions
        c (int)             : the ratio between the number of auxiliary basis functions and 
                              the number of atomic basis functions
                              if naux is none, then naux is set to c * cell.nao
        max_iter (int)      : max number of iterations for kmean
        verbose (int)       : verbosity level
        kpts (numpy.ndarray)   : 

    Returns:
        W (numpy.ndarray)      : (naux,naux) matrix of the ISDF potential
        aoRg (numpy.ndarray)   : (naux,ngrids) matrix of the auxiliary basis
        aoR (numpy.ndarray)    : (nao, ngrids) matrix of the (scaled) atomic basis in real space
        V_R (numpy.ndarray)    : (naux,ngrids) matrix of the ISDF potential in real space
        idx (numpy.ndarray)    : (naux,) index of the auxiliary basis in the real space grid

    Ref: 
    (1) Lu2015 
    (2) Hu2023      10.1021/acs.jctc.2c00927
    (3) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720

    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    ### step 1 , evaluate ao_values on the grid

    cell   = mydf.cell
    grids  = mydf.grids
    coords = numpy.asarray(grids.coords).reshape(-1,3)
    mesh   = grids.mesh
    ngrids = numpy.prod(mesh)
    assert ngrids == coords.shape[0]

    log   = logger.new_logger(mydf, verbose=verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    aoR   = mydf._numint.eval_ao(cell, coords)[0]

    aoR  *= numpy.sqrt(cell.vol / ngrids)   ## NOTE: scaled !

    print("aoR.shape = ", aoR.shape)

    cput1 = log.timer('eval_ao', *cput0)
    if naux is None:
        naux = cell.nao * c  # number of auxiliary basis functions

    ### step 2, evaluate the density on the grid as the weight for kmean
    ### TODO: make it linear scaling

    dm_kpts = numpy.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    assert nset == 1
    assert nkpts == 1  # only gamma point for now
    # kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    kpts_band = _format_kpts_band(kpts_band, kpts)

    # density in grid space   $\rho(G)=\int_\Omega \rho(R) e^{-iGr} dr$
    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    # the above comment is from pyscf/pbc/dft/multigrid_pair.py
    # $\rho(R) = 1/\Omega \int_BZ \rho(G) e^{iGr} dG$ ???
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.flatten()
    assert rhoR.size == ngrids

    ### step 3, kmean clustering get the IP
    ### TODO: implement QRCP as an option

    cput1 = log.timer('eval_rhoR', *cput1)
    # from cuml.cluster import KMeans
    # from scikit-learn.cluster import KMeans
    from sklearn.cluster import KMeans
    kmeans_float = KMeans(n_clusters=naux,
                          max_iter=max_iter,
                          # max_samples_per_batch=32768*8//naux,
                          # output_type='numpy'
                          )
    kmeans_float.fit(coords, sample_weight=rhoR)
    centers = kmeans_float.cluster_centers_

    cput1 = log.timer('kmeans', *cput1)

    t2 = (logger.process_clock(), logger.perf_counter())
    # _benchmark_time(t1, t2, "kmeans")
    t1 = t2

    ### step 4, get the auxiliary basis

    a = cell.lattice_vectors()
    scaled_centers = numpy.dot(centers, numpy.linalg.inv(a))

    idx = (numpy.rint(scaled_centers*mesh[None,:]) + mesh[None,:]) % (mesh[None,:])
    idx = idx[:,2] + idx[:,1]*mesh[2] + idx[:,0]*(mesh[1]*mesh[2])
    idx = idx.astype(int)
    idx = list(set(idx))
    idx.sort()
    idx = numpy.asarray(idx)
    print("idx = ", idx)

    cput1 = log.timer('get idx', *cput1)

    aoRg = aoR[idx]  # (nIP, nao), nIP = naux
    # A = numpy.dot(aoRg, aoRg.T) ** 2  # (Naux, Naux)
    A = numpy.asarray(lib.dot(aoRg, aoRg.T), order='C')
    A = A ** 2
    cput1 = log.timer('get A', *cput1)

    X = numpy.empty((naux,ngrids))
    blksize = int(10*1e9/8/naux)
    for p0, p1 in lib.prange(0, ngrids, blksize):
        # B = numpy.dot(aoRg, aoR[p0:p1].T) ** 2
        B = numpy.asarray(lib.dot(aoRg, aoR[p0:p1].T), order='C')
        B = B ** 2
        X[:,p0:p1] = scipy.linalg.lstsq(A, B)[0]
        B = None
    A = None

    cput1 = log.timer('least squre fit', *cput1)

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "Construct Xg")
    t1 = t2

    ### step 5, get the ISDF potential, V(R_g, R')

    V_R   = numpy.empty((naux,ngrids))
    coulG = tools.get_coulG(cell, mesh=mesh)

    blksize1 = int(5*1e9/8/ngrids)
    for p0, p1 in lib.prange(0, naux, blksize1):
        X_freq     = numpy.fft.fftn(X[p0:p1].reshape(-1,*mesh), axes=(1,2,3)).reshape(-1,ngrids)
        V_G        = X_freq * coulG[None,:]
        X_freq     = None
        V_R[p0:p1] = numpy.fft.ifftn(V_G.reshape(-1,*mesh), axes=(1,2,3)).real.reshape(-1,ngrids)
        V_G        = None
    coulG = None
    # V_R  *= 2 * numpy.pi

    cput1 = log.timer('fft', *cput1)

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "Construct VR")
    t1 = t2

    W = numpy.zeros((naux,naux))
    for p0, p1 in lib.prange(0, ngrids, blksize):
        W += numpy.dot(X[:,p0:p1], V_R[:,p0:p1].T)

    # for i in range(naux):
    #     for j in range(i):
    #         print("W[%5d, %5d] = %15.8e" % (i, j, W[i,j]))

    cput1 = log.timer('get W', *cput1)

    t2 = (logger.process_clock(), logger.perf_counter())
    _benchmark_time(t1, t2, "Construct WR")

    return W, aoRg.T, aoR.T, V_R, idx, X

if __name__ == "__main__":
    from pyscf.pbc import gto as pbcgto
    cell   = pbcgto.Cell()
    cell.a = numpy.array([[3.5668, 0.0, 0.0], [0.0, 3.5668, 0.0],[0.0, 0.0, 3.5668]])

    cell.atom = '''
                   C     0.8917  0.8917  0.8917
                   C     2.6751  2.6751  0.8917
                   C     2.6751  0.8917  2.6751
                   C     0.8917  2.6751  2.6751
                '''

    cell.basis   = 'gth-szv'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4

    # cell.ke_cutoff  = 100   # kinetic energy cutoff in a.u.
    cell.ke_cutoff  = 128
    cell.max_memory = 800  # 800 Mb
    cell.precision  = 1e-8  # integral precision
    cell.use_particle_mesh_ewald = True
    cell.build()

    print("Number of electrons: ", cell.nelectron)
    print("Number of atoms    : ", cell.natm)
    print("Number of basis    : ", cell.nao)
    print("Number of images   : ", cell.nimgs)

    # make a super cell

    from pyscf.pbc import tools
    cell = tools.super_cell(cell, [1, 1, 1])

    print("Number of electrons: ", cell.nelectron)
    print("Number of atoms    : ", cell.natm)
    print("Number of basis    : ", cell.nao)
    print("Number of images   : ", cell.nimgs)

    # # construct DF object

    # mf            = pbcdft.RKS(cell)
    # mf.xc         = "PBE,PBE"
    # mf.init_guess = 'atom'  # atom guess is fast
    # mf.with_df    = multigrid.MultiGridFFTDF2(cell)

    # # mf.with_df.ngrids = 4  # number of sets of grid points ? ? ?
    # # mf.kernel()

    # dm1 = mf.get_init_guess(cell, 'atom')
    # mydf = MultiGridFFTDF2(cell)

    # s1e = mf.get_ovlp(cell)

    # print(s1e.shape)
    # print(dm1.shape)
    # print(mydf.grids.mesh)
    # print(mydf.grids.coords.shape)

    # # perform ISDF

    # rhoR = _get_rhoR(mydf, dm1)
    # print("rhoR.shape = ", rhoR.shape)
    # print("nelec from rhoR is ", numpy.sum(rhoR) * cell.vol / numpy.prod(cell.mesh))

    # W, aoRg, aoR, V_R, idx, _ = isdf(mydf, dm1, naux=cell.nao*10, max_iter=100, verbose=4)
