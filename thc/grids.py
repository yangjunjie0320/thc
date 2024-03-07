import pyscf, numpy, scipy
from scipy.sparse import dok_array

import pyscf
from pyscf.lib import logger
from pyscf.dft import numint

from pyscf.dft.gen_grid import Grids

def kmeans(coords, weighs, nip=10, ind0=None, max_cycle=100, tol=1e-4, verbose=0):
    """
    Perform K-Means clustering to find interpolation points.

    Args:
        coords (numpy.ndarray): real-space grid coordinates.
        weighs (numpy.ndarray): weights for each grid point.

        nip (int):   number of interpolation points.
        max_cycle (int): maximum iterations for K-Means algorithm.
        tol (float):   yhreshold for centroid convergence.

    Returns:
        numpy.ndarray: Interpolation indices after K-Means clustering.
    """
    log = logger.new_logger(None, verbose)
    ng, ndim = coords.shape
    nip = min(nip, ng)

    # Guess initial centroids by randomly selecting grid points
    if ind0 is None:
        ind0 = numpy.argsort(weighs)[::-1][:nip]
    
    centd_old = coords[ind0] # centroids

    icycle = 0
    is_converged = False
    is_max_cycle = False

    while (not is_converged) and (not is_max_cycle):
        # Classify grid points to the nearest centroid
        from scipy.cluster.vq import vq
        label, _ = vq(coords, centd_old)

        # Update centroids
        centd_new = [numpy.bincount(label, weights=coords[:, d] * weighs) for d in range(ndim)]
        centd_new = numpy.array(centd_new).T / numpy.abs(numpy.bincount(label, weights=weighs)[:, None] + 1e-6)

        # Check for convergence
        err = numpy.linalg.norm(centd_new - centd_old) / nip
        is_converged = (err < tol)
        is_max_cycle = (icycle >= max_cycle)

        log.debug("K-Means: cycle = % 3d, error = % 6.4e" % (icycle, err))

        icycle += 1
        centd_old = centd_new

    # Post-processing: map centroids to grid points
    return numpy.unique(vq(centd_old, coords)[0])

def dump_grids(mol, grid_coord, isdf_coord, outfile):
    from pyscf.lib import param

    s  = mol.tostring("xyz")
    s += "\n\n"

    s += "# The Parent Grid\n"
    for i in range(grid_coord.shape[0]):
        coord = grid_coord[i] * param.BOHR
        s += "% 12.8f % 12.8f % 12.8f\n" % tuple(coord)

    s += "\n# The Pruned Grid\n"
    for i in range(isdf_coord.shape[0]):
        coord = isdf_coord[i] * param.BOHR
        s += "% 12.8f % 12.8f % 12.8f\n" % tuple(coord)

    with open(outfile, "w") as f:
        f.write(s)

class InterpolatingPoints(Grids):
    tol = 1e-8
    c_isdf = 10
    alignment = 0
    
    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        '''
        Build ISDF grids.
        '''
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()

        log = logger.new_logger(self, self.verbose)
        log.info('Set up ISDF grids with QR decomposition.')

        atom_grids_tab = self.gen_atomic_grids(
            mol, self.atom_grid, self.radi_method, 
            self.level, self.prune, **kwargs
            )
        
        coords = []
        weighs = []

        tmp = zip(
            mol.aoslice_by_atom(), 
            *self.get_partition(
                mol, atom_grids_tab, self.radii_adjust, 
                self.atomic_radii,   self.becke_scheme,
                concat=False)
            )

        for ia, (s, c, w) in enumerate(tmp):
            phi  = numint.eval_ao(mol, c, deriv=0)
            phi *= (numpy.abs(w) ** 0.5)[:, None]
            mask = 

            nao = s[3] - s[2]
            ng  = phi.shape[0]

            # The cost of this method will not
            # grow with system size, but depends
            # on the number of AOs/grids per atom.
            import scipy.linalg
            from pyscf.lib import pack_tril
            rho = numpy.einsum('Im,In->Imn', phi, phi)
            rho = pack_tril(rho)

            q, r, perm = scipy.linalg.qr(rho.T, pivoting=True)
            diag  = numpy.abs(numpy.diag(r))
            d = diag / diag[0]

            nip = int(self.c_isdf) * nao if self.c_isdf else ng
            nip = min(nip, ng)
            mask = numpy.where(d > self.tol)[0]
            mask = mask if len(mask) < nip else mask[:nip]
            nip = len(mask)

            ind = perm[mask]
            coords.append(c[ind])
            weighs.append(w[ind])

            log.info("Atom %d %s: nao = % 4d, %6d -> %4d, err = % 6.4e" % (ia, mol.atom_symbol(ia), nao, ng, nip, d[mask[-1]]))

        self.coords  = numpy.vstack(coords)
        self.weights = numpy.hstack(weighs)

        if sort_grids:
            from pyscf.dft.gen_grid import arg_group_grids
            ind = arg_group_grids(mol, self.coords)
            self.coords = self.coords[ind]
            self.weights = self.weights[ind]

        if self.alignment > 1:
            raise KeyError("Alignment is not supported for ISDF grids.")

        if with_non0tab:
            self.non0tab = self.make_mask(mol, self.coords)
            self.screen_index = self.non0tab
        else:
            self.screen_index = self.non0tab = None

        return self
    
def build_rho(phi=None, tol=1e-8):
    ng, nao = phi.shape
    rho = dok_array((ng, nao * (nao + 1) // 2))

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
    rho  = dok_array((ngrid, nao * (nao + 1) // 2))
    mask = numpy.where(numpy.abs(phi) > numpy.sqrt(tol))

    for ig, mu in zip(*mask):
        rho_g_mu = phi[ig, mu] * phi[ig, :(mu+1)]
        rho_g_mu[-1] /= numpy.sqrt(2)

        munu = mu * (mu + 1) // 2 + numpy.arange(mu+1)
        ix = numpy.abs(rho_g_mu) > tol
        rho[ig, munu[ix]] = rho_g_mu[ix]

    ss  = rho.dot(rho.T)
    ss += ss.T
    if log is not None:
        log.info("nnz = % 6.4e / % 6.4e " % (ss.nnz, ss.shape[0] * ss.shape[1]))

    from scipy.linalg.lapack import dpstrf
    chol, perm, rank, info = dpstrf(ss.todense(), tol=tol) 

    nisp = rank
    if log is not None:
        log.info("Cholesky: rank = %d / %d" % (rank, ngrid))

    perm = (numpy.array(perm) - 1)[:nisp]

    tril = numpy.tril_indices(nisp, k=-1)
    chol = chol[:nisp, :nisp]
    chol[tril] *= 0.0
    visp = phi[perm]
    return chol, visp
    
if __name__ == "__main__":
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
    
    from pyscf.dft import Grids
    from pyscf.dft.numint import NumInt
    ni   = NumInt()

    # Are "kmeans" and "kmeans-density" methods really useful?
    for t in [1e-1]: #, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        print("\nTolerance = % 6.4e" % t)

        grid = InterpolatingPoints(m)
        grid.level   = 1
        grid.verbose = 10
        grid.c_isdf  = None
        grid.tol     = t
        grid.build()

        from pyscf.lib.logger import perf_counter, process_clock
        log = pyscf.lib.logger.Logger(verbose=5)
        
        phi  = ni.eval_ao(m, grid.coords)
        phi *= (numpy.abs(grid.weights) ** 0.5)[:, None]
        chol, visp = cholesky(phi, tol=1e-12, log=log)
        nisp, nao = visp.shape
        rho = build_rho(visp, tol=1e-12)
        
        df = pyscf.df.DF(m)
        df.max_memory = 400
        df.auxbasis = "weigend"
        df.build()
        naux = df.get_naoaux()

        coul = numpy.zeros((naux, nisp))

        p1 = 0
        blksize = 10
        for istep, chol_l in enumerate(df.loop(blksize=blksize)):
            p0, p1 = p1, p1 + chol_l.shape[0]
            coul[p0:p1] = rho.dot(chol_l.T).T * 2.0

        ww = scipy.linalg.solve_triangular(chol.T, coul.T, lower=True).T
        vv = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T  

        from pyscf.lib import unpack_tril
        df_chol_ref = unpack_tril(df._cderi)
        df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, visp, visp)
        
        err1 = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
        err2 = numpy.linalg.norm(df_chol_ref - df_chol_sol) / m.natm
        print("Method = %s, Error = % 6.4e % 6.4e" % ("cholesky", err1, err2))
