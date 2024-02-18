import numpy, scipy

import pyscf
from pyscf.lib import logger
from pyscf.dft import numint

from pyscf.dft import gen_grid
from pyscf.dft.gen_grid import Grids

def kmeans(coords, weighs, nip=10, ind0=None, max_cycle=100, tol=1e-4):
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
    ng, ndim = coords.shape

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

        print("K-Means: cycle = % 3d, error = % 6.4e" % (icycle, err))

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
    alignment = 0
    c_isdf = 10
    method = "qr"

    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        '''
        Build ISDF grids.
        Currently, only the K-means method is applied to construct the grid.
        The weighting function is defined as the atomic density multiplied by
        the square root of the DFT quadrature weight, which works best for THCDF.

        Returns:
            grids : :class:`dft.gen_grid.Grids`
                ISDF grid
        '''
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()

        atom_grids_tab = self.gen_atomic_grids(
            mol, self.atom_grid, self.radi_method, 
            self.level, self.prune, **kwargs
            )
        
        self.coords, self.weights = self.get_partition(
            mol, atom_grids_tab, self.radii_adjust, 
            self.atomic_radii, self.becke_scheme,
            concat=False
            )

        coords = []
        weighs = []

        aoslice = mol.aoslice_by_atom()
        for ia, (c, w) in enumerate(zip(self.coords, self.weights)):
            nao = (lambda s: s[3] - s[2])(aoslice[ia])
            nip = int(self.c_isdf) * nao

            phi = numint.eval_ao(mol, c, deriv=0)
            ng, nao = phi.shape

            if self.method == "kmeans":
                ind = kmeans(
                    c, w, nip=nip*2, 
                    ind0=None,
                    max_cycle=20, tol=1e-4,
                    )
                ind = ind[:nip]
                
            elif self.method == "qr":
                import scipy.linalg
                from pyscf.lib import pack_tril

                rho = numpy.einsum('gm,gn->gmn', phi, phi)
                rho = pack_tril(rho)

                q, r, perm = scipy.linalg.qr(rho.T, pivoting=True)
                ind = perm[:nip]

            else:
                raise KeyError("Unknown method: %s" % self.method)

            coords.append(c[ind])
            weighs.append(w[ind])

        if self.verbose >= logger.DEBUG:
            coord1 = numpy.vstack(self.coords)
            coord2 = numpy.vstack(coords)
            dump_grids(mol, coord1, coord2, "/Users/yangjunjie/Downloads/h2o-%s.log" % self.method)

        self.coords = numpy.vstack(coords)
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

        # logger.info(self, 'tot grids = %d', len(self.weights))
        return self
    
if __name__ == "__main__":
    m = pyscf.gto.M(
        atom="""
        O   -6.0082242   -6.2662586   -4.5802338
        H   -6.5424905   -5.6103439   -4.0656431
        H   -5.6336920   -6.8738199   -3.8938766
        O   -5.0373186   -5.2694388   -1.2581917
        H   -4.2054186   -5.5931034   -0.8297803
        H   -4.7347234   -4.8522045   -2.1034720
        """, basis="sto3g", verbose=0
        )

    grids = InterpolatingPoints(m)
    grids.level = 0
    grids.c_isdf = 5
    grids.method = "qr"
    grids.build()

    grids = InterpolatingPoints(m)
    grids.level = 0
    grids.c_isdf = 5
    grids.method = "kmeans"
    grids.build()
