import ctypes, numpy, scipy
import scipy.linalg

import pyscf
from pyscf.lib import logger
from pyscf.pbc.dft import numint

from pyscf import dft, lib
import pyscf.dft.gen_grid

from pyscf.dft.gen_grid import nwchem_prune, gen_atomic_grids
from pyscf.pbc.gto import eval_gto as pbc_eval_gto
libpbc = lib.load_library('libpbc')

import thc
import thc.mol.gen_grids

# modified from pyscf.dft.gen_grid.gen_partition and pyscf.pbc.dft.gen_grid.get_becke_grids
def gen_partition(grid, cell, atom_grid={}, radi_method=dft.radi.gauss_chebyshev,
                  level=3, concat=True, prune=nwchem_prune):
    '''real-space grids using Becke scheme

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (N, 3) ndarray
            The real-space grid point coordinates.
        weights : (N) ndarray
    '''

    # When low_dim_ft_type is set, pbc_eval_gto treats the 2D system as a 3D system.
    # To get the correct particle number in numint module, the atomic grids needs to
    # be consistent with the treatment in pbc_eval_gto (see issue 164).
    if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
        dimension = cell.dimension
    else:
        dimension = 3

    rcut = pbc_eval_gto._estimate_rcut(cell)
    Ls = pbc_eval_gto.get_lattice_Ls(cell, rcut=rcut.max())
    b = cell.reciprocal_vectors(norm_to=1)

    coord_atom = Ls.reshape(-1,1,3) + cell.atom_coords()
    atom_grids_tab = gen_atomic_grids(cell, atom_grid, radi_method, level, prune)

    tol = 1e-15
    coord_grid = []
    weigh_grid = []
    coord_atm_img = []

    for ia in range(cell.natm):
        c_ia = []
        w_ia = []

        for iL, L in enumerate(Ls):
            c, w = atom_grids_tab[cell.atom_symbol(ia)]
            c = c + coord_atom[iL, ia]

            # search for grids in unit cell
            b_dot_c = b.dot(c.T)
            mask = numpy.ones(b_dot_c.shape[1], dtype=bool)
            if dimension >= 1:
                mask &= (b_dot_c[0] > -0.5 - tol) & (b_dot_c[0] < 0.5 + tol)
            if dimension >= 2:
                mask &= (b_dot_c[1] > -0.5 - tol) & (b_dot_c[1] < 0.5 + tol)
            if dimension == 3:
                mask &= (b_dot_c[2] > -0.5 - tol) & (b_dot_c[2] < 0.5 + tol)
            
            c = c[mask]
            w = w[mask]

            if w.size > 8:
                b_dot_c = b_dot_c[:, mask]
                if dimension >= 1:
                    w[abs(b_dot_c[0] + 0.5) < tol] *= .5
                    w[abs(b_dot_c[0] - 0.5) < tol] *= .5

                if dimension >= 2:
                    w[abs(b_dot_c[1] + 0.5) < tol] *= .5
                    w[abs(b_dot_c[1] - 0.5) < tol] *= .5

                if dimension == 3:
                    w[abs(b_dot_c[2] + 0.5) < tol] *= .5
                    w[abs(b_dot_c[2] - 0.5) < tol] *= .5

                c_ia.append(c)
                w_ia.append(w)
                coord_atm_img.append(coord_atom[iL, ia])

        c_ia = numpy.vstack(c_ia)
        w_ia = numpy.hstack(w_ia)

        coord_grid.append(c_ia)
        weigh_grid.append(w_ia)

    coord_atm_img = numpy.vstack(coord_atm_img)
    p_radii_table = lib.c_null_ptr()

    coord_all = []
    weigh_all = []

    for ia, (c, w) in enumerate(zip(coord_grid, weigh_grid)):
        c  = numpy.asarray(c, order='C')
        ng = c.shape[0]
        na = coord_atm_img.shape[0]
        p  = numpy.empty((na, ng))

        dft.gen_grid.libdft.VXCgen_grid(
            p.ctypes.data_as(ctypes.c_void_p),
            c.ctypes.data_as(ctypes.c_void_p),
            coord_atm_img.ctypes.data_as(ctypes.c_void_p),
            p_radii_table, ctypes.c_int(na), ctypes.c_int(ng)
        )
        w /= p.sum(axis=0)

        coord_all.append(c)
        weigh_all.append(w)

    if concat:
        coord_all = numpy.vstack(coord_all)
        weigh_all = numpy.hstack(weigh_all)

    logger.info(cell, "Total number of grids %d", len(numpy.hstack(weigh_all)))
    return coord_all, weigh_all

class InterpolatingPoints(thc.mol.gen_grids.InterpolatingPoints):
    def __init__(self, cell):
        self.cell = cell
        thc.mol.gen_grids.InterpolatingPoints.__init__(self, cell)

    gen_partition = gen_partition

    def make_mask(self, cell, coords, relativity=0, shls_slice=None, verbose=None):
        if cell is None: cell = self.cell
        if coords is None: coords = self.coords
        from pyscf.pbc.dft.gen_grid import make_mask
        return make_mask(cell, coords, relativity, shls_slice, verbose)

    def build(self, cell=None, with_non0tab=False):
        '''
        Build ISDF grids.
        '''
        if cell is None: cell = self.cell
        log = logger.new_logger(self, self.verbose)
        log.info('\nSet up interpolating points with Pivoted Cholesky decomposition.')
        if self.c_isdf is not None:
            log.info('c_isdf = %d', self.c_isdf)

        grid = zip(
            cell.aoslice_by_atom(),
            *self.gen_partition(
                cell, self.atom_grid,
                radi_method=self.radi_method,
                level=self.level,
                prune=self.prune,
                concat=False
            )
        )

        cput0 = (logger.process_clock(), logger.perf_counter())

        coords = []
        weights = []

        for ia, (s, c, w) in enumerate(grid):
            sym = cell.atom_symbol(ia)
            nao = s[3] - s[2]

            phi  = numint.eval_ao(cell, c, deriv=0, shls_slice=None)
            phi *= (numpy.abs(w) ** 0.5)[:, None]
            ng   = phi.shape[0]

            nip = int(self.c_isdf) * nao if self.c_isdf else ng
            nip = min(nip, ng)
            
            from pyscf.lib import pivoted_cholesky
            phi4 = pyscf.lib.dot(phi, phi.T) ** 2
            chol, perm, rank = pivoted_cholesky(phi4, tol=self.tol, lower=False)
            mask = perm[:nip]

            coords.append(c[mask])
            weights.append(w[mask])

            log.info(
                "Atom %d %s: nao = % 4d, %6d -> %4d" % (
                    ia, sym, nao, w.size, nip
                )
            )

        log.timer("Building Interpolating Points", *cput0)

        self.coords  = numpy.vstack(coords)
        self.weights = numpy.hstack(weights)

        if with_non0tab:
            self.non0tab = self.make_mask(cell, self.coords)
            self.screen_index = self.non0tab
        else:
            self.screen_index = self.non0tab = None

        log.info('Total number of interpolating points = %d', len(self.weights))
        return self

Grids = InterpolatingPoints

if __name__ == "__main__":
    c   = pyscf.pbc.gto.Cell()
    c.a = numpy.eye(3) * 3.5668
    c.atom = '''C     0.0000  0.0000  0.0000
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.0000
                C     2.6751  2.6751  0.8917
                C     1.7834  0.0000  1.7834
                C     2.6751  0.8917  2.6751
                C     0.0000  1.7834  1.7834
                C     0.8917  2.6751  2.6751'''
    c.basis  = 'gth-szv'
    c.pseudo = 'gth-pade'
    c.verbose = 4
    c.build()

    grid = InterpolatingPoints(c)
    grid.level = 0
    grid.verbose = 6
    grid.c_isdf  = 20
    grid.tol     = 1e-8
    grid.kernel()
