import ctypes, numpy, scipy
import scipy.linalg

import pyscf
from pyscf.lib import logger
from pyscf.dft import numint

from pyscf import dft, lib
import pyscf.dft.gen_grid

from pyscf.dft.gen_grid import (nwchem_prune, gen_atomic_grids,
                                BLKSIZE, NBINS, CUTOFF)
from pyscf.pbc.gto import eval_gto as pbc_eval_gto

libpbc = lib.load_library('libpbc')

# modified from pyscf.dft.gen_grid.gen_partition and pyscf.pbc.dft.gen_grid.get_becke_grids
def gen_partition(cell, atom_grid={}, radi_method=dft.radi.gauss_chebyshev,
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
        
        # print(c_ia, w_ia)
        c_ia = numpy.vstack(c_ia)
        w_ia = numpy.hstack(w_ia)

        coord_grid.append(c_ia)
        weigh_grid.append(w_ia)

    coord_atm_img = numpy.vstack(coord_atm_img)
    p_radii_table = lib.c_null_ptr()

    def gen_grid_partition(c):
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

        return p

    coord_all = []
    weigh_all = []

    for ia in range(cell.natm):
        c_ia = coord_grid[ia]
        w_ia = weigh_grid[ia]

        p_ia = gen_grid_partition(c_ia)
        w_ia /= p_ia.sum(axis=0)

        coord_all.append(c_ia)
        weigh_all.append(w_ia)

    if concat:
        coord_all = numpy.vstack(coord_all)
        weigh_all = numpy.hstack(weigh_all)
    
    print(coord_all, weigh_all)



    # offs = numpy.append(0, numpy.cumsum([w.size for w in weights_all]))

    # coords_all = numpy.vstack(coords_all)
    # weighs_all = numpy.hstack(weighs_all)

    # atm_coords = numpy.asarray(atm_coords.reshape(-1,3)[supatm_idx], order='C')
    # sup_natm = len(atm_coords)
    # p_radii_table = lib.c_null_ptr()
    # fn = dft.gen_grid.libdft.VXCgen_grid
    # ngrids = weights_all.size

    # for iL, L in enumerate(Ls):
    #     for ia in range(cell.natm):
    #         coords, vol = atom_grids_tab[cell.atom_symbol(ia)]
    #         coords = coords + atm_coords[iL, ia]

    # # max_memory = cell.max_memory - lib.current_memory()[0]
    # # blocksize = min(ngrids, max(2000, int(max_memory*1e6/8 / sup_natm)))
    # # displs = lib.misc._blocksize_partition(offs, blocksize)

    # # for n0, n1 in zip(displs[:-1], displs[1:]):
    # #     p0, p1 = offs[n0], offs[n1]
    # #     pbecke = numpy.empty((sup_natm,p1-p0))
    # #     coords = numpy.asarray(coords_all[p0:p1], order='F')
    # #     fn(pbecke.ctypes.data_as(ctypes.c_void_p),
    # #        coords.ctypes.data_as(ctypes.c_void_p),
    # #        atm_coords.ctypes.data_as(ctypes.c_void_p),
    # #        p_radii_table, ctypes.c_int(sup_natm), ctypes.c_int(p1-p0))

    # #     weights_all[p0:p1] /= pbecke.sum(axis=0)
    # #     for ia in range(n0, n1):
    # #         i0, i1 = offs[ia], offs[ia+1]
    # #         weights_all[i0:i1] *= pbecke[ia,i0-p0:i1-p0]

    # # return coords_all, weights_all

# class InterpolatingPoints(pyscf.dft.gen_grid.Grids):
#     tol = 1e-8
#     c_isdf = 50
#     alignment = 0
    
#     def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
#         '''
#         Build ISDF grids.
#         '''
#         if mol is None: mol = self.mol
#         log = logger.new_logger(self, self.verbose)
#         log.info('\nSet up ISDF grids with QR decomposition.')
#         if self.c_isdf is not None:
#             log.info('c_isdf = %d', self.c_isdf)

#         atom_grids_tab = self.gen_atomic_grids(
#             mol, self.atom_grid, self.radi_method,
#             self.level, self.prune, **kwargs
#         )

#         grid = zip(
#             mol.aoslice_by_atom(),
#             *self.get_partition(
#                 mol, atom_grids_tab, self.radii_adjust,
#                 self.atomic_radii, self.becke_scheme,
#                 concat=False
#             )
#         )

#         cput0 = (logger.process_clock(), logger.perf_counter())

#         coords = []
#         weights = []

#         for ia, (s, c, w) in enumerate(grid):
#             sym = mol.atom_symbol(ia)
#             nao = s[3] - s[2]

#             phi  = numint.eval_ao(mol, c, deriv=0, shls_slice=None)
#             phi *= (numpy.abs(w) ** 0.5)[:, None]
#             phi  = phi[numpy.linalg.norm(phi, axis=1) > self.tol]
#             ng   = phi.shape[0]
            
#             phi_pair  = numpy.einsum("xm,xn->xmn", phi, phi[:, s[2]:s[3]])
#             phi_pair  = phi_pair.reshape(ng, -1)

#             q, r, perm = scipy.linalg.qr(phi_pair.T, pivoting=True)
#             diag = (lambda d: d / d[0])(numpy.abs(numpy.diag(r)))

#             nip = int(self.c_isdf) * nao if self.c_isdf else ng
#             nip = min(nip, ng)

#             mask = numpy.where(diag > self.tol)[0]
#             mask = mask if len(mask) < nip else mask[:nip]
#             nip = len(mask)

#             ind = perm[mask]
#             coords.append(c[ind])
#             weights.append(w[ind])

#             log.info(
#                 "Atom %d %s: nao = % 4d, %6d -> %4d, err = % 6.4e" % (
#                     ia, sym, nao, w.size, nip, diag[mask[-1]]
#                 )
#             )

#         log.timer("Building Interpolating Points", *cput0)

#         self.coords  = numpy.vstack(coords)
#         self.weights = numpy.hstack(weights)

#         if sort_grids:
#             from pyscf.dft.gen_grid import arg_group_grids
#             ind = arg_group_grids(mol, self.coords)
#             self.coords = self.coords[ind]
#             self.weights = self.weights[ind]

#         if self.alignment > 1:
#             raise KeyError("Alignment is not supported for ISDF grids.")

#         if with_non0tab:
#             self.non0tab = self.make_mask(mol, self.coords)
#             self.screen_index = self.non0tab
#         else:
#             self.screen_index = self.non0tab = None

#         log.info('Total number of interpolating points = %d', len(self.weights))
#         return self

# Grids = InterpolatingPoints

if __name__ == "__main__":
    n = 7
    cell = pyscf.pbc.gto.Cell()
    cell.a = '''
    4.000 0.0000 0.0000
    0.000 4.0000 0.0000
    0.000 0.0000 4.0000
    '''
    cell.mesh = [n, n, n]

    cell.atom = '''He 0.0000 0.0000 1.0000
                   He 1.0000 0.0000 1.0000'''
    cell.basis = "sto3g"
    cell.build()

    gen_partition(
        cell, concat=False
    )


    # grid = InterpolatingPoints(m)
    # grid.level = 0
    # grid.verbose = 6
    # grid.c_isdf  = 20
    # grid.tol     = 1e-8
    # grid.kernel()
