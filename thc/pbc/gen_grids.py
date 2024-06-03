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

from thc.mol.gen_grids import InterpolatingPointsMixin

# class InterpolatingPoints(thc.mol.gen_grids.InterpolatingPoints):
#     def __init__(self, cell):
#         self.cell = cell
#         thc.mol.gen_grids.InterpolatingPoints.__init__(self, cell)

#     gen_partition = gen_partition

#     def make_mask(self, cell, coords, relativity=0, shls_slice=None, verbose=None):
#         if cell is None: cell = self.cell
#         if coords is None: coords = self.coords
#         from pyscf.pbc.dft.gen_grid import make_mask
#         return make_mask(cell, coords, relativity, shls_slice, verbose)

#     def build(self, cell=None, with_non0tab=False):
#         '''
#         Build ISDF grids.
#         '''
#         if cell is None: cell = self.cell
#         log = logger.new_logger(self, self.verbose)
#         log.info('\nSet up interpolating points with Pivoted Cholesky decomposition.')
#         if self.c_isdf is not None:
#             log.info('c_isdf = %d', self.c_isdf)

#         grid = zip(
#             cell.aoslice_by_atom(),
#             *self.gen_partition(
#                 cell, self.atom_grid,
#                 radi_method=self.radi_method,
#                 level=self.level,
#                 prune=self.prune,
#                 concat=False
#             )
#         )

#         cput0 = (logger.process_clock(), logger.perf_counter())

#         coords = []
#         weights = []

#         for ia, (s, c, w) in enumerate(grid):
#             sym = cell.atom_symbol(ia)
#             nao = s[3] - s[2]

#             phi  = numint.eval_ao(cell, c, deriv=0, shls_slice=None)
#             phi *= (numpy.abs(w) ** 0.5)[:, None]
#             ng   = phi.shape[0]

#             nip = int(self.c_isdf) * nao if self.c_isdf else ng
#             nip = min(nip, ng)
            
#             from pyscf.lib import pivoted_cholesky
#             phi4 = pyscf.lib.dot(phi, phi.T) ** 2
#             chol, perm, rank = pivoted_cholesky(phi4, tol=self.tol, lower=False)
#             err = chol[nip-1, nip-1] / chol[0, 0]

#             mask = perm[:nip]
#             coords.append(c[mask])
#             weights.append(w[mask])

#             log.info(
#                 "Atom %4d %3s: nao = % 4d, %6d -> %4d, err = % 6.4e" % (
#                     ia, sym, nao, w.size, nip, err
#                 )
#             )

#         log.timer("Building Interpolating Points", *cput0)

#         self.coords  = numpy.vstack(coords)
#         self.weights = numpy.hstack(weights)

#         if with_non0tab:
#             self.non0tab = self.make_mask(cell, self.coords)
#             self.screen_index = self.non0tab
#         else:
#             self.screen_index = self.non0tab = None

#         log.info('Total number of interpolating points = %d', len(self.weights))
#         return self

# Grids = InterpolatingPoints

class UniformGridsForSolid(InterpolatingPointsMixin, pyscf.pbc.dft.gen_grid.UniformGrids):
    def __init__(self, cell):
        self.mol = self.cell = cell
        pyscf.pbc.dft.gen_grid.UniformGrids.__init__(self, cell)

    def _eval_gto(self, coord, weigh):
        from pyscf.pbc.dft import numint
        phi = numint.eval_ao(self.cell, coord, deriv=0, shls_slice=None)
        phi *= (numpy.abs(weigh) ** 0.5)[:, None]
        return phi
    
class BeckeGridsForSolid(InterpolatingPointsMixin, pyscf.pbc.dft.gen_grid.UniformGrids):
    pass
    # def __init__(self, cell):
    #     self.mol = self.cell = cell
    #     pyscf.pbc.dft.gen_grid.UniformGrids.__init__(self, cell)

    # def _eval_gto(self, coord, weigh):
    #     from pyscf.pbc.dft import numint
    #     phi = numint.eval_ao(self.cell, coord, deriv=0, shls_slice=None)
    #     phi *= (numpy.abs(weigh) ** 0.5)[:, None]
    #     return phi

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

    grid = BeckeGridsForSolid(c)
    grid.mesh = [15, 15, 15]
    grid.level = 0
    grid.verbose = 6
    grid.c_isdf  = 40
    grid.kernel()
