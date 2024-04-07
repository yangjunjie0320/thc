import pyscf, numpy, scipy
import scipy.linalg

import pyscf
from pyscf.lib import logger
from pyscf.dft import numint
import pyscf.dft.gen_grid

from pyscf.dft import gen_grid
class InterpolatingPoints(pyscf.dft.gen_grid.Grids):
    tol = 1e-8
    c_isdf = 50
    alignment = 0

    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        '''
        Build ISDF grids.
        '''
        if mol is None: mol = self.mol
        log = logger.new_logger(self, self.verbose)
        log.info('\nSet up interpolating points with Pivoted Cholesky decomposition.')
        if self.c_isdf is not None:
            log.info('c_isdf = %d', self.c_isdf)

        atom_grids_tab = self.gen_atomic_grids(
            mol, self.atom_grid, self.radi_method,
            self.level, self.prune, **kwargs
        )

        grid = zip(
            mol.aoslice_by_atom(),
            *self.get_partition(
                mol, atom_grids_tab, self.radii_adjust,
                self.atomic_radii, self.becke_scheme,
                concat=False
            )
        )

        cput0 = (logger.process_clock(), logger.perf_counter())

        coords = []
        weights = []

        for ia, (s, c, w) in enumerate(grid):
            sym = mol.atom_symbol(ia)
            nao = s[3] - s[2]

            phi  = numint.eval_ao(mol, c, deriv=0, shls_slice=None)
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

        log.info('Total number of interpolating points = %d', len(self.weights))
        return self

Grids = InterpolatingPoints

if __name__ == "__main__":
    m = pyscf.gto.M(
        atom="""
        O   -6.0082242   -6.2662586   -4.5802338
        H   -6.5424905   -5.6103439   -4.0656431
        H   -5.6336920   -6.8738199   -3.8938766
        O   -5.0373186   -5.2694388   -1.2581917
        H   -4.2054186   -5.5931034   -0.8297803
        H   -4.7347234   -4.8522045   -2.1034720
        """, basis="ccpvqz", verbose=0
    )

    grid = InterpolatingPoints(m)
    grid.level = 0
    grid.verbose = 6
    grid.c_isdf  = 10
    grid.tol     = 1e-8
    grid.kernel()
