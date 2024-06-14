import pyscf, numpy, scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint
import pyscf.dft.gen_grid
    

class InterpolatingPointsMixin(lib.StreamObject):
    c_isdf = 10
    tol = 1e-20

    def __init__(self):
        raise NotImplementedError

    def _eval_gto(self, coord, weigh):
        raise NotImplementedError
    
    def _divide(self, coord):
        xa = self.mol.atom_coords()
        xg = coord

        na = xa.shape[0]
        ng = xg.shape[0]

        assert xa.shape == (na, 3)
        assert xg.shape == (ng, 3)
        assert na < ng

        d = numpy.linalg.norm(xa[:, None, :] - xg[None, :, :], axis=2)
        assert d.shape == (na, ng)

        ind = numpy.argmin(d, axis=0)
        return [numpy.where(ind == ia)[0] for ia in range(na)]

    def _select(self, ia, coord=None, weigh=None, c_isdf=None, tol=None):
        mol = self.mol
        assert not (c_isdf is None and tol is None)

        sym = mol.atom_symbol(ia)
        nao = (lambda s: s[3] - s[2])(mol.aoslice_by_atom()[ia])
        
        ng = coord.shape[0]
        assert coord.shape == (ng, 3)
        assert weigh.shape == (ng,)

        if ng <= 0:
            info = "Atom %4d %3s: nao = % 4d, %6d -> %4d, err = % 6.4e" % (
                ia, sym, nao, weigh.size, 0, 0.0
            )
            return coord, weigh, info

        nip = ng if c_isdf is None else int(c_isdf) * nao
        nip = min(nip, ng)

        phi  = self._eval_gto(coord, weigh).real
        phi4 = numpy.dot(phi, phi.T) ** 2

        from pyscf.lib import pivoted_cholesky
        chol, perm, rank = pivoted_cholesky(phi4, tol=-1.0, lower=False)
        nip = min(30 * nao, ng)
        # nip = 
        err = chol[nip-1, nip-1]

        mask = perm[:nip]
        info = "Atom %4d %3s: nao = % 4d, %6d -> %4d, err = % 6.4e" % (
            ia, sym, nao, weigh.size, nip, err
        )

        return coord[mask], weigh[mask], info
    
    def dump_flags(self):
        log = logger.new_logger(self, self.verbose)
        log.info('\n******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('')

    def build(self, *args, **kwargs):
        '''
        Build ISDF grids. The grids are selected by local
        Pivoted Cholesky decomposition.

        If both self.c_isdf and self.tol are None, all grids
        will be used as the interpolating points. Otherwise,
        if self.c_isdf is set, the max number of interpolating
        points for each atom is self.c_isdf * nao.
        '''

        log = logger.new_logger(self, self.verbose)
        c_isdf = kwargs.get("c_isdf", self.c_isdf)
        tol = kwargs.get("tol", self.tol)

        self.dump_flags()
        if c_isdf is None and tol is None:
            log.info('c_isdf and tol are not specified. Using all grids.')
            return self
        
        log.info('\nSet up interpolating points with Pivoted Cholesky decomposition.')
        log.info("c_isdf = %s, tol = %s", c_isdf, tol)
        assert self.coords is not None
        assert self.weights is not None

        coords = []
        weights = []

        cput0 = (logger.process_clock(), logger.perf_counter())
        for ia, mask in enumerate(self._divide(self.coords)):
            tmp = self._select(
                ia, self.coords[mask],
                self.weights[mask],
                c_isdf=c_isdf, tol=tol
                )
            c, w, info = tmp
            coords.append(c)
            weights.append(w)

            log.info(info)

        self.coords  = numpy.vstack(coords)
        log.info('Selected %d interpolating points out of %d', self.coords.shape[0], len(self.weights))
        self.weights = numpy.hstack(weights)
        log.timer("Building Interpolating Points", *cput0)

        return self
    
    def kernel(self, *args, **kwargs):
        return self.build(*args, **kwargs)

class BeckeGrids(InterpolatingPointsMixin, pyscf.dft.gen_grid.Grids):
    _keys = pyscf.dft.gen_grid.Grids._keys | set(["c_isdf", "tol"])

    def __init__(self, mol, *args, **kwargs):
        self.mol = mol
        pyscf.dft.gen_grid.Grids.__init__(self, mol, *args, **kwargs)

    def build(self, *args, **kwargs):
        pyscf.dft.gen_grid.Grids.build(self, *args, **kwargs)
        return super().build(*args, **kwargs)

    def _eval_gto(self, coord, weigh):
        phi = numint.eval_ao(self.mol, coord, deriv=0, shls_slice=None)
        phi *= (numpy.abs(weigh) ** 0.5)[:, None]
        return phi

Grids = BeckeGridsForMolecule = BeckeGrids

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

    grid = Grids(m)
    grid.level   = 0
    grid.verbose = 666

    # Use c_isdf to control the number of interpolating points
    grid.build(c_isdf=10, tol=1e-20)

    # Use tol to control the interpolation error
    grid.build(c_isdf=None, tol=1e-20)

    # Use all grids as the interpolating points
    grid.build(c_isdf=None, tol=None)
