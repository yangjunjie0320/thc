import pyscf, numpy, scipy
import scipy.linalg

import pyscf
from pyscf.lib import logger
from pyscf.dft import numint
import pyscf.dft.gen_grid

from pyscf.dft import gen_grid

def divide(xx: numpy.ndarray, yy: numpy.ndarray, batch_size: int = 2000) -> numpy.ndarray:
    """
    Calculate the Euclidean distances between each point in xx and each point in yy,
    and divide the points in yy into nx groups based on their distances to the points in xx.
    
    Parameters:
    xx (numpy.ndarray): An array of shape (nx, 3) representing nx points in 3D space.
    yy (numpy.ndarray): An array of shape (ny, 3) representing ny points in 3D space.
    
    Returns:
    list: A list of nx arrays, each containing the points in yy that are closest to the corresponding point in xx.
    """
    nx, ny = xx.shape[0], yy.shape[0]
    assert xx.shape == (nx, 3)
    assert yy.shape == (ny, 3)
    assert nx < ny

    d = numpy.linalg.norm(xx[:, numpy.newaxis, :] - yy[numpy.newaxis, :, :], axis=2)
    assert d.shape == (nx, ny)

    ind = numpy.argmin(d, axis=0)
    return [numpy.where(ind == ix)[0] for ix in range(nx)]

def select(mol=None, ia=0, coord=None, weigh=None, c_isdf=10, tol=1e-10):
    sym = mol.atom_symbol(ia)
    nao = (lambda s: s[3] - s[2])(mol.aoslice_by_atom()[ia])
    
    ng = coord.shape[0]
    assert coord.shape == (ng, 3)
    assert weigh.shape == (ng,)

    nip = int(c_isdf * nao)
    nip = min(nip, ng)

    phi  = numint.eval_ao(mol, coord, deriv=0, shls_slice=None)
    phi *= (numpy.abs(weigh) ** 0.5)[:, None]
    phi4 = numpy.dot(phi, phi.T) ** 2

    from pyscf.lib import pivoted_cholesky
    chol, perm, rank = pivoted_cholesky(phi4, tol=tol, lower=False)
    nip = min(nip, rank)
    err = chol[nip-1, nip-1] # / chol[0, 0]

    mask = perm[:nip]
    info = "Atom %4d %3s: nao = % 4d, %6d -> %4d, err = % 6.4e" % (
        ia, sym, nao, weigh.size, nip, err
    )

    return coord[mask], weigh[mask], (ia, info)

class InterpolatingPoints(pyscf.dft.gen_grid.Grids):
    _keys = pyscf.dft.gen_grid.Grids._keys | set([
        'c_isdf', 'tol', 'batch_size'
    ])

    def __init__(self, *args, **kwargs):
        self.c_isdf = 10
        self.tol = 1e-20
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        '''
        Build ISDF grids.
        '''
        super().build(*args, **kwargs)

        log = logger.new_logger(self, self.verbose)
        if self.c_isdf is not None:
            log.info('\nSelecting interpolating points with Pivoted Cholesky decomposition.')
            log.info('c_isdf = %d', self.c_isdf)
        else:
            log.info('No c_isdf is specified. Using all grids.')
            return self
        
        mol = self.mol

        coords = []
        weights = []

        cput0 = (logger.process_clock(), logger.perf_counter())
        for ia, m in enumerate(divide(mol.atom_coords(), self.coords)):
            tmp = select(
                mol, ia, self.coords[m], self.weights[m],
                self.c_isdf, self.tol
                )
            coords.append(tmp[0])
            weights.append(tmp[1])
            log.info(tmp[2][1])

        log.timer("Building Interpolating Points", *cput0)
        self.coords  = numpy.vstack(coords)
        self.weights = numpy.hstack(weights)
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
    grid.level   = 0
    grid.verbose = 6
    grid.c_isdf  = 30
    grid.kernel()
