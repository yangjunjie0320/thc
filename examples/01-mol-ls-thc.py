import numpy, pyscf, thc

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

import thc
thc = thc.LS(m)
thc.verbose = 6
thc.grids.level  = 2
thc.grids.c_isdf = 20
thc.max_memory = 2000
thc.build()

vv = thc.coul
xx = thc.vipt

from pyscf.lib import unpack_tril
df_chol_ref = unpack_tril(thc.with_df._cderi)
df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xx, xx, optimize=True)

err = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
print("nip = %d, err = %g" % (xx.shape[0], err))
