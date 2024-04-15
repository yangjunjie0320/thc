import numpy, pyscf, thc

c = pyscf.pbc.gto.Cell()
c.atom = '''C     0.      0.      0.    
            C     0.8917  0.8917  0.8917
            C     1.7834  1.7834  0.    
            C     2.6751  2.6751  0.8917
            C     1.7834  0.      1.7834
            C     2.6751  0.8917  2.6751
            C     0.      1.7834  1.7834
            C     0.8917  2.6751  2.6751'''
c.basis = '321g'
c.a = numpy.eye(3) * 3.5668
c.unit = 'aa'
c.build()

# Use Becke grid
import thc
thc = thc.LS(c)
thc.with_df = pyscf.pbc.df.GDF(c)
thc.with_df.verbose = 6
thc.verbose = 6
thc.grids.c_isdf = 20
thc.max_memory = 2000
thc.build()

vv = thc.coul
xx = thc.vipt

from pyscf.lib import pack_tril, unpack_tril
df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xx, xx, optimize=True)

df_chol_ref = numpy.zeros_like(df_chol_sol)
a0 = a1 = 0 # slice for auxilary basis
for cderi in thc.with_df.loop(blksize=20):
    a1 = a0 + cderi.shape[0]
    df_chol_ref[a0:a1] = unpack_tril(cderi)
    a0 = a1

err = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
print("Becke grid:   nip = %d, err = %g" % (xx.shape[0], err))

# Use uniform grid
mesh = [10, 10, 10]
thc.grids.coords   = c.gen_uniform_grids(mesh)
thc.grids.weights  = numpy.ones(numpy.prod(mesh))
thc.grids.weights *= c.vol / numpy.prod(mesh)
thc.max_memory = 2000
thc.build()

vv = thc.coul
xx = thc.vipt

from pyscf.lib import pack_tril, unpack_tril
df_chol_sol = numpy.einsum("QI,Im,In->Qmn", vv, xx, xx, optimize=True)

df_chol_ref = numpy.zeros_like(df_chol_sol)
a0 = a1 = 0 # slice for auxilary basis
for cderi in thc.with_df.loop(blksize=20):
    a1 = a0 + cderi.shape[0]
    df_chol_ref[a0:a1] = unpack_tril(cderi)
    a0 = a1

err = numpy.max(numpy.abs(df_chol_ref - df_chol_sol))
print("Uniform grid: nip = %d, err = %g" % (xx.shape[0], err))
