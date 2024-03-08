import pyscf

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

df = pyscf.df.DF(m)
df.max_memory = 500
df.auxbasis = "weigend"
df.build()

p0 = 0
for cderi in df.loop(blksize=10):
    p1 = p0 + cderi.shape[0]
    print(p0, p1, cderi.shape)