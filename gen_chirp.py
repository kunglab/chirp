from tftb.generators import amgauss, fmlin
from tftb.processing import inst_freq
import matplotlib.pyplot as plt
import numpy as np



## all zs - len = num_amps
num_amps = 3.
amps = np.linspace(1./num_amps, 1., num_amps)
num_samp = 2**13
z = fmlin(num_samp, 0.01, .1)[0]
zs = np.array([z*amp for amp in amps[::-1]])

for z_amp in zs:
    plt.figure(1)
    plt.plot(np.real(z_amp), 'x-')

plt.show()

# useful code - saving
# plt.figure(2)
# r = inst_freq(z_amp)[0]
# plt.plot(r, 'x-')