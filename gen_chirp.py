from tftb.generators import amgauss, fmlin
from tftb.processing import inst_freq
import matplotlib.pyplot as plt
import numpy as np

# plt.figure()

# num_amps = 3.
# amps = np.linspace(1./num_amps, 1., num_amps)
# num_samp = 2**13
# z = fmlin(num_samp, 0.01, .1)[0]
# for amp in amps[::-1]:

#     # creates nump_samples from frequency range .01 to .1 - this 
#     z_amp = z * amp
#     plt.figure(1)
#     # The real part is all we need to save, np.real(z) turns only the real portion of the signal
#     plt.plot(np.real(z_amp), 'x-')


# plt.show()

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

# plt.figure(2)
# r = inst_freq(z_amp)[0]
# plt.plot(r, 'x-')