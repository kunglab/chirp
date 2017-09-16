from tftb.generators import amgauss, fmlin
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
num_samp = 2**13
# creates nump_samples from frequency range .01 to .1 - this 
z = fmlin(num_samp, 0.01, .1)[0]

# The real part is all we need to save, np.real(z) turns only the real portion of the signal
plt.plot(np.real(z), 'x-')
plt.title("Linear Frequency Modulation")
plt.show()


r = inst_freq(z)[0]
print r