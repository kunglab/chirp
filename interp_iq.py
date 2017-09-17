import argparse
import os
import sys

import numpy as np
import chainer
import chainer.functions as F
from chainer import serializers
from chainer import Variable
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import common.net
import load_models

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def slerpn(p0, p1, pts=10, min_t=0.0, max_t=1.0):
    return np.array([slerp(np.squeeze(p0), np.squeeze(p1), i)
                     for i in np.linspace(min_t, max_t, pts)])


parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
args = parser.parse_args()

# download models
if not os.path.exists('_models/DCGANGenerator_iq.npz'):
    print('Downloading models...')
    load_models.download_iq_models()

sample_width = 512
n_hidden = 2
generator = common.net.DCGANGenerator(n_hidden=n_hidden, bottom_width=sample_width/8)
discriminator = common.net.WGANDiscriminator(bottom_width=sample_width/8)
#serializers.load_npz('_models/DCGANGenerator_iq.npz', generator)
#serializers.load_npz('_models/WGANDiscriminator_iq.npz', discriminator)
serializers.load_npz('results/iq_512/DCGANGenerator_10000.npz', generator)
serializers.load_npz('results/iq_512/WGANDiscriminator_10000.npz', discriminator)
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()

xp = generator.xp
# interpolate between two randomly sampled points
z = generator.make_hidden(64)
z = slerpn(z[0], z[1], pts=128)
z = Variable(xp.asarray(z))
x = generator(z)
z = chainer.cuda.to_cpu(xp.asarray(z.data))
x = chainer.cuda.to_cpu(x.data)
x = x.reshape(x.shape[0], x.shape[2], x.shape[3])

fig = plt.figure()
ax = plt.axes()
line1, = plt.plot(x[0, 0], '-x', animated=True)
#line2, = plt.plot(x[0, 1], '-+', animated=True)

def init():
    print line1
    line1.set_ydata(x[0, 0])
    #line2.set_ydata(x[0, 1])
    return line1, #line2

def func(i):
    print 'frame: ', i
    # 'pause' animation at beginning and end
    if i < 5:
        i = 0
    if i >= len(z):
        i = len(z) - 1

    line1.set_ydata(x[i, 0])
    #line2.set_ydata(x[i, 1])
    return line1, #line2


if not os.path.exists('figures'):
    os.makedirs('figures')

#plt.axis('off')
plt.tight_layout()
ani = animation.FuncAnimation(fig, func, init_func=init, frames=np.arange(len(z))+10, interval=100, blit=True)
ani.save('figures/interp_iq_z.gif', dpi=80, writer='imagemagick')