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
from load_models import download_trained_models

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
if not os.path.exists('_models/DCGANGenerator.npz'):
    print('Downloading models...')
    download_trained_models()

generator = common.net.DCGANGenerator()
discriminator = common.net.WGANDiscriminator()
serializers.load_npz('_models/DCGANGenerator.npz', generator)
serializers.load_npz('_models/WGANDiscriminator.npz', discriminator)
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
x = x.reshape(len(x), -1)

fig = plt.figure()
ax = plt.axes()
line, = plt.plot(x[0], '-x', animated=True)

def init():
    line.set_ydata(x[0])
    return line,

def func(i):
    print 'frame: ', i
    # 'pause' animation at beginning and end
    if i < 5:
        i = 0
    if i >= len(z):
        i = len(z) - 1

    line.set_ydata(x[i])
    return line,


if not os.path.exists('figures'):
    os.makedirs('figures')

#plt.axis('off')
plt.tight_layout()
ani = animation.FuncAnimation(fig, func, init_func=init, frames=np.arange(len(z))+10, interval=100, blit=True)
ani.save('figures/interp_z.gif', dpi=300, writer='imagemagick')