import argparse
import os
import sys
from functools import partial

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
if not os.path.exists('_models/DCGANGenerator.npz'):
    print('Downloading models...')
    load_models.download_models()

def make_amp_hidden(n_hidden, batchsize):
    zs = np.random.randn(batchsize, n_hidden, 1, 1).astype(np.float32)
    amp_zs = np.random.uniform(0.2, 1.0, (batchsize, 1, 1, 1)).astype(np.float32)
    return np.concatenate((zs, amp_zs), axis=1)


chainer.config.train = False
chainer.config.enable_backprop = False
sample_width=256
n_hidden=5
make_hidden_f = partial(make_amp_hidden, n_hidden)
generator = common.net.DCGANGenerator(make_hidden_f, n_hidden=n_hidden+1, bottom_width=sample_width/8)
discriminator = common.net.LabeledDiscriminator(bottom_width=sample_width/8)
serializers.load_npz('results/amp_labeled/DCGANGenerator_30000.npz', generator)
serializers.load_npz('results/amp_labeled/LabeledDiscriminator_30000.npz', discriminator)
if args.gpu >= 0:
    generator.to_gpu()
    discriminator.to_gpu()

xp = generator.xp
# interpolate between two randomly sampled points
z = generator.make_hidden(64)
z[:] = z[0, np.newaxis]
z = z.reshape(z.shape[0], z.shape[1])
#z = np.array([(t*z[0] + (1-t)*z[1])/2. for t in np.linspace(0, 1, 128)])
#z = z.reshape(z.shape[0], z.shape[1])
#z = slerpn(z[0], z[1], pts=64)
amp_real = np.concatenate((np.linspace(0.2, 1, 32), np.linspace(0.2, 1, 32)[::-1]))
z[:, -1] = amp_real
#z[:, -1] = 0.2
z = z.reshape(z.shape[0], z.shape[1], 1, 1)
z = xp.asarray(z)
#z = Variable(xp.asarray(z))
#x = generator(z)
x = generator(Variable(z))
#x = gen_per_sample(z, generator)
y_status, y_pred = discriminator(x)
y_pred.to_cpu()
y_pred = y_pred.data.flatten()
x.to_cpu()
x = x.data
x = x.reshape(len(x), -1)
#z = chainer.cuda.to_cpu(xp.asarray(z.data))
#x = chainer.cuda.to_cpu(x.data)

plt.plot(chainer.cuda.to_cpu(z[:, -1]).flatten(), label='input')
plt.plot([np.mean(np.sqrt(xi[:256]**2 + xi[256:]**2)) for xi in x], label='generated')
plt.plot(y_pred, label='predicted')
plt.legend(loc=0)
plt.savefig('figures/amp_pred.png')

fig = plt.figure()
ax = plt.axes()
line, = plt.plot(x[0][:256], '-x', animated=True)
line2, = plt.plot(x[0][256:], '-x', animated=True)
plt.ylim((-1, 1))

def init():
    line.set_ydata(x[0][:256])
    line2.set_ydata(x[0][256:])
    return line, line2

def func(i):
    print 'frame: ', i
    # 'pause' animation at beginning and end
    if i < 5:
        i = 0
    if i >= len(z):
        i = len(z) - 1

    line.set_ydata(x[i][:256])
    line2.set_ydata(x[i][256:])
    return line, line2


if not os.path.exists('figures'):
    os.makedirs('figures')

#plt.axis('off')
plt.tight_layout()
ani = animation.FuncAnimation(fig, func, init_func=init, frames=np.arange(len(z))+10, interval=100, blit=True)
ani.save('figures/interp_z.gif', dpi=50, writer='imagemagick')